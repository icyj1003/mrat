import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *

from tqdm import tqdm
from common import Actor, Critic


from typing import *

import torch


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories for PPO.
    """

    def __init__(self, device="cpu"):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.violations = []
        self.device = device

    def add(
        self,
        state,
        mask,
        action,
        log_prob,
        reward,
        done,
        violations,
    ):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.violations.append(violations)

    def get(self):
        return (
            torch.stack(self.states).to(self.device),
            torch.stack(self.masks).to(self.device),
            torch.stack(self.actions).to(self.device),
            torch.stack(self.log_probs).to(self.device),
            torch.tensor(self.rewards).to(self.device),
            torch.tensor(self.dones).to(self.device),
            torch.tensor(self.violations).to(self.device),
        )

    def clear(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.violations = []

    def __len__(self):
        return len(self.states)


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        state_dim,
        num_action,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        num_epoch=10,
        eps=0.2,
        gamma=0.99,
        lam=0.95,
        mini_batch_size=64,
        tau=1e-3,
        entropy_coeff=0.01,
        lambd_init=0.1,
        device="cpu",
        writer=None,
    ):

        self.state_dim = state_dim
        self.num_action = num_action
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epoch = num_epoch
        self.eps = eps
        self.gamma = gamma
        self.lam = lam
        self.mini_batch_size = mini_batch_size
        self.tau = tau
        self.entropy_coeff = entropy_coeff
        self.device = device
        self.writer = writer
        self.steps = 0

        self.actor = Actor(state_dim, num_action, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_c = Critic(state_dim, hidden_dim).to(device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_c_target = copy.deepcopy(self.critic_c)

        self.lambd = nn.Parameter(torch.tensor(lambd_init)).to(
            device
        )  # Lagrange multiplier for constraint violation

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_c_optimizer = torch.optim.Adam(self.critic_c.parameters(), lr=lr)
        self.lambd_optimizer = torch.optim.Adam([self.lambd], lr=lr)

        self.buffer = RolloutBuffer(
            device=device,
        )

    def evaluate(self, states, masks, actions):
        """
        Compute log probabilities of actions given states and masks.
        Args:
            states (torch.Tensor): Current states.
            masks (torch.Tensor): Action masks.
            actions (torch.Tensor): Actions taken.
        Returns:
            torch.Tensor: Log probabilities of actions.
        """
        # Compute logits from the actor network
        logits = self.actor(states, masks)  # [batch_size, num_action, action_dim]

        # Reshape for Categorical: merge batch and action dims
        logits_flat = logits.view(
            -1, self.action_dim
        )  # [(batch * num_action), action_dim]

        actions_flat = actions.view(-1)  # [(num_action * action_dim), 1]

        # Sample actions
        dists = torch.distributions.Categorical(
            logits=logits_flat
        )  # each row is a [logit_0, logit_1]

        log_probs = dists.log_prob(actions_flat).view(-1, self.num_action)
        entropy = dists.entropy().view(-1, self.num_action)

        values = self.critic(states).squeeze(-1)
        values_c = self.critic_c(states).squeeze(-1)

        return (values, values_c, log_probs, entropy)

    def act(self, state, mask, projection=None):
        """
        Sample an action from the actor network.
        Args:
            state (torch.Tensor): Current state.
            mask (torch.Tensor): Action mask.
        Returns:
            torch.Tensor: Sampled action.
        """

        # Add batch dimension
        state = state.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        # Compute logits from the actor network
        logits = self.actor(state, mask)  # [1, num_action, action_dim]

        logits_flat = logits.view(
            -1, self.action_dim
        )  # [(1 * num_action), self.action_dim]

        # Sample actions
        dists = torch.distributions.Categorical(logits=logits_flat)

        action = dists.sample()  # shape: [(num_action * action_dim)]

        # project raw sample to valid action space
        if projection is not None:
            action = projection(action)

        log_prob = dists.log_prob(action)

        return action.detach().squeeze(), log_prob.detach().squeeze()

    def gae(self, signals, values, dones, normalize=True):
        """
        Compute Generalized Advantage Estimation (GAE) for the given signals.
        Args:
            signals (torch.Tensor): Immediate rewards or costs.
            values (torch.Tensor): Value estimates (V(s)).
            dones (torch.Tensor): Done flags (1 if terminal, 0 otherwise).
        Returns:
            torch.Tensor: Returns and GAE advantages.
        """
        # Pad values and dones by appending one step (bootstrap value and non-terminal flag)
        values = torch.cat([values, torch.zeros_like(values[-1:])], dim=0)
        dones = torch.cat(
            [dones, torch.ones_like(dones[-1:])], dim=0
        )  # Assume terminal

        advantages = torch.zeros_like(signals)
        returns = torch.zeros_like(signals)

        gae = 0.0
        for t in reversed(range(len(signals))):
            delta = (
                signals[t]
                + self.gamma * values[t + 1] * (1.0 - dones[t + 1].float())
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1.0 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.detach(), advantages.detach()

    def process_batch(self, batch):
        """
        Process a single batch during the PPO update.
        """
        (
            batch_states,
            batch_masks,
            batch_actions,
            batch_log_probs,
            batch_returns,
            batch_returns_c,
            batch_advantages,
        ) = batch

        # Compute new log prob and values
        new_values, new_values_c, new_log_probs, entropy = self.evaluate(
            batch_states, batch_masks, batch_actions
        )

        # Compute PPO ratio
        ratios = torch.exp(new_log_probs.sum(dim=-1) - batch_log_probs.sum(dim=-1))

        # Compute surrogate loss
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * batch_advantages

        # Compute actor loss
        obj_surr = (-torch.min(surr1, surr2)).mean()
        entropy_loss = -entropy.mean() * self.entropy_coeff
        actor_loss = obj_surr + entropy_loss

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(new_values, batch_returns)

        # Compute critic_c loss
        critic_c_loss = F.smooth_l1_loss(new_values_c, batch_returns_c)

        return actor_loss, entropy_loss, critic_loss, critic_c_loss

    def update(self):
        """
        Update the actor and critic networks using the PPO algorithm.
        """
        # Get data from the buffer
        (
            states,
            masks,
            actions,
            log_probs,
            rewards,
            dones,
            violations,
        ) = self.buffer.get()

        values = self.critic_target(states).squeeze(-1)
        values_c = self.critic_c_target(states).squeeze(-1)

        # Compute discounted rewards returns and advantages
        returns, advantages = self.gae(rewards, values, dones)  # [rollout_len]

        # Compute discounted constraints violations returns and advantages
        returns_c, advantages_c = self.gae(
            violations,
            values_c,
            dones,
        )  # [rollout_len]

        advantages = (
            (advantages - self.lambd * advantages_c) / (self.lambd + 1)
        ).detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare data for mini-batch training
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                states,
                masks,
                actions,
                log_probs,
                returns,
                returns_c,
                advantages,
            ),
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        num_batches = len(data_loader)

        for _ in tqdm(range(self.num_epoch), desc="PPO Update", leave=False):
            # Initialize average losses
            avg_actor_loss, avg_entropy_loss, avg_critic_loss, avg_critic_c_loss = (
                0.0,
                0.0,
                0.0,
                0.0,
            )

            for batch in tqdm(data_loader, desc="Mini-batch Update", leave=False):
                # Process batch
                actor_loss, entropy_loss, critic_loss, critic_c_loss = (
                    self.process_batch(batch)
                )

                # Accumulate losses
                avg_actor_loss += actor_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()
                avg_critic_c_loss += critic_c_loss.item()

                # Backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.critic_c_optimizer.zero_grad()
                critic_c_loss.backward()
                self.critic_c_optimizer.step()

                # Soft update target networks
                self.soft_update()

            # Normalize average losses
            avg_actor_loss /= num_batches
            avg_entropy_loss /= num_batches
            avg_critic_loss /= num_batches
            avg_critic_c_loss /= num_batches

            # Log average losses
            if self.writer is not None:
                self.writer.add_scalar(
                    "loss/avg_actor_loss", avg_actor_loss, self.steps
                )
                self.writer.add_scalar(
                    "loss/avg_critic_loss", avg_critic_loss, self.steps
                )
                self.writer.add_scalar(
                    "loss/avg_entropy_loss", avg_entropy_loss, self.steps
                )
                self.writer.add_scalar(
                    "loss/avg_critic_c_loss", avg_critic_c_loss, self.steps
                )

            self.steps += 1

        # Update lambda
        # Log lambda
        if self.writer is not None:
            self.writer.add_scalar("log/lambda", self.lambd.item(), self.steps)

        # Mean violation
        mean_violations = torch.mean(violations).detach()

        # Dual loss
        lagrange_loss = -self.lambd * mean_violations

        # Optimize
        self.lambd_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lambd_optimizer.step()

        # Clip lambda
        with torch.no_grad():
            self.lambd.clamp_(min=0.0)

    def soft_update(self):
        """
        Soft update the target networks.
        """
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_c_target.parameters(), self.critic_c.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def add(self, state, mask, action, log_prob, reward, done, violation):
        """
        Add a transition to the buffer.
        """
        self.buffer.add(state, mask, action, log_prob, reward, done, violation)

    def buffer_length(self):
        """
        Get the length of the buffer.
        """
        return len(self.buffer)
