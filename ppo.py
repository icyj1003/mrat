import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *

from tqdm import tqdm
from buffer import RolloutBuffer
from common import Actor, Critic


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        state_dim,
        num_action,
        action_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        lagrange_multiplier_dim,
        actor_lr,
        critic_lr,
        lagrange_lr,
        num_epoch,
        eps=0.2,
        gamma=0.99,
        lam=0.95,
        mini_batch_size=64,
        device="cpu",
        writer=None,
    ):
        self.actor = Actor(state_dim, num_action, action_dim, actor_hidden_dim).to(
            device
        )
        self.critic = Critic(state_dim, num_action, critic_hidden_dim).to(device)

        self.lagrange_multiplier = torch.zeros(
            lagrange_multiplier_dim, requires_grad=False, device=device
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer(
            device=device,
        )

        self.lagrange_lr = lagrange_lr
        self.gamma = gamma
        self.eps = eps
        self.lam = lam
        self.num_epoch = num_epoch
        self.mini_batch_size = mini_batch_size

        self.state_dim = state_dim
        self.num_action = num_action
        self.action_dim = action_dim
        self.lagrange_multiplier_dim = lagrange_multiplier_dim

        self.device = device
        self.writer = writer
        self.current_step = 0

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

        values = self.critic(states, actions).squeeze(-1)

        return (values, log_probs, entropy)

    def act(self, state, mask):
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

        log_prob = dists.log_prob(action)

        value = self.critic(state, action)

        return (
            action.detach().squeeze(),
            log_prob.detach().squeeze(),
            value.detach().squeeze(),
        )

    def gae_signal(self, states, signals, values, actions, dones, normalize=True):
        """
        Compute Generalized Advantage Estimation (GAE) for the given signals.
        Args:
            states (torch.Tensor): Current states.
            signals (torch.Tensor): Signals for GAE.
            values (torch.Tensor): Value estimates.
            actions (torch.Tensor): Actions taken.
            dones (torch.Tensor): Done flags.
        Returns:
            torch.Tensor: GAE for the signals.
        """
        advantages = torch.zeros_like(signals)
        returns = torch.zeros_like(signals)

        next_value = (
            self.critic(states[-1].unsqueeze(0), actions[-1].unsqueeze(0))
            .squeeze(0)
            .detach()
        )
        gae = 0.0
        for t in reversed(range(len(signals))):
            delta = (
                signals[t]
                + self.gamma * next_value * (1.0 - dones[t].float())
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1.0 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        # Normalize advantages
        if normalize:
            if advantages.dim() == 1:  # Case 1: shape [batch]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            elif advantages.dim() == 2:  # Case 2: shape [batch, dim]
                advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (
                    advantages.std(dim=0, keepdim=True) + 1e-8
                )

        return returns, advantages

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
            batch_advantages,
        ) = batch

        # Compute new log prob and values
        new_values, new_log_probs, entropy = self.evaluate(
            batch_states, batch_masks, batch_actions
        )

        # Compute PPO ratio
        ratios = torch.exp(new_log_probs.sum(dim=-1) - batch_log_probs.sum(dim=-1))

        # Compute surrogate loss
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * batch_advantages

        # Compute actor loss
        main_actor_loss = (-torch.min(surr1, surr2)).mean()
        entropy_loss = entropy.mean()
        actor_loss = main_actor_loss - entropy_loss * 1e-2

        # Compute critic loss
        critic_loss = F.mse_loss(new_values, batch_returns)

        return actor_loss, entropy_loss, critic_loss

    def update(self):
        """
        Update the actor and critic networks using the PPO algorithm.
        """
        # Get data from the buffer
        states, masks, actions, log_probs, values, rewards, dones, violations = (
            self.buffer.get()
        )

        # Compute discounted rewards returns and advantages
        returns, reward_advantages = self.gae_signal(
            states, rewards, values, actions, dones
        )  # [rollout_len]

        # Compute discounted constraints violations returns and advantages
        _, violation_advantages = self.gae_signal(
            states, violations, values, actions, dones
        )  # [rollout_len, lagrange_multiplier_dim]

        advantages = reward_advantages - torch.matmul(
            violation_advantages, self.lagrange_multiplier
        )

        # Prepare data for mini-batch training
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                states,
                masks,
                actions,
                log_probs,
                returns,
                advantages,
            ),
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        num_batches = len(data_loader)

        for _ in tqdm(range(self.num_epoch), desc="PPO Update", leave=False):
            # Initialize average losses
            avg_actor_loss, avg_entropy_loss, avg_critic_loss = (
                0.0,
                0.0,
                0.0,
            )

            for batch in tqdm(data_loader, desc="Mini-batch Update", leave=False):
                # Process batch
                actor_loss, entropy_loss, critic_loss = self.process_batch(batch)

                # Accumulate losses
                avg_actor_loss += actor_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()

                # Backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Normalize average losses
            avg_actor_loss /= num_batches
            avg_entropy_loss /= num_batches
            avg_critic_loss /= num_batches

            # Log average losses
            if self.writer is not None:
                self.writer.add_scalar(
                    "loss/avg_actor_loss", avg_actor_loss, self.current_step
                )
                self.writer.add_scalar(
                    "loss/avg_critic_loss", avg_critic_loss, self.current_step
                )
                self.writer.add_scalar(
                    "loss/avg_entropy_loss", avg_entropy_loss, self.current_step
                )

            self.current_step += 1

        # Update Lagrange multipliers
        mean_violations = violations.mean(dim=0)
        self.lagrange_multiplier += self.lagrange_lr * (
            mean_violations - self.lagrange_multiplier
        )

        # Log Lagrange multipliers
        if self.writer is not None:
            for i in range(self.lagrange_multiplier_dim):
                self.writer.add_scalar(
                    f"lambda/lambda_{i}",
                    self.lagrange_multiplier[i],
                    self.current_step,
                )

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def add(self, state, mask, action, log_prob, values, reward, done, violation):
        """
        Add a transition to the buffer.
        """
        self.buffer.add(state, mask, action, log_prob, values, reward, done, violation)

    def buffer_length(self):
        """
        Get the length of the buffer.
        """
        return len(self.buffer)
