import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *

from tqdm import tqdm
from buffer import RolloutBuffer
from common import Actor, Critic


from typing import *

import torch


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        state_dim,
        num_actions,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        num_epoch=10,
        eps=0.2,
        gamma=0.99,
        lam=0.95,
        tau=1e-3,
        mini_batch_size=64,
        entropy_coeff=0.01,
        penalty_coeff=1,
        penalty_lr=1e-3,
        device="cpu",
        writer=None,
    ):

        # Hyperparameters
        self.state_dim = state_dim
        self.num_actions = num_actions
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
        self.penalty_coeff = penalty_coeff
        self.penalty_lr = penalty_lr
        self.device = device
        self.writer = writer
        self.steps = 0

        # define networks
        self.actor = Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer(
            device=device,
        )
        self.global_step = 0

    def act(self, state, mask, projection=None):
        """
        Sample an action from the actor network given the current state and mask.
        Args:
            state (torch.Tensor): Current state.
            mask (torch.Tensor): Action mask.
            projection (callable, optional): Function to project raw actions to valid action space.
        Returns:
            torch.Tensor: Sampled action.
            torch.Tensor: Log probability of the sampled action.
        """

        # Add batch dimension
        state = state.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        # Compute logits from the actor network
        logits = self.actor(state, mask).squeeze(
            0
        )  # [1, num_actions, action_dim] --> [num_actions, action_dim]

        # Sample actions
        dists = torch.distributions.Categorical(logits=logits)

        actions = dists.sample().detach()  # shape: [(num_actions * action_dim)]

        # project raw sample to valid action space
        if projection is not None:
            valid_actions = projection(actions)
        else:
            valid_actions = actions

        log_probs = dists.log_prob(valid_actions).detach()

        return valid_actions, log_probs

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
        logits = self.actor(states, masks)  # [batch_size, num_actions, action_dim]

        # Reshape for Categorical: merge batch and action dims
        logits_flat = logits.view(
            -1, self.action_dim
        )  # [(batch * num_actions), action_dim]

        actions_flat = actions.view(-1)  # [(num_actions * action_dim), 1]

        # Sample actions
        dists = torch.distributions.Categorical(
            logits=logits_flat
        )  # each row is a [logit_0, logit_1]

        log_probs = dists.log_prob(actions_flat).view(-1, self.num_actions)
        entropy = dists.entropy().view(-1, self.num_actions)

        return log_probs, entropy

    def gae(self, signals, values, dones, normalize=True):
        """
        Generalized Advantage Estimation (GAE).
        Args:
            signals: (T, ...) Tensor of signals (rewards, violations, etc.)
            values: (T+1, ...) Value estimates (bootstrap included)
            dones: (T, ...) Episode done flags (1 if done, else 0)
        Returns:
            advantages: (T, ...) Advantage estimates
            returns: (T, ...) Target values
        """
        # Ensure the same of values and signals
        assert signals.shape[0] == values.shape[0] - 1 == dones.shape[0]

        T = signals.shape[0]
        advantages = torch.zeros_like(signals)
        last_adv = 0

        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            delta = signals[t] + self.gamma * values[t + 1] * not_done - values[t]
            advantages[t] = last_adv = (
                delta + self.gamma * self.lam * not_done * last_adv
            )

        returns = advantages + values[:-1]
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """
        Update the actor and critic networks using the PPO algorithm.
        """
        # Get the data from the buffer
        (
            states,
            masks,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
        ) = self.buffer.get()

        # prepare values for GAE
        values = self.critic(states).detach()
        next_values = self.critic_target(next_states).detach()
        values = torch.cat((values, next_values[-1].unsqueeze(0)), dim=0)

        # lagrangian penalty
        rewards = rewards - self.penalty_coeff * violations

        # compute advantages and returns
        advantages, returns = self.gae(rewards, values, dones, normalize=True)

        # create mini-batches
        dataset = torch.utils.data.TensorDataset(
            states,
            masks,
            actions,
            log_probs,
            advantages,
            returns,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        # Update the actor and critic networks
        for _ in tqdm(range(self.num_epoch), desc="Epochs"):
            avg_actor_loss, avg_entropy_loss, avg_critic_loss = (
                0.0,
                0.0,
                0.0,
            )
            for (
                state_batch,
                mask_batch,
                action_batch,
                old_log_probs_batch,
                advantage_batch,
                return_batch,
            ) in tqdm(dataloader, desc="PPO minibatch"):
                # Update the actor network
                self.actor_optimizer.zero_grad()

                # Compute new log probabilities
                new_log_probs, entropy = self.evaluate(
                    state_batch, mask_batch, action_batch
                )

                # Compute the ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # Compute the surrogate loss
                surrogate_loss = -torch.min(
                    ratio * advantage_batch,
                    torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage_batch,
                ).mean()

                # Compute the entropy loss
                entropy_loss = self.entropy_coeff * entropy.mean()

                # Total loss
                actor_loss = surrogate_loss + entropy_loss

                # Backpropagation
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # Update the critic network
                self.critic_optimizer.zero_grad()

                # Compute the critic loss
                critic_loss = F.smooth_l1_loss(self.critic(state_batch), return_batch)

                # Backpropagation
                critic_loss.backward()
                self.critic_optimizer.step()

                # Update the target network
                self.soft_update()

                # Accumulate losses
                avg_actor_loss += actor_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()

            # Average losses over the mini-batches
            avg_actor_loss /= len(dataloader)
            avg_entropy_loss /= len(dataloader)
            avg_critic_loss /= len(dataloader)
            self.writer.add_scalar("actor_loss", avg_actor_loss, self.global_step)
            self.writer.add_scalar("entropy_loss", avg_entropy_loss, self.global_step)
            self.writer.add_scalar("critic_loss", avg_critic_loss, self.global_step)

            self.global_step += 1

        # update penalty coefficient
        self.penalty_coeff += self.penalty_lr * (violations.mean()).detach()
        self.penalty_coeff = max(0, min(self.penalty_coeff, 1))

        # clear the buffer
        self.buffer.clear()

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
