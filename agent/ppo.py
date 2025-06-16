import copy
from typing import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

from agent.buffer import RolloutBuffer
from agent.common import Actor, Critic, TransformerActor, TransformerCritic


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        name,
        num_actions,
        action_dim,
        state_dim,
        hidden_dim=64,
        lr=3e-4,
        num_epochs=10,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        tau=1e-3,
        mini_batch_size=64,
        vf_coeff=0.5,
        entropy_coeff=0.01,
        penalty_coeff=1.0,
        penalty_lr=1e-3,
        max_grad_norm=0.5,
        device="cpu",
        writer=None,
        use_lagrange=True,
    ):

        self.name = name

        # Hyperparameters
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.tau = tau
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.penalty_coeff = penalty_coeff
        self.penalty_lr = penalty_lr
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.writer = writer
        self.use_lagrange = use_lagrange

        # define networks
        self.actor = TransformerActor(
            input_dim=self.state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            action_dim=action_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
        )

        self.buffer = RolloutBuffer(
            device=device,
        )
        self.global_step = 0

    def act(self, state, mask, projection=None):
        # Add batch dimension
        state = state.unsqueeze(0).to(self.device)
        if mask is not None:
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

        # lagrangian penalty
        if self.use_lagrange:
            # apply penalty to rewards
            rewards = rewards - self.penalty_coeff * violations

        # compute advantages and returns
        # advantages, returns = self.gae(rewards, values, dones, normalize=True)

        # dont use gae
        advantages = rewards

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # create mini-batches
        dataset = torch.utils.data.TensorDataset(
            states,
            masks,
            actions,
            log_probs,
            advantages,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        # Update the actor and critic networks
        for _ in tqdm(range(self.num_epochs), desc="Epochs"):
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
            ) in tqdm(dataloader, desc="PPO minibatch"):
                # Compute new log probabilities
                new_log_probs, entropy = self.evaluate(
                    state_batch, mask_batch, action_batch
                )

                # Compute the ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # Compute the surrogate loss
                surr1 = ratio * advantage_batch
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * advantage_batch
                )

                # Surrogate loss
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # Compute the entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = surrogate_loss + self.entropy_coeff * entropy_loss

                # update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Accumulate losses
                avg_actor_loss += surrogate_loss.item()
                avg_entropy_loss += entropy_loss.item()

            # Average losses over the mini-batches
            avg_actor_loss /= len(dataloader)
            avg_entropy_loss /= len(dataloader)

            # Log the losses
            if self.writer is not None:
                self.writer.add_scalar(
                    f"{self.name}_loss/actor_loss", avg_actor_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/entropy_loss", avg_entropy_loss, self.global_step
                )

            self.global_step += 1

        # update penalty coefficient
        if self.use_lagrange:
            self.penalty_coeff += self.penalty_lr * (violations.mean()).detach()
            self.penalty_coeff = max(0, min(self.penalty_coeff, 1))

        # clear the buffer
        self.buffer.clear()
