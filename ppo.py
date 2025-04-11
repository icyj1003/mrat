import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *


class Actor(nn.Module):
    """
    Actor network for Gumbel-Sigmoid sampling.
    Args:
        state_dim (int): Dimension of the state.
        num_action (int): Number of actions.
        action_dim (int): Dimension of each action.
        hidden_dim (int): Hidden dimension of the network.
        activation (str): Activation function to use ("relu" or "tanh").
    """

    def __init__(
        self,
        state_dim: int,
        num_action: int,
        action_dim: int,
        hidden_dim: int = 128,
        activation: Literal["relu", "tanh"] = "relu",
    ):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, num_action * action_dim),
        )
        self.num_action = num_action
        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self, state, logits_mask):
        x = self.fc(state)
        x = x + logits_mask.view(-1, self.num_action * self.action_dim) * 1e9
        return x.view(-1, self.num_action, self.action_dim)


class Critic(nn.Module):
    """
    Critic network for value estimation.
    Args:
        state_dim (int): State dimension.
        num_action (int): Number of actions.
        action_dim (int): Action dimension.
        hidden_dim (int): Hidden layer dimension.
        activation (str): Activation function.
    """

    def __init__(
        self,
        state_dim: int,
        num_action: int,
        action_dim: int,
        hidden_dim: int = 128,
        activation: Literal["relu", "tanh"] = "relu",
    ):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + num_action * action_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.action_dim = action_dim
        self.num_action = num_action

    def forward(self, state, action):
        x = torch.cat(
            [
                state,
                action.view(-1, self.num_action * self.action_dim),
            ],
            dim=-1,
        )
        x = self.fc(x)
        return x


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories for PPO.
    """

    def __init__(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state, mask, action, log_prob, values, reward, done):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.masks),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.tensor(self.rewards),
            torch.tensor(self.dones),
        )

    def clear(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    Args:
        state_dim (int): Dimension of the state.
        num_action (int): Number of actions.
        action_dim (int): Dimension of each action.
        actor_hidden_dim (int): Hidden dimension of the actor network.
        critic_hidden_dim (int): Hidden dimension of the critic network.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
        num_epoch (int): Number of epochs for training.
        eps (float): Clipping parameter for PPO.
        gamma (float): Discount factor for rewards.
        writer: TensorBoard writer for logging.
    """

    def __init__(
        self,
        state_dim,
        num_action,
        action_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        actor_lr,
        critic_lr,
        num_epoch,
        eps=0.2,
        gamma=0.99,
        writer=None,
    ):
        self.actor = Actor(state_dim, num_action, action_dim, actor_hidden_dim)
        self.critic = Critic(state_dim, num_action, action_dim, critic_hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.state_dim = state_dim
        self.num_action = num_action
        self.action_dim = action_dim
        self.num_epoch = num_epoch
        self.eps = eps
        self.writer = writer
        self.current_step = 0

    def get_log_prob(self, states, masks, actions):
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
        logits = self.actor(states, masks)

        dist = torch.distributions.Bernoulli(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy

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
        state = state.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Compute logits from the actor network
        logits = self.actor(state, mask)

        dist = torch.distributions.Bernoulli(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Compute state-action value from the critic network
        value = self.critic(state, action)

        return (
            action.detach().squeeze(),
            log_prob.detach().squeeze(),
            value.detach().squeeze(),
        )

    def compute_discounted_returns(self, rewards, dones):
        """
        Compute discounted returns for the given rewards and dones.
        Args:
            rewards (torch.Tensor): Rewards.
            dones (torch.Tensor): Dones.
        Returns:
            torch.Tensor: Discounted returns.
        """
        returns = torch.zeros_like(rewards)

        R = 0.0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t].float())
            returns[t] = R

        return returns

    def update(self):
        """
        Update the actor and critic networks using the PPO algorithm.
        """
        # Get data from the buffer
        states, masks, actions, log_probs, values, rewards, dones = self.buffer.get()

        # Compute discounted returns
        returns = self.compute_discounted_returns(rewards, dones)

        # Compute advantages
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-10
        )

        for _ in range(self.num_epoch):
            # Compute new log prob
            new_log_probs, entropy = self.get_log_prob(states, masks, actions)

            # Compute new action-state value
            new_values = self.critic(states, actions).squeeze(-1)

            # Compute PPO ratio
            ratios = torch.exp(
                new_log_probs.sum(dim=(1, 2)) - log_probs.sum(dim=(1, 2))
            )

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages

            # Compute actor loss
            main_actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean() * 0.01
            actor_loss = main_actor_loss + entropy_loss

            # Compute critic loss
            critic_loss = F.mse_loss(new_values, returns)

            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar(
                    "loss/actor_loss", actor_loss.item(), self.current_step
                )
                self.writer.add_scalar(
                    "loss/entropy_loss", entropy_loss.item(), self.current_step
                )
                self.writer.add_scalar(
                    "loss/critic_loss", critic_loss.item(), self.current_step
                )

            self.current_step += 1

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def add(self, state, mask, action, log_prob, values, reward, done):
        """
        Add a transition to the buffer.
        Args:
            state (torch.Tensor): Current state.
            mask (torch.Tensor): Action mask.
            action (torch.Tensor): Action taken.
            log_prob (torch.Tensor): Log probability of the action.
            values (torch.Tensor): State-action value.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.buffer.add(state, mask, action, log_prob, values, reward, done)
