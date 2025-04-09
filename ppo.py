import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
from gumbel_sigmoid import gumbel_sigmoid, sigmoid_log_prob


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


class GumbelPPO:
    """
    Gumbel Proximal Policy Optimization (PPO) algorithm.
    Args:
        state_dim (int): Dimension of the state.
        num_action (int): Number of actions.
        action_dim (int): Dimension of each action.
        actor_hidden_dim (int): Hidden dimension of the actor network.
        critic_hidden_dim (int): Hidden dimension of the critic network.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
        num_epoch (int): Number of epochs to train per update.
        eps (float): Epsilon for PPO clipping.
        gumbel_temperature (float): Temperature for Gumbel sampling.
        gamma (float): Discount factor for rewards.
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
        gumbel_temperature=0.5,
        gamma=0.99,
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
        self.gumbel_temperature = gumbel_temperature
        self.eps = eps

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

        # Compute log probabilities
        log_probs = sigmoid_log_prob(logits, actions)

        return log_probs

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

        action = gumbel_sigmoid(logits, self.gumbel_temperature)
        log_prob = sigmoid_log_prob(logits, action)

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
            new_log_probs = self.get_log_prob(states, masks, actions)

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
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss
            critic_loss = F.mse_loss(new_values, returns)

            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()


if __name__ == "__main__":
    state_dim = 10
    num_rows = 2
    num_cols = 5

    output_dim = num_rows * num_cols
    hidden_dim = 128

    state = torch.rand(state_dim)
    mask = torch.tensor([[1, -1, 0, 0, 0], [0, 0, 0, -1, 1]]).float()

    ppo = GumbelPPO(
        state_dim=state_dim,
        num_action=num_rows,
        action_dim=num_cols,
        actor_hidden_dim=hidden_dim,
        critic_hidden_dim=hidden_dim,
        actor_lr=0.001,
        critic_lr=0.001,
        num_epoch=10,
    )

    num_episodes = 5
    max_steps_per_episode = 1000

    for episode in range(num_episodes):

        # reset dummy environment
        state = torch.rand(state_dim)
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            action, log_prob, value = ppo.act(state, mask)
            reward = torch.rand(1)

            done = True
            ppo.buffer.add(
                state,
                mask,
                action,
                log_prob,
                value,
                reward,
                done,
            )

            # Next state
            state = torch.rand(state_dim)
            step += 1

        ppo.update()
        ppo.clear()
        print(f"Episode {episode + 1} finished.")
