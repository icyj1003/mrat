import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *


def gumbel_sigmoid(logits, tau=1.0):
    """
    Gumbel-Sigmoid function with temperature and hard sampling.
    Args:
        logits (torch.Tensor): Input logits.
        tau (float): Temperature parameter.
        hard (bool): If True, use hard sampling.
    Returns:
        torch.Tensor: Sampled output.
        torch.Tensor: Soft output.
        torch.Tensor: Hard output.
    """
    # Compute the Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))).to(logits.device)

    # Apply the Gumbel trick
    y = (logits + gumbel_noise) / tau

    soft = torch.sigmoid(y)

    return soft


class Actor(nn.Module):
    """
    Actor network for Gumbel-Sigmoid sampling.
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        hidden_dim (int): Hidden layer dimension.
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
        # x = x + logits_mask.view(-1, self.num_action * self.action_dim) * 1e9
        return x.view(-1, self.num_action, self.action_dim)


class Critic(nn.Module):
    """
    Critic network for value estimation.
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        hidden_dim (int): Hidden layer dimension.
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
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


class GumbelPPO:
    """
    Gumbel Proximal Policy Optimization (PPO) algorithm.
    Args:
        actor (nn.Module): Actor network.
        critic (nn.Module): Critic network.
        lr (float): Learning rate.
        gamma (float): Discount factor.
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
        epoch,
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
        self.epoch = epoch
        self.gumbel_temperature = torch.tensor(gumbel_temperature)

    def act(self, state, mask):
        """
        Sample an action from the actor network.
        Args:
            state (torch.Tensor): Current state.
            mask (torch.Tensor): Action mask.
        Returns:
            torch.Tensor: Sampled action.
        """
        state = state.unsqueeze(0)
        mask = mask.unsqueeze(0)
        logits = self.actor(state, mask)
        m = torch.distributions.RelaxedBernoulli(
            temperature=self.gumbel_temperature, logits=logits
        )
        action = m.rsample()
        log_prob = m.log_prob(action)

        values = self.critic(state, action)
        return action.squeeze(), log_prob.squeeze(), values

    def update(self):
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        rewards = torch.tensor(self.buffer.rewards)
        dones = torch.tensor(self.buffer.dones)

        # Compute discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

        for _ in range(self.epoch):
            logits = self.actor(
                states, torch.zeros_like(actions)
            )  # no masking during update

            dist = torch.distributions.RelaxedBernoulli(
                temperature=self.gumbel_temperature, logits=logits
            )

            new_log_probs = dist.log_prob(actions).view(
                -1, self.num_action * self.action_dim
            )

            new_log_probs_sum = new_log_probs.sum(dim=-1)

            old_log_probs_sum = old_log_probs.view(
                -1, self.num_action * self.action_dim
            ).sum(dim=-1)

            state_values = self.critic(states, actions).squeeze()
            advantages = returns - state_values.detach()

            # PPO Ratio
            ratios = torch.exp(new_log_probs_sum - old_log_probs_sum.sum(dim=-1))

            # PPO Losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(state_values, returns)

            # Optimization steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer.clear()
        return actor_loss.item(), critic_loss.item()


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
        epoch=10,
    )

    num_episodes = 5
    step_per_episode = 10

    for episode in range(num_episodes):
        for step in range(step_per_episode):
            action, log_prob, values = ppo.act(state, mask)
            reward = torch.randn(1)
            done = torch.tensor([0])
            ppo.buffer.add(state, action, log_prob, reward, done)
        actor_loss, critic_loss = ppo.update()
        print(
            f"Episode {episode + 1}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}"
        )
