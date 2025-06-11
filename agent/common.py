from typing import *

import torch.nn as nn


class TransformerActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(TransformerActor, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, logits_mask=None):
        x1 = self.embedding(x)
        x2 = self.transformer(x1)
        x3 = self.policy_head(x2).squeeze(-1)  # [batch_size, num_actions * action_dim]
        if logits_mask is not None:
            x3 = x3 + logits_mask * -1e10  # Apply mask to logits
        return x3


class TransformerCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(TransformerCritic, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x1 = self.embedding(x)
        x2 = self.transformer(x1)
        x3 = self.policy_head(x2).squeeze(-1)  # [batch_size, num_actions * action_dim]
        x4 = x3.mean(dim=1, keepdim=True)  # Average over the sequence length
        return x4


class Actor(nn.Module):
    """
    Actor network for Gumbel-Sigmoid sampling.
    Args:
        state_dim (int): Dimension of the state.
        num_actions (int): Number of actions.
        action_dim (int): Dimension of each action.
        hidden_dim (int): Hidden dimension of the network.
        activation (str): Activation function to use ("relu" or "tanh").
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        action_dim: int,
        hidden_dim: int = 128,
        activation: Literal["relu", "tanh"] = "tanh",
    ):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, num_actions * action_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
        )
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self, state, logits_mask):
        mask = logits_mask.view(-1, self.num_actions, self.action_dim)
        x = self.fc(state)  # [batch_size, num_actions * num_sub_action * action_dim]
        x = x.view(-1, self.num_actions, self.action_dim)
        x = x + mask * -1e10
        return x


class Critic(nn.Module):
    """
    Critic network for value estimation.
    Args:
        state_dim (int): State dimension.
        num_actions (int): Number of actions.
        action_dim (int): Action dimension.
        hidden_dim (int): Hidden layer dimension.
        activation (str): Activation function.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        activation: Literal["relu", "tanh"] = "tanh",
    ):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        x = self.fc(state)
        return x
