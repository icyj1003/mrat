import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from common import Actor, Critic


class MAPPO:
    def __init__(
        self,
        num_agents,
        num_actions,
        action_dim,
        state_dim,
        hidden_dim=64,
        lr=3e-4,
        num_epochs=10,
        eps=0.2,
        gamma=0.99,
        lam=0.95,
        tau=0.5,
        entropy_coef=0.01,
        lagrange_init=1.0,
        lagrange_lr=1e-3,
        device="cpu",
        writer=None,
        shared_actor=False,
        shared_critic=False,
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.eps = eps
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.device = device
        self.writer = writer
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        if shared_actor:
            self.actors = [
                Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
            ] * num_agents
            self.lagranges = [
                nn.Parameter(torch.tensor(lagrange_init)).to(device)
            ] * num_agents
        else:
            self.actors = [
                Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
                for _ in range(num_agents)
            ]
            self.lagranges = [
                nn.Parameter(torch.tensor(lagrange_init)).to(device)
                for _ in range(num_agents)
            ]

        if shared_critic:
            self.critics = [Critic(state_dim, hidden_dim).to(device)] * num_agents
            self.critic_c = [Critic(state_dim, hidden_dim).to(device)] * num_agents
        else:
            self.critics = [
                Critic(state_dim, hidden_dim).to(device) for _ in range(num_agents)
            ]
            self.critic_c = [
                Critic(state_dim, hidden_dim).to(device) for _ in range(num_agents)
            ]

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=lr) for critic in self.critics
        ]

        self.critic_c_optimizers = [
            torch.optim.Adam(critic_c.parameters(), lr=lr) for critic_c in self.critic_c
        ]

        self.lagrange_optimizers = [
            torch.optim.Adam([lagrange], lr=lagrange_lr) for lagrange in self.lagranges
        ]
