from copy import deepcopy

import torch

from common import Actor, Critic


class MARolloutBuffer:
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
        states,  # tensor: num_agents x state_dim
        masks,  # tensor: num_agents x num_actions x action_dim
        actions,  # tensor: num_agents x num_actions
        log_probs,  # tensor: num_agents x num_actions
        rewards,  # tensor: num_agents x 1
        dones,  # tensor: num_agents x 1
        violations,  # tensor: num_agents x 1
    ):
        self.states.append(states)
        self.masks.append(masks)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.violations.append(violations)

    def get(self):
        stack_dim = 1  # put the trajectory length behind the number of agents
        return (
            torch.stack(self.states, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x state_dim
            torch.stack(self.masks, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x num_actions x action_dim
            torch.stack(self.actions, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x num_actions
            torch.stack(self.log_probs, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x num_actions
            torch.stack(self.rewards, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x 1
            torch.stack(self.dones, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x 1
            torch.stack(self.violations, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x 1
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
        mini_batch_size=64,
        device="cpu",
        writer=None,
        shared_actor=False,
        shared_critic=False,
        shared_lagrange=False,
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
        self.lagrange_lr = lagrange_lr
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.writer = writer
        self.shared_actor = shared_actor
        self.shared_critic = shared_critic

        # define actors
        if shared_actor:
            self.actors = [
                Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
            ] * num_agents
        else:
            self.actors = [
                Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
                for _ in range(num_agents)
            ]

        # define lagrange multipliers
        if shared_lagrange:
            self.lagranges = [torch.tensor(lagrange_init).to(device)] * num_agents
        else:
            self.lagranges = [
                torch.tensor(lagrange_init).to(device) for _ in range(num_agents)
            ]

        # define critics
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

        self.critics_target = [deepcopy(critic) for critic in self.critics]
        self.critic_c_target = [deepcopy(critic) for critic in self.critic_c]
        for i in range(num_agents):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.critic_c_target[i].load_state_dict(self.critic_c[i].state_dict())

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=lr) for critic in self.critics
        ]

        self.buffer = MARolloutBuffer(device=device)

    def act(self, states, masks, projection=None):
        """
        Sample actions from the policy.
        states: tensor of shape (num_agents, state_dim)
        masks: tensor of shape (num_agents, num_actions, action_dim)
        projection (optional): function to project the actions to valid actions space
        returns: tensor of shape (num_agents, num_actions)
        """
        actions = []
        log_probs = []
        dists = []

        for i in range(self.num_agents):
            # send to device
            state = states[i].to(self.device)
            mask = masks[i].to(self.device)

            # get the raw logits from the actor
            logit = self.actors[i](state, mask).squeeze(
                0
            )  # 1 x num_actions x action_dim

            # save dist
            dists.append(torch.distributions.Categorical(logits=logit))

            # save sampled action
            actions.append(dists[i].sample())

        actions = torch.stack(actions, dim=0).detach()  # num_agents x num_actions

        if projection is not None:
            valid_actions = projection(actions)

        # calculate log probs
        for i in range(self.num_agents):
            log_probs.append(dists[i].log_prob(valid_actions[i]))

        log_probs = torch.stack(log_probs, dim=0).detach()  # num_agents x num_actions
        return actions, log_probs

    def evaluate(self, states, masks, actions):
        """
        Evaluate the policy.
        states: tensor of shape (num_agents, state_dim)
        masks: tensor of shape (num_agents, num_actions, action_dim)
        actions: tensor of shape (num_agents, num_actions)
        returns: tuple of (log_probs, entropy)
        """
        log_probs = []
        entropy = []

        for i in range(self.num_agents):
            # send to device
            state = states[i].to(self.device)
            mask = masks[i].to(self.device)

            # get the raw logits from the actor
            logit = self.actors[i](state, mask).squeeze(0)
            # calculate log probs
            dist = torch.distributions.Categorical(logits=logit)
            log_probs.append(dist.log_prob(actions[i]))
            entropy.append(dist.entropy())
        log_probs = torch.stack(log_probs, dim=0).detach()
        entropy = torch.stack(entropy, dim=0).detach()
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
        Update the policy and critic.
        """
        # get the data from the buffer
        (
            states,
            masks,
            actions,
            log_probs,
            rewards,
            dones,
            violations,
        ) = self.buffer.get()
