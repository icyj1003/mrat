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
        self.values = []
        self.rewards = []
        self.dones = []
        self.violations = []
        self.device = device

    def add(self, state, mask, action, log_prob, values, reward, done, violation):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)
        self.violations.append(violation)

    def get(self):
        return (
            torch.stack(self.states).to(self.device),
            torch.stack(self.masks).to(self.device),
            torch.stack(self.actions).to(self.device),
            torch.stack(self.log_probs).to(self.device),
            torch.stack(self.values).to(self.device),
            torch.tensor(self.rewards).to(self.device),
            torch.tensor(self.dones).to(self.device),
            torch.tensor(self.violations).to(self.device),
        )

    def clear(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.violations = []

    def __len__(self):
        return len(self.states)
