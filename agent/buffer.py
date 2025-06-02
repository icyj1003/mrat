import torch


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
        self.next_states = []
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
        next_states,  # tensor: num_agents x state_dim
        dones,  # tensor: num_agents x 1
        violations,  # tensor: num_agents x 1
    ):
        self.states.append(states)
        self.masks.append(masks)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
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
            torch.stack(self.next_states, dim=stack_dim).to(
                self.device
            ),  # trajectory length x num_agents x state_dim
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
        self.next_states = []
        self.dones = []
        self.violations = []

    def __len__(self):
        return len(self.states)


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories for PPO.
    """

    def __init__(self, device="cpu"):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.violations = []
        self.device = device

    def add(
        self,
        state,
        mask,
        action,
        log_prob,
        reward,
        next_state,
        done,
        violations,
    ):
        self.states.append(state)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.violations.append(violations)

    def get(self):
        return (
            torch.stack(self.states).to(self.device),
            torch.stack(self.masks).to(self.device),
            torch.stack(self.actions).to(self.device),
            torch.stack(self.log_probs).to(self.device),
            torch.stack(self.rewards).to(self.device),
            torch.stack(self.next_states).to(self.device),
            torch.stack(self.dones).to(self.device),
            torch.stack(self.violations).to(self.device),
        )

    def clear(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.violations = []

    def __len__(self):
        return len(self.states)
