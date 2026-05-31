from copy import deepcopy

import torch
from tqdm import tqdm, trange

from agent.buffer import MARolloutBuffer
from agent.common import Actor, Critic


class MAPPO:
    def __init__(
        self,
        name,
        num_agents,
        num_actions,
        action_dim,
        state_dim,
        hidden_dim=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
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
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
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

        self.actor = Actor(state_dim, num_actions, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.actor_lr},
                {"params": self.critic.parameters(), "lr": self.critic_lr},
            ]
        )

        self.buffer = MARolloutBuffer(device=device)
        self.global_step = 0

    def act(self, states, masks):
        # Batch actor inference only for active agents.
        states = torch.stack([state.to(self.device) for state in states], dim=0)
        masks = torch.stack([mask.to(self.device) for mask in masks], dim=0)

        active_mask = states[:, -1] > 0.5
        active_indices = torch.where(active_mask)[0]

        actions = torch.zeros(
            states.size(0), self.num_actions, dtype=torch.long, device=self.device
        )
        log_probs = torch.zeros(
            states.size(0), self.num_actions, dtype=torch.float32, device=self.device
        )

        if active_indices.numel() == 0:
            return actions.cpu(), log_probs.cpu()

        active_states = states[active_indices]
        active_masks = masks[active_indices]

        # get the raw logits from the actor
        logit = self.actor(active_states, active_masks)
        if logit.dim() == 4 and logit.size(0) == 1:
            logit = logit.squeeze(0)

        dist = torch.distributions.Categorical(logits=logit)
        sampled_actions = dist.sample().detach()  # num_active x num_actions

        # calculate log probs
        sampled_log_probs = dist.log_prob(
            sampled_actions
        ).detach()  # num_active x num_actions

        actions[active_indices] = sampled_actions
        log_probs[active_indices] = sampled_log_probs

        return actions.cpu(), log_probs.cpu()

    def evaluate(self, state, mask, action):
        state = state.to(self.device)
        mask = mask.to(self.device)
        action = action.to(self.device)

        # get the raw logits from the actor
        logit = self.actor(state, mask).squeeze(0)

        # calculate log probs
        dist = torch.distributions.Categorical(logits=logit)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
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
                delta + self.gamma * self.gae_lambda * not_done * last_adv
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
            states,  # num_agents x trajectory length x state_dim
            masks,  # num_agents x trajectory length x num_actions x action_dim
            actions,  # num_agents x trajectory length x num_actions
            log_probs,  # num_agents x trajectory length x num_actions
            rewards,  # num_agents x trajectory length x 1
            next_states,  # num_agents x trajectory length x state_dim
            dones,  # num_agents x trajectory length x 1
            violations,  # num_agents x trajectory length x 1
            active_masks,  # num_agents x trajectory length x 1
        ) = self.buffer.get()

        num_agents = states.size(0)
        if num_agents == 0:
            self.buffer.clear()
            return

        active_agent_mask = active_masks[:, 0, 0].to(torch.bool)

        # Shared reward over the active batch: mean only across active vehicles.
        if self.use_lagrange:
            adjusted_rewards = rewards - self.penalty_coeff * violations
        else:
            adjusted_rewards = rewards

        active_count = int(active_agent_mask.sum().item())
        if active_count > 0:
            shared_reward = adjusted_rewards[active_agent_mask].mean(
                dim=0, keepdim=True
            )
        else:
            shared_reward = torch.zeros_like(adjusted_rewards[:1])
        shared_rewards = shared_reward.expand_as(adjusted_rewards)

        list_advantages = []
        list_returns = []

        for agent_idx in range(num_agents):

            # Get the state values from the target critics
            agent_values = self.critic_target(states[agent_idx]).detach()

            # Get the next state values from the target critics
            next_agent_values = self.critic_target(next_states[agent_idx]).detach()

            # Prepare the values for the GAE : concat the last value to the end of the trajectory
            agent_values = torch.cat(
                [agent_values, next_agent_values[-1].unsqueeze(0)], dim=0
            )

            # compute discounted returns, advantages
            agent_advantages, agent_returns = self.gae(
                shared_rewards[agent_idx], agent_values, dones[agent_idx]
            )

            # save the advantages and returns
            list_advantages.append(agent_advantages)
            list_returns.append(agent_returns)

        # convert to tensor (only active agents included)
        if len(list_advantages) == 0:
            # no active agents, nothing to update
            self.buffer.clear()
            return

        advantages = torch.stack(list_advantages, dim=0)
        returns = torch.stack(list_returns, dim=0)

        # reshape dim 1 of all agents to create a joint dataset
        joint_states = states.reshape(-1, self.state_dim)
        joint_masks = masks.reshape(-1, self.num_actions, self.action_dim)
        joint_actions = actions.reshape(-1, self.num_actions)
        joint_log_probs = log_probs.reshape(-1, self.num_actions)
        joint_advantages = advantages.reshape(-1, 1)
        joint_returns = returns.reshape(-1, 1)
        joint_active_mask = active_masks.reshape(-1).to(torch.bool)

        # filter out padded samples after flattening
        joint_states = joint_states[joint_active_mask]
        joint_masks = joint_masks[joint_active_mask]
        joint_actions = joint_actions[joint_active_mask]
        joint_log_probs = joint_log_probs[joint_active_mask]
        joint_advantages = joint_advantages[joint_active_mask]
        joint_returns = joint_returns[joint_active_mask]

        # create a single dataset to train the joint policy
        dataset = torch.utils.data.TensorDataset(
            joint_states,
            joint_masks,
            joint_actions,
            joint_log_probs,
            joint_advantages,
            joint_returns,
        )

        # create mini batch
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
        )

        for _ in trange(self.num_epochs, desc="Epochs", leave=False):
            avg_actor_loss, avg_entropy_loss, avg_critic_loss = (
                0.0,
                0.0,
                0.0,
            )
            # update the actor and critic
            for (
                state,
                mask,
                action,
                old_log_prob,
                advantage,
                return_,
            ) in tqdm(
                dataloader,
                desc=f"Mini-batch",
                leave=False,
            ):
                # evaluate the policy
                new_log_prob, entropy = self.evaluate(state, mask, action)

                # calculate the ratio
                ratio = torch.exp(new_log_prob - old_log_prob)

                # calculate the surrogate loss
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * advantage
                )

                # surrogate loss
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # entropy loss
                entropy_loss = -entropy.mean()

                # critic loss
                critic_loss = torch.nn.functional.mse_loss(self.critic(state), return_)

                # total loss
                loss = (
                    surrogate_loss
                    + self.vf_coeff * critic_loss
                    + self.entropy_coeff * entropy_loss
                )

                # update
                self.optimizer.zero_grad()
                loss.backward()

                # clip the gradients
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # log the losses
                avg_actor_loss += surrogate_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()

                # soft update the target critics
                self.soft_update()

            # normalize the losses
            avg_actor_loss /= len(dataloader)
            avg_entropy_loss /= len(dataloader)
            avg_critic_loss /= len(dataloader)

            # log the losses
            if self.writer is not None:
                self.writer.add_scalar(
                    f"{self.name}_loss/actor_loss",
                    avg_actor_loss,
                    self.global_step,
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/entropy_loss",
                    avg_entropy_loss,
                    self.global_step,
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/critic_loss",
                    avg_critic_loss,
                    self.global_step,
                )

            self.global_step += 1

        # update the penalty coefficient if using lagrangian penalty
        if self.use_lagrange:
            mean_violation = violations.mean()
            self.penalty_coeff += self.penalty_lr * (mean_violation).detach()
            self.penalty_coeff = max(0, min(self.penalty_coeff, 10))

        # clear the buffer
        self.buffer.clear()

    def soft_update(self):
        """
        Soft update the target critics.
        """
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
