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
        # Episodic storage for aggregated training
        self.episodic_states = []
        self.episodic_masks = []
        self.episodic_actions = []
        self.episodic_old_log_probs = []
        self.episodic_advantages = []
        self.episodic_returns = []
        self.episodic_active_mask = []
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

        # active_count = int(active_agent_mask.sum().item())
        # if active_count > 0:
        #     shared_reward = adjusted_rewards[active_agent_mask].mean(
        #         dim=0, keepdim=True
        #     )
        # else:
        #     shared_reward = torch.zeros_like(adjusted_rewards[:1])
        # shared_rewards = shared_reward.expand_as(adjusted_rewards)

        shared_rewards = adjusted_rewards

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
            avg_actor_loss = avg_entropy_loss = avg_critic_loss = 0.0
            batch_count = 0
            for (
                state,
                mask,
                action,
                old_log_prob,
                advantage,
                return_,
            ) in tqdm(dataloader, desc="Mini-batch", leave=False):
                new_log_prob, entropy = self.evaluate(state, mask, action)
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * advantage
                )
                surrogate_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                critic_loss = torch.nn.functional.mse_loss(self.critic(state), return_)
                loss = (
                    surrogate_loss
                    + self.vf_coeff * critic_loss
                    + self.entropy_coeff * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.soft_update()

                avg_actor_loss += surrogate_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_actor_loss /= batch_count
                avg_entropy_loss /= batch_count
                avg_critic_loss /= batch_count

            if self.writer is not None:
                self.writer.add_scalar(
                    f"{self.name}_loss/actor_loss", avg_actor_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/entropy_loss", avg_entropy_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/critic_loss", avg_critic_loss, self.global_step
                )

            # print(
            #     f"[MAPPO] Epoch {self.global_step}: actor_loss={avg_actor_loss:.6f}, entropy_loss={avg_entropy_loss:.6f}, critic_loss={avg_critic_loss:.6f}"
            # )
            self.global_step += 1

        # update the penalty coefficient if using lagrangian penalty
        if self.use_lagrange:
            mean_violation = violations.mean()
            self.penalty_coeff += self.penalty_lr * (mean_violation).detach()
            self.penalty_coeff = max(0, min(self.penalty_coeff, 10))

            # log the penalty coefficient
            if self.writer is not None:
                self.writer.add_scalar(
                    f"{self.name}_penalty/penalty_coeff",
                    self.penalty_coeff,
                    self.global_step,
                )

        # clear the buffer
        self.buffer.clear()

    def compute_and_store_episode(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        active_masks,
    ):
        """
        Compute per-agent GAE for a finished episode and store flattened samples
        for later aggregated training.

        Inputs are expected as tensors on any device with shapes:
            states: num_agents x T x state_dim
            masks: num_agents x T x num_actions x action_dim
            actions: num_agents x T x num_actions
            log_probs: num_agents x T x num_actions
            rewards: num_agents x T x 1
            next_states: num_agents x T x state_dim
            dones: num_agents x T x 1
            active_masks: num_agents x T x 1
        """
        # ensure tensors on the agent device
        device = self.device
        states = states.to(device)
        masks = masks.to(device)
        actions = actions.to(device)
        log_probs = log_probs.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        active_masks = active_masks.to(device)

        num_agents = states.size(0)
        # compute advantages and returns per agent
        list_advantages = []
        list_returns = []
        for agent_idx in range(num_agents):
            agent_states = states[agent_idx]
            agent_next_states = next_states[agent_idx]
            # keep rewards and dones with trailing dim for consistency: (T,1)
            agent_rewards = rewards[agent_idx]
            agent_dones = dones[agent_idx]

            # values for each timestep (force shape (T,1))
            agent_values = self.critic_target(agent_states).detach().view(-1, 1)
            next_agent_values = (
                self.critic_target(agent_next_states).detach().view(-1, 1)
            )
            agent_values = torch.cat(
                [agent_values, next_agent_values[-1].unsqueeze(0)], dim=0
            )

            agent_adv, agent_ret = self.gae(agent_rewards, agent_values, agent_dones)

            # ensure (T,1) shapes
            agent_adv = agent_adv.view(-1, 1)
            agent_ret = agent_ret.view(-1, 1)

            list_advantages.append(agent_adv)
            list_returns.append(agent_ret)

        advantages = torch.stack(list_advantages, dim=0)  # num_agents x T x 1
        returns = torch.stack(list_returns, dim=0)

        # flatten and filter by active mask
        flat_states = states.reshape(-1, self.state_dim)
        flat_masks = masks.reshape(-1, self.num_actions, self.action_dim)
        flat_actions = actions.reshape(-1, self.num_actions)
        flat_old_log_probs = log_probs.reshape(-1, self.num_actions)
        flat_advantages = advantages.reshape(-1, 1)
        flat_returns = returns.reshape(-1, 1)
        flat_active = active_masks.reshape(-1).to(torch.bool)

        if flat_active.sum().item() == 0:
            if self.writer is not None:
                print("[MAPPO] No active samples to store for this episode.")
            else:
                print("[MAPPO] No active samples to store for this episode.")
            return

        sel_states = flat_states[flat_active]
        sel_masks = flat_masks[flat_active]
        sel_actions = flat_actions[flat_active]
        sel_old_log_probs = flat_old_log_probs[flat_active]
        sel_advantages = flat_advantages[flat_active]
        sel_returns = flat_returns[flat_active]

        # store into episodic lists (on CPU to save GPU memory)
        self.episodic_states.append(sel_states.cpu())
        # print(f"[MAPPO] Stored episode samples: {sel_states.size(0)}")
        self.episodic_masks.append(sel_masks.cpu())
        self.episodic_actions.append(sel_actions.cpu())
        self.episodic_old_log_probs.append(sel_old_log_probs.cpu())
        self.episodic_advantages.append(sel_advantages.cpu())
        self.episodic_returns.append(sel_returns.cpu())
        self.episodic_active_mask.append(
            torch.ones(sel_states.size(0), dtype=torch.bool)
        )

    def train_from_episodes(self, num_epochs=None, mini_batch_size=None):
        """Aggregate stored episode samples and perform PPO updates on the shared actor/critic."""
        if len(self.episodic_states) == 0:
            print("[MAPPO] No episodic data to train on.")
            return

        num_epochs = num_epochs or self.num_epochs
        mini_batch_size = mini_batch_size or self.mini_batch_size

        states = torch.cat(self.episodic_states, dim=0).to(self.device)
        masks = torch.cat(self.episodic_masks, dim=0).to(self.device)
        actions = torch.cat(self.episodic_actions, dim=0).to(self.device)
        old_log_probs = torch.cat(self.episodic_old_log_probs, dim=0).to(self.device)
        advantages = torch.cat(self.episodic_advantages, dim=0).to(self.device)
        returns = torch.cat(self.episodic_returns, dim=0).to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # print(f"[MAPPO] Training on aggregated samples: {states.size(0)} samples")

        dataset = torch.utils.data.TensorDataset(
            states, masks, actions, old_log_probs, advantages, returns
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=mini_batch_size, shuffle=True
        )

        for _ in trange(num_epochs, desc="Epochs", leave=False):
            avg_actor_loss = avg_entropy_loss = avg_critic_loss = 0.0
            batch_count = 0
            for (
                state,
                mask,
                action,
                old_log_prob,
                advantage,
                return_,
            ) in tqdm(dataloader, desc="Mini-batch", leave=False):
                new_log_prob, entropy = self.evaluate(state, mask, action)
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * advantage
                )
                surrogate_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                critic_loss = torch.nn.functional.mse_loss(self.critic(state), return_)
                loss = (
                    surrogate_loss
                    + self.vf_coeff * critic_loss
                    + self.entropy_coeff * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.soft_update()

                avg_actor_loss += surrogate_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_critic_loss += critic_loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_actor_loss /= batch_count
                avg_entropy_loss /= batch_count
                avg_critic_loss /= batch_count

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(
                    f"{self.name}_loss/actor_loss", avg_actor_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/entropy_loss", avg_entropy_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_loss/critic_loss", avg_critic_loss, self.global_step
                )
                self.writer.add_scalar(
                    f"{self.name}_train/surrogate1_loss",
                    surr1.mean().item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    f"{self.name}_train/surrogate2_loss",
                    surr2.mean().item(),
                    self.global_step,
                )

            # print(
            #     f"[MAPPO] Aggregated Epoch {self.global_step}: actor_loss={avg_actor_loss:.6f}, entropy_loss={avg_entropy_loss:.6f}, critic_loss={avg_critic_loss:.6f}"
            # )
            self.global_step += 1

        # clear episodic storage
        self.episodic_states = []
        self.episodic_masks = []
        self.episodic_actions = []
        self.episodic_old_log_probs = []
        self.episodic_advantages = []
        self.episodic_returns = []
        self.episodic_active_mask = []

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
