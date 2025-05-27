from copy import deepcopy

import torch
from tqdm import tqdm, trange

from buffer import MARolloutBuffer
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
        tau=1e-3,
        mini_batch_size=64,
        entropy_coeff=0.01,
        penalty_coeff=1.0,
        penalty_lr=1e-3,
        device="cpu",
        writer=None,
        use_lagrange=True,
        shared_actor=False,
        shared_critic=False,
    ):

        # Hyperparameters
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
        self.entropy_coeff = entropy_coeff
        self.penalty_coeff = penalty_coeff
        self.penalty_lr = penalty_lr
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.writer = writer
        self.use_lagrange = use_lagrange
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

        # define critics
        if shared_critic:
            self.critics = [Critic(state_dim, hidden_dim).to(device)] * num_agents
        else:
            self.critics = [
                Critic(state_dim, hidden_dim).to(device) for _ in range(num_agents)
            ]

        self.critics_target = [deepcopy(critic) for critic in self.critics]
        for i in range(num_agents):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=lr) for critic in self.critics
        ]

        self.buffer = MARolloutBuffer(device=device)
        self.global_step = 0

    def act(self, states, masks, projection=None):
        """
        Sample actions from the policy for each agent.
        Args:
            states: list of tensors, each of shape (state_dim)
                representing the state for each agent
            masks: list of tensors, each of shape (num_actions, action_dim)
                representing the mask for each agent
            projection: optional function to project actions to valid actions
        Returns:
            actions: tensor of shape (num_agents, num_actions)
                representing the sampled actions for each agent
            log_probs: tensor of shape (num_agents, num_actions)
                representing the log probabilities of the sampled actions
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
        return valid_actions, log_probs

    def evaluate(self, agent_idx, state, mask, action):
        """
        Evaluate the policy for a given agent.
        Args:
            agent_id: id of the agent
            state: tensor of shape (state_dim)
            mask: tensor of shape (num_actions, action_dim)
            action: tensor of shape (num_actions)
        Returns:
            log_probs: tensor of shape (num_actions)
            entropy: tensor of shape (num_actions)
        """
        state = state.to(self.device)
        mask = mask.to(self.device)
        action = action.to(self.device)

        # get the raw logits from the actor
        logit = self.actors[agent_idx](state, mask).squeeze(0)
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
            states,  # num_agents x trajectory length x state_dim
            masks,  # num_agents x trajectory length x num_actions x action_dim
            actions,  # num_agents x trajectory length x num_actions
            log_probs,  # num_agents x trajectory length x num_actions
            rewards,  # num_agents x trajectory length x 1
            next_states,  # num_agents x trajectory length x state_dim
            dones,  # num_agents x trajectory length x 1
            violations,  # num_agents x trajectory length x 1
        ) = self.buffer.get()

        list_advantages = []
        list_returns = []

        for agent_idx in range(self.num_agents):

            # Get the state values from the target critics
            agent_values = self.critics_target[agent_idx](
                states[agent_idx]
            ).detach()  # trajectory length x 1

            # Get the next state values from the target critics
            next_agent_values = self.critics_target[agent_idx](
                next_states[agent_idx]
            ).detach()

            # Prepare the values for the GAE : concat the last value to the end of the trajectory
            agent_values = torch.cat(
                [agent_values, next_agent_values[-1].unsqueeze(0)], dim=0
            )

            # if using lagrangian penalty, apply it to the rewards
            if self.use_lagrange:
                rewards[agent_idx] = (
                    rewards[agent_idx] - self.penalty_coeff * violations[agent_idx]
                )

            # compute discounted returns, advantages
            agent_advantages, agent_returns = self.gae(
                rewards[agent_idx], agent_values, dones[agent_idx]
            )

            # save the advantages and returns
            list_advantages.append(agent_advantages)
            list_returns.append(agent_returns)

        # convert to tensor
        advantages = torch.stack(list_advantages, dim=0)
        returns = torch.stack(list_returns, dim=0)

        # create dataset for each agent
        datasets = [
            torch.utils.data.TensorDataset(
                states[agent_idx],  # trajectory length x state_dim
                masks[agent_idx],  # trajectory length x num_actions x action_dim
                actions[agent_idx],  # trajectory length x num_actions
                log_probs[agent_idx],  # trajectory length x num_actions
                advantages[agent_idx],  # trajectory length x 1
                returns[agent_idx],  # trajectory length x 1
            )
            for agent_idx in range(self.num_agents)
        ]

        # create mini batches for each agent
        mini_batches = [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.mini_batch_size,
                shuffle=True,
                drop_last=True,
            )
            for dataset in datasets
        ]

        for _ in trange(self.num_epochs, desc="Epochs"):
            for agent_idx in trange(self.num_agents, desc="Agents", leave=False):
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
                    mini_batches[agent_idx],
                    desc=f"Mini-batch Agent {agent_idx}",
                    leave=False,
                ):
                    # evaluate the policy
                    new_log_prob, entropy = self.evaluate(
                        agent_idx, state, mask, action
                    )

                    # calculate the ratio
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # calculate the surrogate loss
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

                    # surrogate loss
                    surrogate_loss = -torch.min(surr1, surr2).mean()

                    # entropy loss
                    entropy_loss = self.entropy_coeff * entropy.mean()

                    # calculate the actor loss
                    actor_loss = surrogate_loss + entropy_loss

                    # update the actor
                    self.actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actors[agent_idx].parameters(), max_norm=0.5
                    )
                    self.actor_optimizers[agent_idx].step()

                    # update the critic
                    critic_loss = torch.nn.functional.smooth_l1_loss(
                        self.critics[agent_idx](state), return_
                    )

                    self.critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    self.critic_optimizers[agent_idx].step()

                    # log the losses
                    avg_actor_loss += actor_loss.item()
                    avg_entropy_loss += entropy_loss.item()
                    avg_critic_loss += critic_loss.item()

                    # soft update the target critics
                    self.soft_update()

                # normalize the losses
                avg_actor_loss /= len(mini_batches[agent_idx])
                avg_entropy_loss /= len(mini_batches[agent_idx])
                avg_critic_loss /= len(mini_batches[agent_idx])

                # log the losses
                if self.writer is not None:
                    self.writer.add_scalar(
                        f"actor_loss/agent_{agent_idx}",
                        avg_actor_loss,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        f"entropy_loss/agent_{agent_idx}",
                        avg_entropy_loss,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        f"critic_loss/agent_{agent_idx}",
                        avg_critic_loss,
                        self.global_step,
                    )

            self.global_step += 1

        # update the penalty coefficient if using lagrangian penalty
        if self.use_lagrange:
            self.penalty_coeff += self.penalty_lr * (violations.mean()).detach()
            self.penalty_coeff = max(0, min(self.penalty_coeff, 10))

    def soft_update(self):
        """
        Soft update the target critics.
        """
        for i in range(self.num_agents):
            for target_param, param in zip(
                self.critics_target[i].parameters(), self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
