import torch

from agent.mappo import MAPPO


class DeliveryPolicy:
    def __init__(self, *args, **kwargs):
        self.steps = 0

    def act(self, *args, **kwargs):
        self.steps += 1

    def store_transition(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def model(self, *args, **kwargs):
        pass


class RandomDeliveryPolicy(DeliveryPolicy):
    def __init__(self, num_agents, num_actions, action_dim):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.action_dim = action_dim

    def act(self, states, masks, projection=None):
        super().act()
        logits = (
            torch.rand(self.num_agents, self.num_actions, self.action_dim)
            + masks * -1e10
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        # If projection is provided, apply it to the actions
        if projection is not None:
            valid_actions = projection(actions)
        else:
            valid_actions = actions

        # Calculate log probabilities of the actions
        log_probs = distribution.log_prob(valid_actions)
        return valid_actions, log_probs


class AllLinkDeliveryPolicy(DeliveryPolicy):
    def __init__(self):
        super().__init__()

    def act(self, states, masks, projection=None):
        super().act()
        actions = 1 - masks[:, :, 1]
        if projection is not None:
            valid_actions = projection(actions)
        else:
            valid_actions = actions

        # Calculate log probabilities of the actions
        log_probs = torch.zeros_like(valid_actions)
        return valid_actions, log_probs


class MAPPODeliveryPolicy(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=env.num_rats,
            action_dim=2,
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device="cpu",
            writer=writer,
        )

    def act(self, states, masks, projection=None):
        super().act()
        return self.agent.act(states, masks, projection)

    def store_transition(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        violations,
    ):
        super().store_transition()
        self.agent.buffer.add(
            states,
            masks,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
        )

    def train(self, *args, **kwargs):
        self.agent.update()
        return super().train(*args, **kwargs)

    def model(self):
        return self.agent.actor.state_dict()


class RATSelection(DeliveryPolicy):
    def __init__(self, args, env, writer=None):
        super().__init__()
        self.args = args
        self.num_vehicles = args.num_vehicles
        self.num_rats = env.num_rats
        self.agent = MAPPO(
            name="mappo_delivery",
            num_agents=args.num_vehicles,
            num_actions=1,
            action_dim=self.num_rats + 1,  # +1 for idling action
            state_dim=env.state_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device="cpu",
            writer=writer,
        )

    def mask_convertion(self, masks):
        # Convert mask from (num_agents, num_rats, 2) to (num_agents, 5)
        # where 1 indicates the action is restricted
        # Action 0 (idling) is always available, actions 1-4 correspond to RATs 0-3
        batch_size = masks.size(0)
        new_masks = torch.zeros(
            batch_size, self.num_rats + 1, dtype=masks.dtype, device=masks.device
        )
        # Action 0 (idling) is always available (mask = 0)
        new_masks[:, 0] = 0
        # Actions 1-4 correspond to RATs 0-3

        # force disable if masks[:, :, 1] is 1
        new_masks[:, 1:] = masks[:, :, 1]  #  = 1 later * with -1e10 in actor

        # force enable if masks[:, :, 0] is 1
        new_masks[:, 1:] = new_masks[:, 1:] - masks[:, :, 0] * 2  #

        return new_masks

    def action_convertion(self, actions):
        """
        Convert from RAT selection format to one-hot combination format
        Input: actions (num_agents, 1) - values 0 to num_rats
               - 0 = idle (no RAT selected)
               - 1 to num_rats = select RAT 0 to num_rats-1
        Output: one_hot_actions (num_agents, num_rats) - binary encoding
               - [0,0,0,0] = idle
               - [1,0,0,0] = RAT 0 selected
               - [0,1,0,0] = RAT 1 selected, etc.
        """
        batch_size = actions.size(0)
        one_hot_actions = torch.zeros(
            batch_size, self.num_rats, dtype=torch.long, device=actions.device
        )

        # Go through each agent
        for i in range(batch_size):
            action_value = actions[i, 0].item()  # Get the action value

            if action_value == 0:
                # Idle: keep all zeros [0,0,0,0]
                pass  # one_hot_actions[i] is already zeros
            else:
                # Select RAT: convert action 1-4 to RAT index 0-3
                rat_index = action_value - 1
                one_hot_actions[i, rat_index] = 1

        return one_hot_actions

    def combination_to_action_convertion(self, one_hot_actions):
        """
        Convert from one-hot combination format back to RAT selection format
        Input: one_hot_actions (num_agents, num_rats) - binary encoding
               - [0,0,0,0] = idle
               - [1,0,0,0] = RAT 0 selected
               - [0,1,0,0] = RAT 1 selected, etc.
        Output: actions (num_agents, 1) - values 0 to num_rats
               - 0 = idle (no RAT selected)
               - 1 to num_rats = select RAT 0 to num_rats-1
        """
        batch_size = one_hot_actions.size(0)
        actions = torch.zeros(
            batch_size, 1, dtype=torch.long, device=one_hot_actions.device
        )

        # Go through each agent
        for i in range(batch_size):
            # Check if any RAT is selected (sum > 0)
            if one_hot_actions[i].sum() == 0:
                # All zeros = idle
                actions[i, 0] = 0
            else:
                # Find which RAT is selected (index of the 1)
                rat_index = torch.argmax(one_hot_actions[i]).item()
                actions[i, 0] = rat_index + 1  # Convert RAT index 0-3 to action 1-4

        return actions

    def act(self, states, masks, projection=None):
        super().act()
        masks = self.mask_convertion(masks)

        actions = []
        dists = []

        # Step 1: Get logits from actor and sample actions (RAT selection format)
        for i in range(self.num_vehicles):
            # send to device
            state = states[i].to(self.agent.device)
            mask = masks[i].to(self.agent.device)

            # get the raw logits from the actor
            logit = self.agent.actor(state, mask).squeeze(
                0
            )  # 1 x num_actions x action_dim

            # save dist
            dists.append(torch.distributions.Categorical(logits=logit))

            # Step 2: Sample actions (in RAT selection format)
            actions.append(dists[i].sample())

        actions = torch.stack(actions, dim=0).detach()  # num_agents x num_actions

        # Step 3: Convert to combination form (one-hot format)
        combination_actions = self.action_convertion(actions)

        # Step 4: Apply projection if provided
        if projection is not None:
            valid_combination_actions = projection(combination_actions)
        else:
            valid_combination_actions = combination_actions

        # Step 5: Convert back to RAT selection format (for debugging/verification)
        valid_actions = self.combination_to_action_convertion(valid_combination_actions)

        # Step 6: Calculate log probs using the original sampled actions (before projection)
        log_probs = []
        for i in range(self.num_vehicles):
            log_probs.append(dists[i].log_prob(actions[i]))

        log_probs = torch.stack(log_probs, dim=0).detach()  # num_agents x num_actions

        return valid_combination_actions, log_probs

    def store_transition(
        self,
        states,
        masks,
        actions,
        log_probs,
        rewards,
        next_states,
        dones,
        violations,
    ):
        super().store_transition()
        self.agent.buffer.add(
            states,
            self.mask_convertion(masks),
            self.combination_to_action_convertion(actions),
            log_probs,
            rewards,
            next_states,
            dones,
            violations,
        )

    def train(self, *args, **kwargs):
        self.agent.update()
        return super().train(*args, **kwargs)

    def model(self):
        return self.agent.actor.state_dict()
