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
