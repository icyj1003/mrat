import torch

from agent.ppo import PPO


class CachePolicy:
    def __init__(self, *args, **kwargs):
        self.steps = 0

    def act(self, *args, **kwargs):
        self.steps += 1

    def store_transition(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def merge_rewards(self, *args, **kwargs):
        pass


class RandomCachePolicy(CachePolicy):
    def __init__(self, num_edges, num_items):
        super().__init__()
        self.num_edges = num_edges
        self.num_items = num_items

    def act(self, states, masks, projection=None):
        super().act()
        logits = torch.rand((self.num_edges + 1) * self.num_items) + masks * -1e10
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


class PPOCachePolicy(CachePolicy):
    def __init__(self, args, writer=None):
        super().__init__()
        self.args = args
        self.agent = PPO(
            name="ppo_cache",
            num_actions=1,
            action_dim=(args.num_edges + 1) * args.num_items,
            state_dim=args.num_edges * args.num_items * 2 + args.num_items * 3,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=1,
            gae_lambda=1,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device="cpu",
            writer=writer,
            use_lagrange=False,
        )

    def act(self, states, masks, projection=None):
        super().act()
        actions, log_probs = self.agent.act(states, masks, projection=projection)
        return actions, log_probs

    def store_transition(
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
        super().store_transition()
        self.agent.buffer.add(
            state,
            mask,
            action,
            log_prob,
            reward,
            next_state,
            done,
            violations,
        )

    def train(self):
        super().train()
        self.agent.update()

    def merge_rewards(self, mean_delivery_rewards):
        super().merge_rewards()
        self.agent.buffer.rewards[-1] = torch.tensor(
            [mean_delivery_rewards], dtype=torch.float32
        )
