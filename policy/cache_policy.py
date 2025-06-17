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


class PPOCachePolicy(CachePolicy):
    def __init__(self, args, writer=None):
        super().__init__()
        self.args = args
        self.agent = PPO(
            name="ppo_cache",
            num_actions=args.num_edges * args.num_items,
            action_dim=2,
            state_dim=4 * args.num_items,
            hidden_dim=512,
            lr=args.lr,
            num_epochs=args.num_epoch,
            clip_range=args.clip_range,
            gamma=1,
            gae_lambda=1,
            tau=args.tau,
            entropy_coeff=args.entropy_coeff,
            penalty_coeff=args.penalty_coeff,
            mini_batch_size=1,
            max_grad_norm=args.max_grad_norm,
            device="cuda",
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
