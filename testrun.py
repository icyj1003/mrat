import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import parse_args
from policy.cache_policy import PPOCachePolicy, RandomCachePolicy
from policy.delivery_policy import MAPPODeliveryPolicy
from policy.selection_policy import GTVS
from utils import create_environment, log_episode

current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
writer = SummaryWriter(log_dir=f"runs/{current}")


if __name__ == "__main__":
    args = parse_args()
    env, cache_env = create_environment(args)

    cache_model = PPOCachePolicy(
        args,
        writer=writer,
    )
    delivery_model = MAPPODeliveryPolicy(
        args,
        env,
        writer=writer,
    )

    # Begin training loop
    for episode in tqdm(
        range(args.episode + args.evaluation_episodes), desc="Running", unit="episode"
    ):
        accumulated_rewards = []
        cummulated_cache_cost = []

        # Step 1: Run the vehicle selection policy here
        cache_vehicle = GTVS(env)

        # Step 2: Run the multi-step caching policy here
        while not cache_env.is_done():
            cache_states = torch.tensor(cache_env.states, dtype=torch.float32)
            cache_masks = torch.tensor(cache_env.masks, dtype=torch.float32).reshape(-1)

            cache_actions, cache_log_probs = cache_model.act(
                cache_states,  # num_edges * num_items * 2 + num_items * 2
                cache_masks,  # (num_edges + 1) * num_items
            )

            # get the reward from the cache environment
            reward, loc_index = cache_env.step(cache_actions)

            # if the action is to cache at the last location, multiply the reward by the number of vehicles
            if loc_index == args.num_edges:
                reward = reward * len(cache_vehicle)

            # accumulate the cache cost
            cummulated_cache_cost.append(reward)

            # create tensors
            reward_tensor = torch.tensor([0], dtype=torch.float32)
            next_state_tensor = torch.tensor(cache_env.states, dtype=torch.float32)
            done_tensor = torch.tensor([cache_env.is_done()], dtype=torch.float32)
            violation_tensor = torch.tensor([0], dtype=torch.float32)

            cache_model.store_transition(
                cache_states,  # num_edges * num_items * 2 + num_items * 2
                cache_masks,  # (num_edges + 1) * num_items
                cache_actions.unsqueeze(0),  # 1 x 1
                cache_log_probs,  # (num_edges + 1) * num_items
                reward_tensor,  # 1
                next_state_tensor,  # num_edges * num_items * 2 + num_items * 2
                done_tensor,  # 1
                violation_tensor,  # 1
            )

        # Step 3: Overwrite the cache states in the environment before performing the small step
        total_cost = env.large_step(cache_env.cache_status, cache_vehicle)

        # Step 4: Run the multi-agent delivery policy here
        accumulated_rewards = []
        while not env.is_small_done():
            # convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # MAPPO Strategy
            actions, log_probs = delivery_model.act(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions
                projection=env.greedy_projection,
            )

            # reshape action to match the environment
            reshaped_actions = actions.view(args.num_vehicles, env.num_rats)

            # step the environment
            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

            # convert to tensor
            reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            done_tensor = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
            violation_tensor = torch.tensor(violations, dtype=torch.float32).view(-1, 1)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32)

            accumulated_rewards.append(rewards.mean())

            # store the transition in the delivery model
            delivery_model.store_transition(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions
                reshaped_actions,  # num_agents x num_actions
                log_probs,  # num_agents x num_actions
                reward_tensor,  # num_agents x 1
                next_state_tensor,  # num_agents x state_dim
                done_tensor,  # num_agents x 1
                violation_tensor,  # num_agents x 1
            )

            # update the policy
            if (
                delivery_model.steps > 0
                and delivery_model.steps % args.steps_per_batch == 0
            ):
                delivery_model.train()

        cache_model.merge_rewards(
            np.mean(accumulated_rewards) - total_cost * env.storage_cost_scale
        )

        # update the cache model
        if episode > 0 and episode % 30 == 0:
            cache_model.train()

        log_episode(
            writer,
            env,
            episode,
            accumulated_rewards,
            cummulated_cache_cost,
        )

        env.reset()
        cache_env.reset(init_states=env.cache_states)
