import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import parse_args
from mappo import MAPPO
from ppo import PPO
from utils import create_environment
from policy.selection_policy import GTVS

current = datetime.datetime.now().strftime("%Y_%m_%d %H:%M:%S")
writer = SummaryWriter(log_dir=f"runs/full_{current}")


if __name__ == "__main__":
    args = parse_args()
    env, cache_env = create_environment(args)

    steps = 0

    cache_model = PPO(
        num_actions=1,
        action_dim=(args.num_edges + 1) * args.num_items,
        state_dim=args.num_edges * args.num_items * 2 + args.num_items * 3,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_epochs=args.num_epoch,
        clip_range=args.clip_range,
        gamma=1,
        gae_lambda=args.gae_lambda,
        tau=args.tau,
        entropy_coeff=args.entropy_coeff,
        penalty_coeff=args.penalty_coeff,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
        device="cpu",
        writer=writer,
        use_lagrange=False,
    )

    delivery_model = MAPPO(
        num_agents=args.num_vehicles,
        num_actions=env.num_rats,
        action_dim=2,
        state_dim=env.state_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr * 1e-1,
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

    # Begin training loop
    for episode in tqdm(range(args.episode), desc="Training", unit="episode"):
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

            reward, loc_index = cache_env.step(cache_actions)

            if loc_index == args.num_edges:
                reward = reward * len(cache_vehicle)

            cummulated_cache_cost.append(reward)

            # create tensors for the cache model
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            next_state_tensor = torch.tensor(cache_env.states, dtype=torch.float32)
            done_tensor = torch.tensor([cache_env.is_done()], dtype=torch.float32)
            violation_tensor = torch.tensor([0], dtype=torch.float32)

            cache_model.buffer.add(
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
                mask_tensor,  # num_agents x num_actions x action_dim
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

            delivery_model.buffer.add(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions x action_dim
                actions,  # num_agents x num_actions
                log_probs,  #  num_agents x num_actions
                reward_tensor,  # num_agents x 1
                next_state_tensor,  # num_agents x state_dim
                done_tensor,  # num_agents x 1
                violation_tensor,  # num_agents x 1
            )
            steps += 1

            if steps == args.steps_per_batch:
                # update the policy
                delivery_model.update()
                steps = 0

        mean_reward = np.mean(accumulated_rewards)

        # add the mean reward to the cache model buffer reward
        normalized_mean_reward = mean_reward / len(cache_model.buffer.rewards)
        cache_model.buffer.rewards = [
            _ + normalized_mean_reward for _ in cache_model.buffer.rewards
        ]

        # update the cache model
        if episode > 0 and episode % 20 == 0:
            cache_model.update()

        writer.add_scalar(
            "log/cache_cost",
            -np.sum(cummulated_cache_cost),
            episode,
        )

        writer.add_scalar(
            "log/objective",
            np.sum(cummulated_cache_cost) + mean_reward,
            episode,
        )

        writer.add_scalar(
            "log/episode_length",
            len(accumulated_rewards),
            episode,
        )

        writer.add_scalar(
            "log/mean_delay",
            np.mean(env.delay),
            episode,
        )

        writer.add_scalar(
            "log/mean_reward",
            np.mean(accumulated_rewards),
            episode,
        )

        writer.add_scalar(
            "log/mean_deadline_violation",
            np.clip(np.mean(env.delay - env.delivery_deadline[env.requested]), 0, None),
            episode,
        )

        writer.add_scalar(
            "log/delay per segment",
            np.mean(env.delay / env.num_code_min[env.requested]),
            episode,
        )

        writer.add_scalar(
            "log/cost_per_bit",
            np.mean(env.cost / env.item_size[env.requested]),
            episode,
        )

        env.reset()

    # Begin evaluation loop

    for episode in tqdm(
        range(args.evaluation_episodes), desc="Evaluation", unit="episode"
    ):
        accumulated_rewards = []
        cummulated_cache_cost = []

        # Step 1: Run the vehicle selection policy here
        cache_vehicle = GTVS(
            init_positions=env.positions,
            init_velocities=env.velocities,
            direction=env.direction,
            vmin=env.vmin,
            vmax=env.vmax,
            dt=env.dt,
            coverage_radius=env.v2v_pc5_coverage,
        )

        # Step 2: Run the multi-step caching policy here
        cache_env.reset()
        while not cache_env.is_done():
            cache_states = torch.tensor(cache_env.states, dtype=torch.float32)
            cache_masks = torch.tensor(cache_env.masks, dtype=torch.float32).reshape(-1)

            cache_actions, cache_log_probs = cache_model.act(
                cache_states,  # num_edges * num_items * 2 + num_items * 2
                cache_masks,  # (num_edges + 1) * num_items
            )

            reward, loc_index = cache_env.step(cache_actions)

            if loc_index == args.num_edges:
                reward = reward * len(cache_vehicle)

            cummulated_cache_cost.append(reward)

        # Step 3: Overwrite the cache states in the environment before performing the small step
        total_cost = env.large_step(cache_env.cache_status, cache_vehicle)

        # Step 4: Run the multi-agent delivery policy here
        while not env.is_small_done():
            # convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # MAPPO Strategy
            actions, log_probs = delivery_model.act(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions x action_dim
                projection=env.greedy_projection,
            )

            # reshape action to match the environment
            reshaped_actions = actions.view(args.num_vehicles, env.num_rats)

            # step the environment
            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

            accumulated_rewards.append(rewards.mean())

        # Log the results
        mean_reward = np.mean(accumulated_rewards)
        writer.add_scalar(
            "log_eval/cache_cost",
            -np.sum(cummulated_cache_cost),
            episode,
        )
        writer.add_scalar(
            "log_eval/objective",
            np.sum(cummulated_cache_cost) + mean_reward,
            episode,
        )
        writer.add_scalar(
            "log_eval/episode_length",
            len(accumulated_rewards),
            episode,
        )
        writer.add_scalar(
            "log_eval/mean_delay",
            np.mean(env.delay),
            episode,
        )
        writer.add_scalar(
            "log_eval/mean_reward",
            np.mean(accumulated_rewards),
            episode,
        )
        writer.add_scalar(
            "log_eval/mean_deadline_violation",
            np.clip(np.mean(env.delay - env.delivery_deadline[env.requested]), 0, None),
            episode,
        )
        writer.add_scalar(
            "log_eval/delay per segment",
            np.mean(env.delay / env.num_code_min[env.requested]),
            episode,
        )
        writer.add_scalar(
            "log_eval/cost_per_bit",
            np.mean(env.cost / env.item_size[env.requested]),
            episode,
        )
        env.reset()
