import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import parse_args
from policy.cache_policy import PPOCachePolicy
from policy.delivery_policy import MAPPODeliveryPolicy
from policy.selection_policy import GTVS
from utils import get_environment, log_and_collect, aggregate_metrics

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Setup logger
    current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{current}_{args.name}")

    # Get the environment and cache environment
    env, cache_env = get_environment(args)

    # Initialize the cache and delivery policies
    cache_model = PPOCachePolicy(
        args,
        writer=writer,
    )
    delivery_model = MAPPODeliveryPolicy(
        args,
        env,
        writer=writer,
    )

    total_episodes = args.training_episodes + args.evaluation_episodes

    evaluate = []

    # Begin training loop
    for episode in tqdm(range(total_episodes), desc="Running", unit="episode"):
        cumulated_rewards = []
        cumulated_cache_costs = []

        # At Large time-scale:
        # Step 1: Run the vehicle selection policy here
        caching_vehicle = GTVS(env)

        # Step 2: Make caching decisions
        # Get the states and masks from the environment
        cache_states = torch.tensor(cache_env.states, dtype=torch.float32)
        cache_masks = torch.tensor(cache_env.masks, dtype=torch.float32)

        # Get the cache actions using the cache model
        cache_actions, cache_log_probs = cache_model.act(
            cache_states,  # num_edges * num_items * 2 + num_items * 2
            cache_masks,  # (num_edges + 1) * num_items
            projection=cache_env.greedy_projection,  # Greedy projection for cache selection
        )

        # cache_actions, cache_log_probs = cache_env.random_policy()

        # Reshape the actions to match the environment
        reshaped_actions = cache_actions.view(cache_env.num_edges, cache_env.num_items)

        # Get the cache replacement costs
        reward = cache_env.step(reshaped_actions)

        # cumulate the cache costs
        cumulated_cache_costs.append(reward)

        # Overwrite the cache states in the environment before performing the small step
        env.large_step(reshaped_actions, caching_vehicle)

        # Small time-scale:
        # Run the multi-agent delivery policy here
        while not env.is_small_done():
            # Convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # MAPPO Policy action selection
            actions, log_probs = delivery_model.act(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions
                projection=env.greedy_projection,
            )

            # Reshape action to match the environment
            reshaped_actions = actions.view(args.num_vehicles, env.num_rats)

            # Step the environment
            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

            # Convert to tensor
            reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            done_tensor = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
            violation_tensor = torch.tensor(violations, dtype=torch.float32).view(-1, 1)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32)

            # cumulate the delivery rewards
            cumulated_rewards.append(rewards.mean())

            # Store the transition in the delivery model
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

            # Update the cache policy
            if episode > 0 and delivery_model.steps % args.small_train_per_n_steps == 0:
                delivery_model.train()

        hit_rate = np.array(env.hit_ratio)
        hit_rate = np.mean(hit_rate[hit_rate != -1], axis=0)

        # Compute the main objective reward
        reward_tensor = torch.tensor(
            [
                -reward * env.storage_cost_scale
                + np.sum(cumulated_rewards)
                + hit_rate * 1000
            ],
            dtype=torch.float32,
        )

        # Collect episode information
        info = log_and_collect(
            writer,
            env,
            episode,
            cumulated_rewards,
            cumulated_cache_costs,
            reward_tensor,
        )

        # Reset the environment
        env.reset()

        # Reset the cache environment before storing transition for the large time-scale
        cache_env.reset(new_main_env=env)

        # If in the training phase, store the transition for the large time-scale cache model
        if episode < args.training_episodes:
            next_state_tensor = torch.tensor(cache_env.states, dtype=torch.float32)
            done_tensor = torch.tensor([True], dtype=torch.float32)
            violation_tensor = torch.tensor([0], dtype=torch.float32)

            # Store the transition for the large time-scale cache model
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

            # Train the cache model
            if episode % args.large_train_per_n_eps == 0 and episode > 0:
                cache_model.train()
                pass

        # If in the evaluation phase, collect the metrics
        if episode >= args.training_episodes:
            evaluate.append(info)

    # Aggregate the evaluation metrics
    metrics = aggregate_metrics(evaluate)

    # Save the model and metrics
    torch.save(
        {
            "cache_model": cache_model.agent.actor.state_dict(),
            "delivery_model": delivery_model.agent.actor.state_dict(),
            "metrics": metrics,
            "args": args,
            "evaluate": evaluate,
        },
        f"runs/{current}_{args.name}/model.pth",  # Save the model with the current time and name
    )
