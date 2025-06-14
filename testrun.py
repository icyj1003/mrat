import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import parse_args
from policy.cache_policy import PPOCachePolicy
from policy.delivery_policy import MAPPODeliveryPolicy
from policy.selection_policy import GTVS
from utils import create_environment, log_episode, sum_metrics

if __name__ == "__main__":
    args = parse_args()
    current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{current}_{args.name}")
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
    # define stage
    # delivery training stage: episode in [0, delivery_training_episodes - 1]
    # cache training stage: episode in [delivery_training_episodes, delivery_training_episodes + cache_training_episodes - 1]
    # joint training stage: episode in [delivery_training_episodes + cache_training_episodes, delivery_training_episodes + cache_training_episodes + joint_training_episodes - 1]
    # evaluation stage: episode in [delivery_training_episodes + cache_training_episodes + joint_training_episodes, delivery_training_episodes + cache_training_episodes + joint_training_episodes + evaluation_episodes - 1]

    def get_stage(episode):
        if episode < args.delivery_training_episodes:
            return "delivery"
        elif episode < args.delivery_training_episodes + args.cache_training_episodes:
            return "cache"
        elif (
            episode
            < args.delivery_training_episodes
            + args.cache_training_episodes
            + args.joint_training_episodes
        ):
            return "joint"
        else:
            return "evaluation"

    total_episodes = (
        args.delivery_training_episodes
        + args.cache_training_episodes
        + args.joint_training_episodes
        + args.evaluation_episodes
    )

    evaluate = []

    # Begin training loop
    for episode in tqdm(range(total_episodes), desc="Running", unit="episode"):
        cumulated_rewards = []
        cumulated_cache_costs = []
        stage = get_stage(episode)
        # At Large time-scale:
        # Step 1: Run the vehicle selection policy here
        cache_vehicle = GTVS(env)

        # Step 2: Make caching decisions
        # If in the delivery training stage: Use random cache policy
        if stage == "delivery":
            env.all_large_step(cache_vehicle)
        # Else: Use the cache model to select the cache actions
        else:
            # Get the states and masks from the environment
            cache_states = torch.tensor(cache_env.states, dtype=torch.float32)
            cache_masks = torch.tensor(cache_env.masks, dtype=torch.float32)

            # Get the cache actions using the cache model
            cache_actions, cache_log_probs = cache_model.act(
                cache_states,  # num_edges * num_items * 2 + num_items * 2
                cache_masks,  # (num_edges + 1) * num_items
                projection=cache_env.greedy_projection,  # Greedy projection for cache selection
            )

            # Reshape the actions to match the environment
            reshaped_actions = cache_actions.view(
                cache_env.num_edges, cache_env.num_items
            )

            # Get the cache replacement costs
            reward = cache_env.step(reshaped_actions)

            # cumulate the cache costs
            cumulated_cache_costs.append(reward)

            # Overwrite the cache states in the environment before performing the small step
            env.large_step(reshaped_actions)

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

            # If in the delivery training stage, store the transition and trigger training
            if stage == "delivery" or stage == "joint":
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
                if episode > 0 and delivery_model.steps % 1024 == 0:
                    delivery_model.train()
            # Else if in the cache training stage, clear the buffer
            else:
                delivery_model.agent.buffer.clear()

        info = log_episode(
            writer,
            env,
            episode,
            cumulated_rewards,
            cumulated_cache_costs,
            stage,
        )

        # Reset the environment
        env.reset()

        # Reset the cache environment before storing transition for the large time-scale
        cache_env.reset(new_main_env=env)

        # If in the cache training stage or joint training stage, store the transition for the cache model and trigger training
        if stage == "cache" or stage == "joint":
            # if stage == "cache" and len(delivery_model.agent.buffer) > 0:
            #     delivery_model.train()

            reward_tensor = torch.tensor(
                # [-reward * env.storage_cost_scale + np.sum(cumulated_rewards)],
                [np.sum(cumulated_rewards)],
                dtype=torch.float32,
            )
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

            # If not in the warmup phase, train the cache model
            if (
                episode % args.large_train_per_n_eps == 0
                and len(cache_model.agent.buffer) > 1
            ):
                cache_model.train()

        # If in the evaluation phase, collect the metrics
        if stage == "evaluation":
            evaluate.append(info)

    metrics = sum_metrics(evaluate)

    # save metrics, cache model, and delivery model
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
