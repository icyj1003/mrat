import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env import Environment
from ppo import PPO

current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = SummaryWriter(log_dir=f"runs/ppo_{current}")


if __name__ == "__main__":
    num_vehicles = 40
    num_edges = 4
    num_items = 100
    episode = 3000
    steps = 0
    mini_batch_size = 64
    steps_per_batch = 1024
    max_steps_per_episode = 3000

    env = Environment(
        num_vehicles=num_vehicles,
        num_edges=num_edges,
        num_items=num_items,
        item_size_max=100,
        item_size_min=10,
        seed=42,
        delay_weight=1,
        cost_weight=0,
        dt=1,
    )

    env.reset()

    ppo = PPO(
        state_dim=env.state_dim,
        num_action=num_vehicles * env.num_rats,
        action_dim=2,
        actor_hidden_dim=1024,
        critic_hidden_dim=1024,
        lagrange_multiplier_dim=5,
        actor_lr=3e-5,
        critic_lr=3e-5,
        lagrange_lr=100,
        num_epoch=10,
        eps=0.2,
        gamma=0.9,
        writer=writer,
        mini_batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Initialize the environment
    state = env.state
    mask = env.delivery_mask

    for episode in tqdm(range(episode), desc="Episodes"):
        accumulated_reward = 0
        mean_count = 0
        for step in tqdm(range(max_steps_per_episode), desc="Steps"):
            # convert to tensor
            state_tensor = torch.tensor(env.state, dtype=torch.float32)
            mask_tensor = torch.tensor(env.delivery_mask, dtype=torch.float32)

            action, log_prob, value = ppo.act(state_tensor, mask_tensor)
            reshaped_action = action.view(num_vehicles, env.num_rats)
            next_state, reward, done, violation = env.small_step(reshaped_action)

            accumulated_reward += reward
            mean_count += 1

            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            ppo.add(
                state_tensor,
                mask_tensor,
                action,
                log_prob,
                value,
                reward_tensor,
                done,
                violation,
            )

            if ppo.buffer_length() >= steps_per_batch:
                ppo.update()
                ppo.clear()

            if done:
                break

            writer.add_scalar(
                "log/avg_deadline_violations",
                violation[-1],
                global_step=steps,
            )

            steps += 1

        writer.add_scalar(
            "log/accumulated_reward",
            accumulated_reward,
            global_step=episode,
        )
        writer.add_scalar(
            "log/avg_reward",
            accumulated_reward / mean_count,
            global_step=episode,
        )
        writer.add_scalar(
            "log/episode_length",
            mean_count,
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_total_delay",
            np.mean(env.delay),
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_total_cost",
            np.mean(env.cost),
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_cost_per_bit",
            np.mean(env.cost / (env.collected * env.code_size)),
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_delay_per_segment",
            np.mean(env.delay / env.collected * 1e3),
            global_step=episode,
        )

        env.reset()
