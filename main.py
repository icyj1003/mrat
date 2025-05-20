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
    num_vehicles = 20
    num_edges = 4
    num_items = 100
    episode = 100000
    steps = 0
    mini_batch_size = 32
    steps_per_batch = 4096
    hidden_dim = 128
    lr = 3e-5
    num_epoch = 10
    eps = 0.1  # Constraint for the ratio of the old and new policy (higher is more exploratory)
    gamma = 0.99

    env = Environment(
        num_vehicles=num_vehicles,
        num_edges=num_edges,
        num_items=num_items,
        delivery_deadline_min=50,
        delivery_deadline_max=100,
        item_size_max=100,
        item_size_min=10,
        seed=42,
        dt=1,
    )

    env.reset()

    ppo = PPO(
        state_dim=env.state_dim,
        num_action=num_vehicles * env.num_rats,
        action_dim=2,  # enable/disable
        hidden_dim=hidden_dim,
        lr=lr,
        num_epoch=num_epoch,
        eps=eps,
        gamma=gamma,
        writer=writer,
        mini_batch_size=mini_batch_size,
        tau=1e-3,
        lambd_init=1.0,
        device="cpu" if torch.cuda.is_available() else "cpu",
    )

    # Initialize the environment
    state = env.state
    mask = env.delivery_mask

    for episode in tqdm(range(episode), desc="Episodes"):
        accumulated_reward = 0
        mean_count = 0
        done = False

        while not done:
            # convert to tensor
            state_tensor = torch.tensor(env.state, dtype=torch.float32)
            mask_tensor = torch.tensor(env.delivery_mask, dtype=torch.float32)

            # get action from PPO
            action, log_prob = ppo.act(state_tensor, mask_tensor, env.greedy_projection)

            # reshape action to match the environment
            reshaped_action = action.view(num_vehicles, env.num_rats)

            # step the environment
            next_state, reward, done, violation = env.small_step(reshaped_action)

            # accumulate reward
            accumulated_reward += reward
            mean_count += 1

            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            ppo.add(
                state_tensor,
                mask_tensor,  # num_agents x num_actions x action_dim
                action,  # num_agents * num_actions
                log_prob,  # 1
                reward_tensor,  #
                done,
                violation,
            )

            if ppo.buffer_length() >= steps_per_batch:
                ppo.update()
                ppo.clear()

            steps += 1

        writer.add_scalar(
            "log/avg_violations",
            (env.delay - env.delivery_deadline[env.requested]).clip(0).mean(),
            global_step=episode,
        )

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
            "log/max_delay",
            np.max(env.delay) / env.dt,
            global_step=episode,
        )

        writer.add_scalar(
            "log/min_delay",
            np.min(env.delay) / env.dt,
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_cost_per_bit",
            np.mean(env.cost / (env.collected * env.code_size)),
            global_step=episode,
        )

        writer.add_scalar(
            "log/avg_delay_per_segment",
            np.mean(env.delay / env.collected * 1e3 / env.dt),
            global_step=episode,
        )

        env.reset()
