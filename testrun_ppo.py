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
    episode = 10000
    steps = 0
    mini_batch_size = 32
    steps_per_batch = 1024
    hidden_dim = 128
    lr = 3e-5
    num_epoch = 10
    eps = 0.2  # Constraint for the ratio of the old and new policy (higher is more exploratory)
    gamma = 0.99
    entropy_coeff = 0.01
    penalty_coeff = 1.0
    lam = 0.95
    tau = 1e-2

    env = Environment(
        num_vehicles=num_vehicles,
        num_edges=num_edges,
        num_items=num_items,
        delivery_deadline_min=50,
        delivery_deadline_max=100,
        item_size_max=100,
        item_size_min=50,
        seed=42,
        dt=1,
    )

    env.reset()

    ppo = PPO(
        state_dim=env.state_dim * num_vehicles,  # join all vehicles' states
        num_actions=num_vehicles * env.num_rats,  # join all vehicles' actions
        action_dim=2,  # enable/disable
        hidden_dim=hidden_dim,
        lr=lr,
        num_epoch=num_epoch,
        eps=eps,
        gamma=gamma,
        lam=lam,
        mini_batch_size=mini_batch_size,
        tau=1e-2,
        entropy_coeff=entropy_coeff,
        penalty_coeff=penalty_coeff,
        device="cpu",
        writer=writer,
    )

    for episode in tqdm(range(episode), desc="Training", unit="episode"):
        accumulated_rewards = []
        while not all(env.delivery_done):
            # state tensor (originally for MAPPO): num_vehicles x state_dim --> num_vehicles * state_dim
            state_tensor = torch.tensor(env.states, dtype=torch.float32).view(
                num_vehicles * env.state_dim
            )

            # mask tensor (originally for MAPPO): num_vehicles x num_rats x 2 --> num_vehicles * num_rats x 2
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32).view(
                num_vehicles * env.num_rats, 2
            )

            # get action from PPO
            actions, log_probs = ppo.act(
                state_tensor, mask_tensor, env.greedy_projection
            )

            # reshape action tensor num_vehicles * num_rats --> num_vehicles x num_rats
            actions = actions.view(num_vehicles, env.num_rats)

            # step the environment
            next_states, rewards, dones, violations = env.small_step(actions)

            # convert to tensor
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32).view(
                num_vehicles * env.state_dim
            )
            reward_tensor = torch.tensor([np.mean(rewards)], dtype=torch.float32)
            done_tensor = torch.tensor([float(np.all(dones == 1))])
            violation_tensor = torch.tensor([np.mean(violations)], dtype=torch.float32)

            accumulated_rewards.append(rewards.mean())

            ppo.buffer.add(
                state_tensor,
                mask_tensor,
                actions,
                log_probs,
                reward_tensor,
                next_state_tensor,
                done_tensor,
                violation_tensor,
            )

            if len(ppo.buffer) >= steps_per_batch:
                ppo.update()

            steps += 1

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
