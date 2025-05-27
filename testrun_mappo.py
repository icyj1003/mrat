import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env import Environment
from mappo import MAPPO

current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = SummaryWriter(log_dir=f"runs/mappo_{current}")


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
        item_size_min=10,
        seed=42,
        dt=1,
    )

    env.reset()

    mappo = MAPPO(
        num_agents=num_vehicles,
        num_actions=env.num_rats,
        action_dim=2,
        state_dim=env.state_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        num_epochs=num_epoch,
        eps=eps,
        gamma=gamma,
        lam=lam,
        tau=tau,
        entropy_coeff=entropy_coeff,
        penalty_coeff=penalty_coeff,
        mini_batch_size=mini_batch_size,
        device="cpu",
        shared_critic=True,
        shared_actor=True,
        writer=writer,
    )

    for episode in tqdm(range(episode), desc="Training", unit="episode"):
        accumulated_rewards = []
        while not all(env.delivery_done):
            # convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # MAPPO Strategy
            actions, log_probs = mappo.act(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions x action_dim
                projection=env.greedy_projection,
            )

            # reshape action to match the environment
            reshaped_actions = actions.view(num_vehicles, env.num_rats)

            # step the environment
            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

            # convert to tensor
            reward_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            done_tensor = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
            violation_tensor = torch.tensor(violations, dtype=torch.float32).view(-1, 1)
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32)

            accumulated_rewards.append(rewards.mean())

            mappo.buffer.add(
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

            if steps == steps_per_batch:
                # update the policy
                mappo.update()
                steps = 0

                # clear the buffer
                mappo.buffer.clear()

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
