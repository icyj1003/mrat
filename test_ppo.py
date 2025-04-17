import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo import PPO

current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = SummaryWriter(log_dir=f"runs/ppo_{current}")


class SimpleEnv:
    def reset(self):
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        reward = sum(1.0 if a == 0 else 0.0 for a in action)
        return np.zeros(4, dtype=np.float32), reward, False, {}


if __name__ == "__main__":
    # Initialize the environment
    state_dim = 4
    num_action = 40
    action_dim = 2
    actor_hidden_dim = 64
    critic_hidden_dim = 64
    lagrange_multiplier_dim = 4
    actor_lr = 3e-4
    critic_lr = 3e-4
    lagrange_lr = 1e-2
    num_epoch = 10
    eps = 0.01
    mini_batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimpleEnv()
    ppo = PPO(
        state_dim=state_dim,
        num_action=num_action,
        action_dim=action_dim,
        actor_hidden_dim=actor_hidden_dim,
        critic_hidden_dim=critic_hidden_dim,
        lagrange_multiplier_dim=lagrange_multiplier_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lagrange_lr=lagrange_lr,
        num_epoch=num_epoch,
        eps=eps,
        writer=writer,
        mini_batch_size=mini_batch_size,
        device=device,
    )
    # Initialize the environment
    state = env.reset()
    mask = np.zeros((num_action, action_dim), dtype=np.float32)

    # Training loop
    for episode in tqdm(range(10000), desc="Episodes"):
        accumulated_reward = 0
        mean_count = 0
        for step in tqdm(range(1024), desc="Steps"):
            # convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

            action, log_prob, value = ppo.act(state_tensor, mask_tensor)
            reshaped_action = action.view(num_action, 1)
            next_state, reward, done, _ = env.step(reshaped_action)

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
                [0.0, 0.0, 0.0, 0.0],
            )

            if ppo.buffer_length() >= 1024:
                ppo.update()
                ppo.clear()

            if done:
                break

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

        state = env.reset()
