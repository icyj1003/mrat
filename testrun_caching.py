from tqdm import tqdm
from env import Environment
import torch
from delivery_policy import random_policy

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

    for episode in tqdm(range(episode), desc="Training", unit="episode"):
        accumulated_rewards = []
        # Large timescale step

        # Small timescale step
        while not all(env.delivery_done):
            # convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # MAPPO Strategy
            actions, log_probs = random_policy(
                num_vehicles,
                env.num_rats,
                action_dim=2,  # enable/disable
                masks=mask_tensor,
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

            steps += 1

        env.reset()
