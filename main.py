import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from env import Environment
from mappo import MAPPO

current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = SummaryWriter(log_dir=f"runs/ppo_{current}")


def random_actions(num_agents, num_actions, action_dim, masks):
    """
    Generate random actions for the given number of agents, actions, and action dimension.
    num_agents: Number of agents
    num_actions: Number of actions
    action_dim: Dimension of each action
    masks: Mask for the actions [num_agents x num_actions x action_dim]
    """
    random_logits = torch.rand(num_agents, num_actions, action_dim) + masks * -1e10
    distribution = torch.distributions.Categorical(logits=random_logits)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    return actions, log_probs


if __name__ == "__main__":
    num_vehicles = 5
    num_edges = 4
    num_items = 100
    episode = 100000
    steps = 0
    mini_batch_size = 32
    steps_per_batch = 4096
    hidden_dim = 32
    lr = 3e-5
    num_epoch = 10
    eps = 0.1  # Constraint for the ratio of the old and new policy (higher is more exploratory)
    gamma = 0.99

    env = Environment(
        num_vehicles=num_vehicles,
        num_edges=num_edges,
        num_items=num_items,
        delivery_deadline_min=10,
        delivery_deadline_max=50,
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
        lam=0.95,
        tau=1e-3,
        entropy_coef=1e-3,
        lagrange_init=1.0,
        lagrange_lr=1e-3,
        mini_batch_size=mini_batch_size,
        device="cpu" if torch.cuda.is_available() else "cpu",
        shared_critic=True,
    )

    for episode in range(episode):
        accumulated_rewards = []
        while not all(env.delivery_done):
            # convert to tensor
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)

            # Random Strategy
            # actions, log_probs = random_actions(
            #     num_agents=num_vehicles,
            #     num_actions=env.num_rats,
            #     action_dim=2,
            #     masks=mask_tensor,
            # )

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

            accumulated_rewards.append(rewards.mean())

            mappo.buffer.add(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions x action_dim
                actions,  # num_agents x num_actions
                log_probs,  #  num_agents x num_actions
                reward_tensor,  # num_agents x 1
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

        print(f"Episode {episode} finished with reward {sum(accumulated_rewards)}")
        env.reset()
