import torch

from config import parse_args
from utils import get_environment


def main():
    args = parse_args()
    env = get_environment(args)

    print("Environment:", type(env).__name__)
    print("Max vehicles:", env.num_vehicles)
    print("Active vehicles:", env.active_num_vehicles)
    print(
        "Deadline range:", args.delivery_deadline_min, "->", args.delivery_deadline_max
    )
    print("State shape:", env.states.shape)
    print("Mask shape:", env.masks.shape)

    episode_rewards = []

    while not env.is_small_done():
        actions = torch.randint(
            low=0,
            high=2,
            size=(env.num_vehicles, env.num_rats),
            dtype=torch.long,
        )
        next_states, rewards, dones, violations = env.small_step(actions)
        episode_rewards.append(float(rewards.mean()))

        print(
            "step=",
            env.steps,
            "reward_mean=",
            float(rewards.mean()),
            "done_count=",
            int(dones.sum()),
            "violation_count=",
            int(violations.sum()),
        )

        if env.steps >= 5:
            break

    print("Episode reward mean:", sum(episode_rewards) / max(len(episode_rewards), 1))


if __name__ == "__main__":
    main()
