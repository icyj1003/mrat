from collections import Counter
import datetime

import numpy as np

from environ import Environment
from torch.utils.tensorboard import SummaryWriter


def log_and_collect(writer, env, episode):
    # cumulative_reward
    cumulative_reward = np.sum(env.rewards_track)

    # delay per segment
    delay_per_segment = (
        np.mean(env.delay / env.num_code_min[env.requested]) * 1000
    )  # to ms

    # cost per bit
    cost_per_bit = np.mean(env.cost / (env.collected * env.code_size))

    # episode length
    episode_length = len(env.rewards_track)

    # utility
    v2n_u, v2v_u, v2i_pc5_u, v2i_wifi_u = env.compute_utility()

    # deadline violation
    mean_deadline_violation = np.clip(
        np.mean(env.delay - env.delivery_deadline[env.requested]), 0, None
    )

    # v2v hit-ratio
    hit_rate = env.compute_hit_ratio()

    writer.add_scalar(
        f"log/cumulative_reward",
        cumulative_reward,
        episode,
    )

    writer.add_scalar(
        f"log/delay_per_segment",
        delay_per_segment,
        episode,
    )

    writer.add_scalar(
        f"log/cost_per_bit",
        cost_per_bit,
        episode,
    )

    writer.add_scalar(
        f"log/v2n_u",
        v2n_u,
        episode,
    )
    writer.add_scalar(
        f"log/v2v_u",
        v2v_u,
        episode,
    )
    writer.add_scalar(
        f"log/v2i_wifi_u",
        v2i_wifi_u,
        episode,
    )
    writer.add_scalar(
        f"log/v2i_pc5_u",
        v2i_pc5_u,
        episode,
    )

    writer.add_scalar(
        f"log/hit_rate_v2i",
        hit_rate,
        episode,
    )

    writer.add_scalar(
        f"log/episode_length",
        episode_length,
        episode,
    )

    writer.add_scalar(
        f"log/mean_deadline_violation",
        mean_deadline_violation,
        episode,
    )

    return {
        "cumulative_reward": cumulative_reward,
        "episode_length": episode_length,
        "delay_per_segment": delay_per_segment,
        "cost_per_bit": cost_per_bit,
        "v2n_u": v2n_u,
        "v2v_u": v2v_u,
        "v2i_wifi_u": v2i_wifi_u,
        "v2i_pc5_u": v2i_pc5_u,
        "v2i_hit_rate": hit_rate,
        "mean_deadline_violation": mean_deadline_violation,
        "episode": episode,
    }


def get_environment(args):
    # Create the environment
    env = Environment(
        num_vehicles=args.num_vehicles,
        num_edges=args.num_edges,
        num_items=args.num_items,
        delivery_deadline_min=args.delivery_deadline_min,
        delivery_deadline_max=args.delivery_deadline_max,
        item_size_max=args.item_size_max,
        item_size_min=args.item_size_min,
        seed=args.seed,
        dt=args.dt,
        cost_weight=args.cost_weight,
        delay_weight=args.delay_weight,
        disable_v2v=args.remove_v2v,
        disable_wifi=args.remove_wifi,
        disable_pc5=args.remove_pc5,
        remove_edge_cooperation=args.remove_edge_cooperation,
    )

    # Reset the environment
    env.reset()

    return env


def aggregate_metrics(data):
    len_data = len(data)
    out = dict(sum((Counter(d) for d in data), Counter()))
    for k, v in out.items():
        out[k] = float(v) / len_data
    return out


def get_logger(args):
    current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{current}_{args.name}")
    return current, writer
