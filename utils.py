from collections import Counter
import datetime
import math

import numpy as np

from environ import Environment
from torch.utils.tensorboard import SummaryWriter


def log_and_collect(writer, env, episode):
    # cumulative_reward
    cumulative_reward = np.sum(env.rewards_track)

    # average activated links per vehicle per step
    # only compute upto vehicle delay
    activated_link_history = np.array(env.utility_track)
    avg_activated_links = 0
    for idx, vehicle_delay in enumerate(env.delay):
        avg_activated_links += (
            activated_link_history[: int(vehicle_delay), idx, :].sum(axis=-1).mean()
        ) / env.num_vehicles

    # total collected segments by RAT / path type
    segment_history = np.array(env.segments_classification_track)
    segment_totals = (
        np.sum(segment_history, axis=(0, 1))
        if segment_history.size > 0
        else np.zeros(8)
    )

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

    # violation ratio
    violation_ratio = np.mean(env.delay > env.delivery_deadline[env.requested])

    # v2v hit-ratio
    hit_rate = env.compute_hit_ratio()

    writer.add_scalar(
        f"log/avg_activated_links",
        avg_activated_links,
        episode,
    )

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
        f"log/segments_v2n",
        segment_totals[0],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2v",
        segment_totals[1],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_pc5",
        segment_totals[2],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_wifi",
        segment_totals[3],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_pc5_remote_to_local",
        segment_totals[4],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_pc5_bs_to_edge_to_local",
        segment_totals[5],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_wifi_remote_to_local",
        segment_totals[6],
        episode,
    )
    writer.add_scalar(
        f"log/segments_v2i_wifi_bs_to_edge_to_local",
        segment_totals[7],
        episode,
    )

    writer.add_scalar(
        f"log/mean_deadline_violation",
        mean_deadline_violation,
        episode,
    )

    return {
        "cumulative_reward": cumulative_reward,
        "avg_activated_links": avg_activated_links,
        "episode_length": episode_length,
        "delay_per_segment": delay_per_segment,
        "cost_per_bit": cost_per_bit,
        "v2n_u": v2n_u,
        "v2v_u": v2v_u,
        "v2i_wifi_u": v2i_wifi_u,
        "v2i_pc5_u": v2i_pc5_u,
        "v2i_hit_rate": hit_rate,
        "mean_deadline_violation": mean_deadline_violation,
        "violation_ratio": violation_ratio,
        "segments_v2n": segment_totals[0],
        "segments_v2v": segment_totals[1],
        "segments_v2i_pc5": segment_totals[2],
        "segments_v2i_wifi": segment_totals[3],
        "segments_v2i_pc5_remote_to_local": segment_totals[4],
        "segments_v2i_pc5_bs_to_edge_to_local": segment_totals[5],
        "segments_v2i_wifi_remote_to_local": segment_totals[6],
        "segments_v2i_wifi_bs_to_edge_to_local": segment_totals[7],
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
        v2n_bandwidth_max=args.v2n_bandwidth_max,
        v2v_bandwidth_max=args.v2v_bandwidth_max,
        v2i_pc5_bandwidth_max=args.v2i_pc5_bandwidth_max,
        v2i_wifi_bandwidth_max=args.v2i_wifi_bandwidth_max,
    )

    # Reset the environment
    env.reset()

    return env


def aggregate_metrics(data):
    len_data = len(data)
    # Aggregate counts
    out = dict(sum((Counter(d) for d in data), Counter()))

    results = {}
    for k in out.keys():
        values = [d.get(k, 0) for d in data]  # collect occurrences per dict

        if k.startswith("segments"):
            # For "segments*" keys, compute sum
            total = sum(values)
            results[k] = {"value": total, "std": 0}
        else:
            # For other keys, compute mean and std
            mean_val = float(out[k]) / len_data
            variance = sum((x - mean_val) ** 2 for x in values) / len_data
            std_val = math.sqrt(variance)
            results[k] = {"value": mean_val, "std": std_val}

    return results


def get_logger(args):
    current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{current}_{args.name}")
    return current, writer
