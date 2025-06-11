from collections import Counter

import numpy as np

from environ import CacheEnv, Environment


def log_episode(writer, env, episode, accumulated_rewards, cummulated_cache_cost):

    mean_reward = np.mean(accumulated_rewards)
    delay_per_segment = np.mean(env.delay / env.num_code_min[env.requested]) * 1000
    cost_per_bit = np.mean(env.cost / env.item_size[env.requested])
    episode_length = len(accumulated_rewards)
    ultility = np.array(env.ultility).mean(axis=0)
    v2n_u = ultility[0]
    v2v_u = ultility[1]
    v2i_wifi_u = ultility[2]
    v2i_pc5_u = ultility[3]
    hit_rate = np.array(env.hit_ratio).mean(axis=0)
    hit_rate_v2i = hit_rate[1]
    cache_cost = -np.sum(cummulated_cache_cost)
    mean_deadline_violation = np.clip(
        np.mean(env.delay - env.delivery_deadline[env.requested]), 0, None
    )

    writer.add_scalar(
        f"log/mean_reward",
        mean_reward,
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

    hit_rate = np.array(env.hit_ratio).mean(axis=0)

    writer.add_scalar(
        f"log/hit_rate_v2i",
        hit_rate_v2i,
        episode,
    )

    writer.add_scalar(
        f"log/cache_cost",
        cache_cost,
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
        "mean_reward": mean_reward,
        "delay_per_segment": delay_per_segment,
        "cost_per_bit": cost_per_bit,
        "v2n_u": v2n_u,
        "v2v_u": v2v_u,
        "v2i_wifi_u": v2i_wifi_u,
        "v2i_pc5_u": v2i_pc5_u,
        "hit_rate_v2i": hit_rate_v2i,
        "cache_cost": cache_cost,
        "mean_deadline_violation": mean_deadline_violation,
        "episode": episode,
    }


def create_environment(args):
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
    )

    env.reset()

    cache_env = CacheEnv(
        main_env=env,
    )

    cache_env.reset(
        new_main_env=env,
    )
    return env, cache_env


def sum_metrics(data):
    out = dict(sum((Counter(d) for d in data), Counter()))
    for k, v in out.items():
        out[k] = float(v)
    return out
