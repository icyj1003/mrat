import numpy as np

from environ import CacheEnv, Environment


def log_episode(
    writer, env, episode, accumulated_rewards, cummulated_cache_cost, mode="train"
):
    writer.add_scalar(
        f"{mode}/log/cache_cost",
        -np.sum(cummulated_cache_cost),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/objective",
        np.sum(cummulated_cache_cost) + np.mean(accumulated_rewards),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/episode_length",
        len(accumulated_rewards),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/mean_delay",
        np.mean(env.delay),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/mean_reward",
        np.mean(accumulated_rewards),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/mean_deadline_violation",
        np.clip(np.mean(env.delay - env.delivery_deadline[env.requested]), 0, None),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/delay per segment",
        np.mean(env.delay / env.num_code_min[env.requested]),
        episode,
    )

    writer.add_scalar(
        f"{mode}/log/cost_per_bit",
        np.mean(env.cost / env.item_size[env.requested]),
        episode,
    )


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
        old_cache=env.cache[: args.num_edges, :],
        num_edges=args.num_edges,
        num_items=args.num_items,
        item_sizes=env.item_size,
        edge_capacity=env.edge_capacity,
        vehicle_capacity=env.vehicle_capacity,
        edge_cost=env.edge_cost,
        vehicle_cost=env.vehicle_cost,
        cost_scale=env.storage_cost_scale,
    )

    cache_env.reset(init_states=env.cache_states)
    return env, cache_env
