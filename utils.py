from typing import Literal, Union

import numpy as np

from environ import CacheEnv, Environment


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
        init_states=env.cache_states,
        num_edges=args.num_edges,
        num_items=args.num_items,
        item_sizes=env.item_size,
        edge_capacity=env.edge_capacity,
        vehicle_capacity=env.vehicle_capacity,
        edge_cost=env.edge_cost,
        vehicle_cost=env.vehicle_cost,
        cost_scale=env.storage_cost_scale,
    )

    cache_env.reset()
    return env, cache_env
