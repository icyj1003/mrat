import numpy as np


def _cache_utility(
    env, edge_index, use_deadline=True, use_popularity=True, use_size=True
):
    num_items = env.num_items
    sizes = env.item_size / 1024 / 1024 / 8
    deadlines = env.delivery_deadline

    sizes_nor = sizes / np.max(sizes)
    deadlines_nor = deadlines / np.max(deadlines)

    popularity = env.popularities[edge_index] if use_popularity else np.ones(num_items)
    size_term = sizes_nor if use_size else np.ones(num_items)
    deadline_term = deadlines_nor if use_deadline else np.ones(num_items)

    return popularity / (size_term * deadline_term)


def heuristic_cache_placement(
    env, use_deadline=True, use_popularity=True, use_size=True
):
    """
    Generate a [num_edges x num_items] binary cache placement matrix
    using a greedy value-based knapsack approach, with edge-specific popularity.

    Args:
        env: Environment object with:
            - env.item_size: [num_items]
            - env.popularities: [num_edges, num_items]
            - env.deadline: [num_items]
            - env.edge_capacity: scalar or [num_edges]
            - env.num_edges: int
            - env.num_items: int

    Returns:
        cache_matrix: np.ndarray of shape [num_edges, num_items]
                      Binary matrix indicating caching decisions
    """
    assert (
        use_deadline or use_popularity or use_size
    ), "At least one of use_deadline, use_popularity, or use_size must be True."

    num_edges = env.num_edges
    num_items = env.num_items
    sizes = env.item_size / 1024 / 1024 / 8

    cache_matrix = np.zeros((num_edges, num_items), dtype=int)

    for edge in range(num_edges):
        utility = _cache_utility(
            env,
            edge,
            use_deadline=use_deadline,
            use_popularity=use_popularity,
            use_size=use_size,
        )
        sorted_indices = np.argsort(-utility)  # descending sort

        remaining_capacity = (
            env.edge_capacity
            if np.isscalar(env.edge_capacity)
            else env.edge_capacity[edge]
        )

        for idx in sorted_indices:
            if sizes[idx] <= remaining_capacity:
                cache_matrix[edge, idx] = 1
                remaining_capacity -= sizes[idx]

    return cache_matrix


def non_redundant_cache_placement(
    env,
    caching_vehicle_indices=None,
    use_deadline=True,
    use_popularity=True,
    use_size=True,
    priority="veh",
):
    """
    Generate split cache matrices for vehicles and RSUs without redundancy.

    Vehicles under the current RSU coverage are filled first using the same
    greedy priority rule as the edge cache. The RSU then stores the remaining
    items that were not already placed in any vehicle cache for that edge.

    Args:
        env: Environment object.
        caching_vehicle_indices: Indices of vehicles selected for caching.
        use_deadline: Include deadline in the utility score.
        use_popularity: Include edge popularity in the utility score.
        use_size: Include item size in the utility score.

    Returns:
        vehicle_cache: np.ndarray of shape [num_vehicles, num_items]
        rsu_cache: np.ndarray of shape [num_edges, num_items]
    """
    assert (
        use_deadline or use_popularity or use_size
    ), "At least one of use_deadline, use_popularity, or use_size must be True."

    num_edges = env.num_edges
    num_vehicles = env.num_vehicles
    num_items = env.num_items
    sizes = env.item_size / 1024 / 1024 / 8

    vehicle_cache = np.zeros((num_vehicles, num_items), dtype=int)
    rsu_cache = np.zeros((num_edges, num_items), dtype=int)

    if caching_vehicle_indices is None:
        caching_vehicle_indices = range(num_vehicles)

    selected_vehicles_by_edge = {edge: [] for edge in range(num_edges)}
    for vehicle_index in caching_vehicle_indices:
        selected_vehicles_by_edge[env.local_of[vehicle_index]].append(vehicle_index)

    for edge in range(num_edges):
        utility = _cache_utility(
            env,
            edge,
            use_deadline=use_deadline,
            use_popularity=use_popularity,
            use_size=use_size,
        )
        sorted_indices = np.argsort(-utility)
        used_items = np.zeros(num_items, dtype=bool)

        # if priority == "veh", fill vehicle caches first, otherwise fill RSU cache first

        if priority == "veh":
            for vehicle_index in selected_vehicles_by_edge[edge]:
                remaining_capacity = env.vehicle_capacity
                for idx in sorted_indices:
                    if used_items[idx]:
                        continue
                    if sizes[idx] <= remaining_capacity:
                        vehicle_cache[vehicle_index, idx] = 1
                        used_items[idx] = True
                        remaining_capacity -= sizes[idx]
                    if remaining_capacity <= 0:
                        break

            remaining_capacity = (
                env.edge_capacity
                if np.isscalar(env.edge_capacity)
                else env.edge_capacity[edge]
            )
            for idx in sorted_indices:
                if used_items[idx]:
                    continue
                if sizes[idx] <= remaining_capacity:
                    rsu_cache[edge, idx] = 1
                    used_items[idx] = True
                    remaining_capacity -= sizes[idx]
                if remaining_capacity <= 0:
                    break
        else:
            remaining_capacity = (
                env.edge_capacity
                if np.isscalar(env.edge_capacity)
                else env.edge_capacity[edge]
            )
            for idx in sorted_indices:
                if sizes[idx] <= remaining_capacity:
                    rsu_cache[edge, idx] = 1
                    used_items[idx] = True
                    remaining_capacity -= sizes[idx]
                if remaining_capacity <= 0:
                    break

            for vehicle_index in selected_vehicles_by_edge[edge]:
                remaining_capacity = env.vehicle_capacity
                for idx in sorted_indices:
                    if used_items[idx]:
                        continue
                    if sizes[idx] <= remaining_capacity:
                        vehicle_cache[vehicle_index, idx] = 1
                        used_items[idx] = True
                        remaining_capacity -= sizes[idx]
                    if remaining_capacity <= 0:
                        break

    return vehicle_cache, rsu_cache


def random_cache_placement(env):
    """
    Generate a random cache placement matrix for the environment.

    Args:
        env: Environment object with:
            - env.num_edges: int
            - env.num_items: int

    Returns:
        cache_matrix: np.ndarray of shape [num_edges, num_items]
                      Binary matrix indicating caching decisions
    """
    num_edges = env.num_edges
    num_items = env.num_items
    sizes = env.item_size / 1024 / 1024 / 8

    cache_matrix = np.zeros((num_edges, num_items), dtype=int)

    for edge in range(num_edges):
        remaining_capacity = (
            env.edge_capacity
            if np.isscalar(env.edge_capacity)
            else env.edge_capacity[edge]
        )
        item_idx = np.random.choice(np.where(cache_matrix[edge] == 0)[0])
        while remaining_capacity - sizes[item_idx] >= 0:
            if cache_matrix[edge, item_idx] == 0:
                cache_matrix[edge, item_idx] = 1
                remaining_capacity -= sizes[item_idx]
            item_idx = np.random.choice(np.where(cache_matrix[edge] == 0)[0])

    return cache_matrix


def no_cache_placement(env):
    """
    Generate a cache placement matrix with no caching.

    Args:
        env: Environment object with:
            - env.num_edges: int
            - env.num_items: int

    Returns:
        cache_matrix: np.ndarray of shape [num_edges, num_items]
                      Binary matrix indicating caching decisions
    """
    num_edges = env.num_edges
    num_items = env.num_items

    # Create a zero matrix indicating no caching
    cache_matrix = np.zeros((num_edges, num_items), dtype=int)

    return cache_matrix
