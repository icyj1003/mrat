import numpy as np


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
    deadlines = env.delivery_deadline

    # normalize size and deadline to 0 - 1 range
    sizes_nor = sizes / np.max(sizes)
    deadlines_nor = deadlines / np.max(deadlines)

    cache_matrix = np.zeros((num_edges, num_items), dtype=int)

    for edge in range(num_edges):
        pop = env.popularities[edge]  # [num_items]
        utility = (pop if use_popularity else 1) / (
            (sizes_nor if use_size else 1) * (deadlines_nor if use_deadline else 1)
        )  # element-wise utility for edge
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
