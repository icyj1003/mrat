import numpy as np


def heuristic_cache_placement(env):
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
    num_edges = env.num_edges
    num_items = env.num_items
    sizes = env.item_size / 1024 / 1024 / 8
    deadlines = env.delivery_deadline

    cache_matrix = np.zeros((num_edges, num_items), dtype=int)

    for edge in range(num_edges):
        pop = env.popularities[edge]  # [num_items]
        utility = pop / (sizes * deadlines)  # element-wise utility for edge
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
