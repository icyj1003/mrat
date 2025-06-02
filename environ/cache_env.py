import numpy as np


class CacheEnv:
    def __init__(
        self,
        old_cache,
        init_states,
        num_edges,
        num_items,
        item_sizes,
        edge_capacity,
        vehicle_capacity,
        edge_cost,
        vehicle_cost,
        cost_scale,
    ):
        self.old_cache = old_cache
        self.init_states = init_states
        self.num_edges = num_edges
        self.num_items = num_items
        self.item_sizes = item_sizes / 8 / 1024 / 1024  # convert to MB
        self.edge_capacity = edge_capacity
        self.vehicle_capacity = vehicle_capacity
        self.total_locations = num_edges + 1  # edges + vehicle
        self.edge_cost = edge_cost
        self.vehicle_cost = vehicle_cost
        self.cost_scale = cost_scale

    def reset(self):
        self.cache_status = np.zeros((self.total_locations, self.num_items))
        self.remaining_capacity = np.ones(self.total_locations) * self.edge_capacity
        self.remaining_capacity[-1] = self.vehicle_capacity

        self.compute_states()
        self.compute_masks()

    def compute_states(self):
        self.states = np.concatenate(
            [self.cache_status, self.init_states],
            axis=0,
        ).flatten()

    def compute_masks(self):
        # for each edge, mask already cached items
        self.masks = self.cache_status.copy()

        # Step 1: Mask items exceeding local capacity
        for idx in range(self.total_locations):
            # Step 1: Mask items exceeding local capacity
            self.masks[idx, :] = np.where(
                self.item_sizes > self.remaining_capacity[idx],
                1,
                self.masks[idx, :],
            )

        # Step 2: Mask neighboring duplication for edge servers only
        for edge_index in range(self.num_edges):  # exclude vehicle
            if edge_index > 0:
                self.masks[edge_index, :] = np.where(
                    self.cache_status[edge_index - 1, :] == 1,
                    1,
                    self.masks[edge_index, :],
                )
            if edge_index < self.num_edges - 1:
                self.masks[edge_index, :] = np.where(
                    self.cache_status[edge_index + 1, :] == 1,
                    1,
                    self.masks[edge_index, :],
                )

    def step(self, action):
        loc_index = action // self.num_items
        item_index = action % self.num_items

        self.cache_status[loc_index, item_index] = 1
        reward = 0

        # if the current item is not cached on the vehicle, update the cost
        if loc_index < self.num_edges:
            if self.old_cache[loc_index, item_index] == 0:
                reward += -(
                    self.edge_cost
                    * self.cost_scale
                    * self.item_sizes[item_index]
                    * 1024
                    * 1024
                    * 8
                )
        else:
            reward += (
                self.vehicle_cost
                * self.cost_scale
                * self.item_sizes[item_index]
                * 1024
                * 1024
                * 8
            )

        self.remaining_capacity[loc_index] -= self.item_sizes[item_index]

        self.compute_states()
        self.compute_masks()

        return reward, loc_index

    def is_done(self):
        return np.all(
            self.masks == 1
        )  # all items are masked (cached or exceed capacity)
