import numpy as np


class CacheEnv:
    def __init__(
        self,
        main_env,
    ):
        self.num_edges = main_env.num_edges
        self.num_items = main_env.num_items
        self.total_locations = self.num_edges + 1
        self.edge_capacity = main_env.edge_capacity
        self.vehicle_capacity = main_env.vehicle_capacity
        self.item_sizes = main_env.item_size / 8 / 1024 / 1024  # convert to MB
        self.item_deadlines = main_env.delivery_deadline
        self.old_cache = main_env.cache.copy()
        self.output_dim = (self.num_edges + 1) * self.num_items * 4
        self.edge_cost = main_env.edge_cost
        self.vehicle_cost = main_env.vehicle_cost
        self.scale = main_env.storage_cost_scale

    def reset(
        self,
        new_main_env,
    ):
        self.cache_status = np.zeros((self.total_locations, self.num_items))
        self.old_cache = new_main_env.cache.copy()
        self.popularity = new_main_env.popularity.copy()
        self.remaining_capacity = np.ones(self.total_locations) * self.edge_capacity
        self.remaining_capacity[-1] = self.vehicle_capacity

        self.compute_states()
        self.compute_masks()

    def compute_states(
        self,
    ):
        # state will have shape m_i_k x dim in (num_edges + 1) * num_items * dim
        # where m_{i,k} = [item_size[i], item_deadline[i], remaining_capacity[k], cache_status[k, i], old_cache[k, i], popularity[i], cost_multiplier]
        self.states = np.zeros((self.total_locations * self.num_items, 7))
        for i in range(self.num_items):
            for k in range(self.total_locations):
                self.states[i + k * self.num_items, 0] = self.item_sizes[i]
                self.states[i + k * self.num_items, 1] = self.item_deadlines[i]
                self.states[i + k * self.num_items, 2] = self.remaining_capacity[k]
                self.states[i + k * self.num_items, 3] = self.cache_status[k, i]
                self.states[i + k * self.num_items, 4] = self.old_cache[k, i]
                self.states[i + k * self.num_items, 5] = self.popularity[i]
                self.states[i + k * self.num_items, 6] = (
                    1  # 1 for edge, 1 init for vehicle
                )

    def update_num_vehicle(self, num_vehicle):
        self.num_vehicles = num_vehicle
        self.states[self.num_edges * self.num_items :, 6] = (
            num_vehicle  # num_vehicles as cost multiplier for vehicle
        )

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
        if self.old_cache[loc_index, item_index] != 1:
            if loc_index < self.num_edges:
                reward = -self.edge_cost * self.item_sizes[item_index]
            else:
                reward = (
                    -self.vehicle_cost * self.item_sizes[item_index] * self.num_vehicles
                )
        else:
            reward = 0

        reward *= self.scale

        self.remaining_capacity[loc_index] -= self.item_sizes[item_index]

        self.compute_states()
        self.compute_masks()

        return reward, loc_index

    def is_done(self):
        return np.all(
            self.masks == 1
        )  # all items are masked (cached or exceed capacity)
