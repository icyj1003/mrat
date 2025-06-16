import numpy as np
import torch


class CacheEnv:
    def __init__(
        self,
        main_env,
    ):
        self.num_edges = main_env.num_edges
        self.num_items = main_env.num_items
        self.total_locations = self.num_edges + 1
        self.item_size = main_env.item_size / 8 / 1024 / 1024  # Convert to MB
        self.delivery_deadline = main_env.delivery_deadline
        self.edge_capacity = main_env.edge_capacity
        self.edge_cost = main_env.edge_cost

    def reset(
        self,
        new_main_env,
    ):
        self.requests_edges = new_main_env.requests_edges
        self.old_cache = np.zeros((self.total_locations, self.num_items))
        self.old_cache[: self.num_edges, :] = new_main_env.old_cache[
            : self.num_edges, :
        ]
        self.popularities = new_main_env.popularities

        self.compute_states()

    def compute_states(self):
        # for each edge, states include:
        # requests - shape: num_items
        # old_cache_status - shape: num_items
        # item_sizes - shape: num_items
        # item_delivery_deadline - shape: num_items
        # cost weight - shape: 1
        self.states = np.zeros((self.total_locations, self.num_items * 4))
        for edge in range(self.num_edges):
            self.states[edge, : self.num_items] = self.requests_edges[edge]
            self.states[edge, self.num_items : 2 * self.num_items] = self.old_cache[
                edge, :
            ]
            self.states[edge, 2 * self.num_items : 3 * self.num_items] = self.item_size
            self.states[edge, 3 * self.num_items :] = self.delivery_deadline

        self.masks = np.zeros((self.total_locations * self.num_items * 2))

    def greedy_projection(self, actions):
        # actions: binary selected mask of shape (num_edges x num_items)
        # greedy select item with high popularity until capacity is exceeded
        actions = actions.view(self.num_edges, self.num_items)
        valid_actions = torch.zeros_like(actions)
        for edge in range(self.num_edges):
            edge_capacity = self.edge_capacity
            for item in np.argsort(-self.popularities[edge]):
                if actions[edge, item] == 1 and edge_capacity >= self.item_size[item]:
                    valid_actions[edge, item] = 1
                    edge_capacity -= self.item_size[item]

        return valid_actions.reshape(-1)

    def step(self, actions):
        if actions.device != torch.device("cpu"):
            actions = actions.cpu()

        # new_item for each edge
        new_item = actions * (1 - self.old_cache)

        # new item count for all edges
        new_item_count = new_item.sum(axis=0)

        # get the total cost
        reward = (new_item_count * self.edge_cost * self.item_size).sum().item()

        return reward
