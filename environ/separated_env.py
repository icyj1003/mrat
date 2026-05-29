import numpy as np
import torch

from environ.env import Environment
from environ.utils import compute_data_rate, zipf
from environ.markov import MarkovTransitionModel


class SeparatedEnvironment(Environment):
    """
    Generalized centralized environment for parameter-shared multi-agent PPO.

    The environment keeps a fixed maximum number of agent slots for the policy,
    but samples a smaller active vehicle count per episode. Inactive vehicle
    slots are padded and masked out.
    """

    def __init__(
        self,
        num_vehicles_min: int = 5,
        randomize_vehicle_count: bool = True,
        *args,
        **kwargs,
    ):
        self.num_vehicles_min = max(1, num_vehicles_min)
        self.randomize_vehicle_count = randomize_vehicle_count
        self.delivery_deadline_min = kwargs.get("delivery_deadline_min", 30)
        self.delivery_deadline_max = kwargs.get("delivery_deadline_max", 60)

        super().__init__(*args, **kwargs)

        self.max_vehicles = self.num_vehicles
        self.active_num_vehicles = self.max_vehicles
        self.active_vehicle_mask = np.ones(self.max_vehicles, dtype=bool)

        self.vehicle_state_dim = 6
        self.link_state_dim = 4
        self.network_state_dim = 4
        self.agent_id_dim = self.max_vehicles

    def reset(self) -> None:
        self.steps = 0
        self.utility_track = []
        self.hit_ratio_track = []
        self.rewards_track = []
        self.load_ratios_track = []
        self.reset_mobility()
        self.reset_request()
        self.set_states()

    def reset_mobility(self, sigma: float = 0.5) -> None:
        super().reset_mobility(sigma=sigma)

        if self.randomize_vehicle_count:
            self.active_num_vehicles = int(
                self.np_random.randint(self.num_vehicles_min, self.max_vehicles + 1)
            )
        else:
            self.active_num_vehicles = self.max_vehicles

        self.active_vehicle_mask = np.zeros(self.max_vehicles, dtype=bool)
        self.active_vehicle_mask[: self.active_num_vehicles] = True

        inactive_mask = ~self.active_vehicle_mask
        if np.any(inactive_mask):
            self.positions[inactive_mask, 0] = -1.0
            self.positions[inactive_mask, 1] = -1.0
            self.velocities[inactive_mask] = 0.0
            self.direction[inactive_mask] = 1

        self.update_mobility_status()

        if np.any(inactive_mask):
            self.out[inactive_mask] = 1
            self.local_of[inactive_mask] = -1
            self.local_edge_distance[inactive_mask] = 1e9
            self.bs_distance[inactive_mask] = 1e9
            self.vehicle_distance[inactive_mask, :] = 1e9
            self.vehicle_distance[:, inactive_mask] = 1e9
            np.fill_diagonal(self.vehicle_distance, 0.0)

    def reset_request(self) -> None:
        self.cache[self.num_edges :, :] = 0

        # Randomize the deadline range per episode so the policy sees varying urgency.
        self.delivery_deadline = self.np_random.randint(
            self.delivery_deadline_min,
            self.delivery_deadline_max + 1,
            size=self.num_items,
        )

        for alpha in self.alphas:
            alpha.value = alpha.step()

        self.popularities = []
        for edge in range(self.num_edges):
            popularity = np.zeros(self.num_items)
            popularity[self.ranks[edge]] = zipf(self.num_items, self.alphas[edge].value)
            self.popularities.append(popularity)

        self.requests_matrix = np.zeros((self.max_vehicles, self.num_items))
        self.requests_edges = np.zeros((self.num_edges, self.num_items))

        active_indices = np.where(self.active_vehicle_mask)[0]
        for vehicle_index in active_indices:
            edge_index = int(self.local_of[vehicle_index])
            if edge_index < 0:
                continue

            requested_item = self.np_random.choice(
                self.num_items, size=1, p=self.popularities[edge_index]
            )
            self.requests_matrix[vehicle_index, requested_item] = 1
            self.requests_edges[edge_index, requested_item] = 1

        self.requested = np.argmax(self.requests_matrix, axis=1)

        self.delivery_done = np.zeros(self.max_vehicles)
        self.collected = np.zeros(self.max_vehicles)
        self.delay = np.zeros(self.max_vehicles)
        self.cost = np.zeros(self.max_vehicles)

        inactive_mask = ~self.active_vehicle_mask
        self.delivery_done[inactive_mask] = 1

    def compute_deadline_violation(self):
        deadline_cost = np.where(
            (self.delay - self.delivery_deadline[self.requested]) > 0, 1, 0
        ).reshape(-1, 1)
        deadline_cost = deadline_cost * (1 - self.delivery_done.reshape(-1, 1))
        deadline_cost[~self.active_vehicle_mask.reshape(-1, 1)] = 0
        return deadline_cost

    def compute_utility(self) -> np.ndarray:
        active_mask = self.active_vehicle_mask
        if len(self.utility_track) == 0 or not np.any(active_mask):
            return 0.0, 0.0, 0.0, 0.0

        utility = (
            np.array(self.utility_track)[:, active_mask, :].sum(axis=0)
            / np.maximum(self.delay[active_mask].reshape(-1, 1), 1e-8)
        ).mean(axis=0)
        return utility[0], utility[1], utility[2], utility[3]

    def compute_hit_ratio(self) -> np.ndarray:
        hit_rate = np.array(self.hit_ratio_track)
        hit_rate = np.mean(hit_rate[hit_rate != -1], axis=0)
        return hit_rate

    def set_states(self) -> None:
        super().set_states()

        active_mask = self.active_vehicle_mask.astype(float).reshape(-1, 1)
        active_vehicle_mask = self.active_vehicle_mask

        vehicle_state = np.zeros((self.max_vehicles, self.vehicle_state_dim))
        link_state = np.zeros((self.max_vehicles, self.link_state_dim))

        max_remaining_segments = max(float(np.max(self.num_code_min)), 1.0)
        max_deadline = max(float(self.delivery_deadline_max), 1.0)

        vehicle_state[:, 0] = np.clip(self.positions[:, 0] / self.road_length, 0, 1)
        vehicle_state[:, 1] = np.clip(self.velocities / max(self.vmax, 1e-8), 0, 1)
        vehicle_state[:, 2] = np.clip(
            self.remaining_segments.reshape(-1) / max_remaining_segments,
            0,
            1,
        )
        vehicle_state[:, 3] = np.clip(
            self.remaining_deadline.reshape(-1) / max_deadline,
            0,
            1,
        )
        vehicle_state[:, 4] = np.clip(
            self.requested.reshape(-1) / max(self.num_items - 1, 1), 0, 1
        )
        vehicle_state[:, 5] = np.where(
            self.local_of >= 0,
            self.local_of / max(self.num_edges - 1, 1),
            0,
        )

        for vehicle_index in range(self.max_vehicles):
            if not active_vehicle_mask[vehicle_index]:
                continue

            link_state[vehicle_index, 3] = np.clip(
                compute_data_rate(
                    allocated_spectrum=self.v2n_bandwidth,
                    transmission_power=self.v2n_transmission_power,
                    noise_power=self.noise_power,
                    distance=float(self.bs_distance[vehicle_index]),
                    path_loss_model="macro",
                )
                / max(self.v2n_bandwidth_max, 1.0),
                0,
                1,
            )

            if self.out[vehicle_index] == 0:
                distance = float(self.local_edge_distance[vehicle_index])
                local_edge = int(self.local_of[vehicle_index])

                if not self.disable_v2v:
                    has_neighbor = any(
                        vehicle_index != other_index
                        and self.vehicle_distance[vehicle_index, other_index]
                        < self.v2v_pc5_coverage
                        and self.cache[
                            self.num_edges + other_index, self.requested[vehicle_index]
                        ]
                        == 1
                        for other_index in range(self.max_vehicles)
                        if self.active_vehicle_mask[other_index]
                    )
                    if has_neighbor:
                        link_state[vehicle_index, 0] = np.clip(
                            compute_data_rate(
                                allocated_spectrum=self.v2v_bandwidth,
                                transmission_power=self.v2v_transmission_power,
                                noise_power=self.noise_power,
                                distance=max(1.0, float(self.v2v_pc5_coverage) / 2),
                                path_loss_model="micro",
                            )
                            / max(self.v2v_bandwidth_max, 1.0),
                            0,
                            1,
                        )

                if not self.disable_pc5 and distance < self.v2i_pc5_coverage:
                    link_state[vehicle_index, 1] = np.clip(
                        compute_data_rate(
                            allocated_spectrum=self.v2i_pc5_bandwidth,
                            transmission_power=self.v2i_pc5_transmission_power,
                            noise_power=self.noise_power,
                            distance=distance,
                            path_loss_model="micro",
                        )
                        / max(self.v2i_pc5_bandwidth_max, 1.0),
                        0,
                        1,
                    )

                if not self.disable_wifi and distance < self.v2i_wifi_coverage:
                    link_state[vehicle_index, 2] = np.clip(
                        compute_data_rate(
                            allocated_spectrum=self.v2i_wifi_bandwidth,
                            transmission_power=self.v2i_wifi_transmission_power,
                            noise_power=self.noise_power,
                            distance=distance,
                            path_loss_model="micro",
                        )
                        / max(self.v2i_wifi_bandwidth_max, 1.0),
                        0,
                        1,
                    )

        network_state = np.array(
            [
                self.active_num_vehicles / max(self.max_vehicles, 1),
                (
                    float(np.mean(self.connection_status[active_vehicle_mask, 3:5]))
                    if np.any(active_vehicle_mask)
                    else 0.0
                ),
                (
                    float(np.mean(self.connection_status[active_vehicle_mask, 0]))
                    if np.any(active_vehicle_mask)
                    else 0.0
                ),
                (
                    float(np.mean(self.delivery_done[active_vehicle_mask]))
                    if np.any(active_vehicle_mask)
                    else 0.0
                ),
            ]
        )

        global_state = np.concatenate(
            [vehicle_state.reshape(-1), link_state.reshape(-1), network_state], axis=0
        )
        repeated_global_state = np.repeat(
            global_state.reshape(1, -1), self.max_vehicles, axis=0
        )
        agent_ids = np.eye(self.max_vehicles)

        self.states = np.concatenate(
            [repeated_global_state, agent_ids, active_mask], axis=1
        )
        self.state_dim = self.states.shape[1]

        self.masks[~self.active_vehicle_mask, :, :] = 1
        self.states[~self.active_vehicle_mask] = 0
