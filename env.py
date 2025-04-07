from typing import Any, List, Literal, Union

import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm

from render import render


class MarkovTransitionModel:
    def __init__(self, states: List[Any], random_state=None):
        self.states = states
        self.num_states = len(states)
        self.transition_matrix = np.full(
            (self.num_states, self.num_states), 1 / self.num_states
        )
        self.random = random_state or np.random  # Use provided random state or default
        self.value = self.random.choice(self.states)

    def step(self) -> Any:
        current = self.states.index(self.value)
        next_state = self.random.choice(self.states, p=self.transition_matrix[current])
        self.value = next_state
        return next_state


def zipf(num_items, alpha) -> np.ndarray:
    """
    Generate a Zipf distribution for the given number of items and alpha parameter.
    Args:
        num_items (int): Number of items.
        alpha (float): Zipf distribution parameter.
    Returns:
        np.ndarray: Zipf distribution probabilities.
    """
    z = np.arange(1, num_items + 1)
    zipf_dist = 1 / (z**alpha)
    zipf_dist /= np.sum(zipf_dist)
    return zipf_dist


def compute_data_rate(
    allocated_spectrum: float,
    transmission_power: float,
    noise_power: float,
    distance: Union[float, np.ndarray],
    path_loss_model: Literal["macro", "micro"] = "macro",
) -> Union[float, np.ndarray]:
    """
    Compute the data rate based on the Shannon-Hartley theorem.
    Args:
        allocated_spectrum (float): Allocated spectrum in Hz.
        transmission_power (float): Transmission power in dBm.
        noise_power (float): Noise power in dBm.
        distance (Union[float, np.ndarray]): Distance in meters.
        path_loss_model (str): Path loss model, either "macro" or "micro".
    Returns:
        float: Data rate in bps.
    """
    if path_loss_model == "macro":
        path_loss = 128.1 + 37.6 * np.log10(max(distance * 1e-3, 1e-6))  # Avoid log(0)
    elif path_loss_model == "micro":
        path_loss = 140.7 + 36.7 * np.log10(max(distance * 1e-3, 1e-6))
    else:
        raise ValueError("Invalid path loss model")

    received_power = transmission_power - path_loss
    noise_power_linear = 10 ** ((noise_power - 30) / 10)
    received_power_linear = 10 ** ((received_power - 30) / 10)

    # Calculate the data rate using Shannon-Hartley theorem
    snr = max(
        received_power_linear / noise_power_linear, 1e-9
    )  # Avoid division by zero
    data_rate = allocated_spectrum * np.log2(1 + snr)
    return data_rate


class Environment:
    def __init__(
        self,
        dt: float = 0.1,
        num_vehicles: int = 30,
        num_edges: int = 4,
        num_items: int = 200,
        seed: int = 42,
        road_length: float = 2000.0,
        road_width: float = 15.0,
        num_lanes: int = 4,
        vmin: float = 8.0,
        vmax: float = 12.0,
        item_size_max: float = 500,
        item_size_min: float = 50,
        code_size: float = 512 * 1024 * 8,
        delivery_deadline_max: float = 300,
        delivery_deadline_min: float = 50,
        edge_capacity: float = 2 * 1024,
        edge_cost: float = 1,
        vehicle_capacity: float = 0.5 * 1024,
        vehicle_cost: float = 3,
        v2i_pc5_coverage: float = 500,
        v2i_wifi_coverage: float = 100,
        v2v_pc5_coverage: float = 100,
        v2n_bandwidth_max: float = 100e6,
        v2n_bandwidth: float = 0.5e6,
        v2v_bandwidth_max: float = 20e6,
        v2v_bandwidth: float = 1e6,
        v2i_pc5_bandwidth_max: float = 20e6,
        v2i_pc5_bandwidth: float = 1e6,
        v2i_wifi_bandwidth_max: float = 80e6,
        v2i_wifi_bandwidth: float = 5e6,
        v2n_cost: float = 10,
        v2i_pc5_cost: float = 1,
        v2i_wifi_cost: float = 0.8,
        v2v_cost: float = 0.8,
        v2n_transmission_power: float = 35,
        v2i_pc5_transmission_power: float = 33,
        v2i_wifi_transmission_power: float = 25,
        v2v_transmission_power: float = 30,
        noise_power: float = -90,
        i2i_data_rate: float = 150e6,
        i2n_data_rate: float = 100e6,
        i2i_cost: float = 0.3,
        i2n_cost: float = 8,
    ):
        # Main parameters
        self.dt = dt
        self.num_vehicles = num_vehicles
        self.num_edges = num_edges
        self.num_items = num_items
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.num_rats = 4

        # Content model
        self.alpha = MarkovTransitionModel(
            states=[0.1, 0.8, 1.0, 1.2], random_state=self.np_random
        )
        self.requests_frequency = self.np_random.randint(
            1, 1000, size=self.num_items
        )  # Dummy requests frequency

        self.item_size = (
            self.np_random.randint(item_size_min, item_size_max, size=self.num_items)
            * 1024
            * 1024
            * 8
        )
        self.delivery_deadline = self.np_random.uniform(
            delivery_deadline_min, delivery_deadline_max, size=self.num_items
        )
        self.code_size = code_size
        self.num_code_min = np.ceil(self.item_size / self.code_size).astype(int)

        # Mobility model
        self.road_length = road_length
        self.road_width = road_width
        self.num_lanes = num_lanes
        self.distance_between_lanes = road_width / num_lanes
        self.vmin = vmin
        self.vmax = vmax

        # Edge parameters
        self.edge_positions = np.stack(
            [
                np.ones(num_edges)
                * [
                    self.road_length / self.num_edges / 2
                    + i * self.road_length / self.num_edges
                    for i in range(self.num_edges)
                ],
                np.ones(self.num_edges) * self.road_width / 2,
            ],
            axis=1,
        )

        # BS parameters
        self.bs_positions = (self.road_length / 2, 2000)

        #
        self.edge_capacity = edge_capacity
        self.edge_cost = edge_cost
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_cost = vehicle_cost
        self.v2i_pc5_coverage = v2i_pc5_coverage
        self.v2i_wifi_coverage = v2i_wifi_coverage
        self.v2v_pc5_coverage = v2v_pc5_coverage
        self.v2n_bandwidth_max = v2n_bandwidth_max
        self.v2n_bandwidth = v2n_bandwidth
        self.v2v_bandwidth_max = v2v_bandwidth_max
        self.v2v_bandwidth = v2v_bandwidth
        self.v2i_pc5_bandwidth_max = v2i_pc5_bandwidth_max
        self.v2i_pc5_bandwidth = v2i_pc5_bandwidth
        self.v2i_wifi_bandwidth_max = v2i_wifi_bandwidth_max
        self.v2i_wifi_bandwidth = v2i_wifi_bandwidth
        self.v2n_cost = v2n_cost
        self.v2i_pc5_cost = v2i_pc5_cost
        self.v2i_wifi_cost = v2i_wifi_cost
        self.v2v_cost = v2v_cost
        self.v2n_transmission_power = v2n_transmission_power
        self.v2i_pc5_transmission_power = v2i_pc5_transmission_power
        self.v2i_wifi_transmission_power = v2i_wifi_transmission_power
        self.v2v_transmission_power = v2v_transmission_power
        self.noise_power = noise_power
        self.i2i_data_rate = i2i_data_rate
        self.i2n_data_rate = i2n_data_rate
        self.i2i_cost = i2i_cost
        self.i2n_cost = i2n_cost

    def reset(self) -> None:
        """
        Reset the environment, including content library and mobility model.
        """
        self.steps = 0
        self.done = False
        self.reset_request()
        self.reset_mobility()
        self.set_states()
        self.update_mask()

    def set_states(self) -> None:
        """
        Set the states of the environment.
        """
        self.state = np.concatenate(
            [
                self.positions.flatten(),  # self.num_vehicles * 2
                self.requests_matrix.flatten(),  # self.num_vehicles * self.num_items
                self.cache.flatten(),  # (self.self.num_edges + self.num_vehicles) * self.num_items
                self.item_size.flatten(),  # self.num_items
                (
                    self.delivery_deadline - self.steps
                ).flatten(),  # self.num_items (the remaining deadline)
                (
                    self.num_code_min[np.argmax(self.requests_matrix, axis=1)]
                    - self.collected
                ).flatten(),  # self.num_vehicles
            ]
        )
        self.state_dim = (
            self.num_vehicles * 2
            + self.num_vehicles * self.num_items
            + (self.num_edges + self.num_vehicles) * self.num_items
            + self.num_items
            + self.num_items
            + self.num_vehicles
        )

    def update_mask(self) -> None:
        """
        Update the action mask based on the current state of the environment.
        """
        # Delivery action mask
        # Initialize the delivery mask (mask all actions as unavailable)
        self.delivery_mask = np.zeros(
            (self.num_vehicles, self.num_rats)
        )  # 0: no mask, 1: force enable, -1: force disable

        # get the requested item index
        requested_item = np.where(self.requests_matrix == 1)[1]

        # if vehicle request downloading is not satisfied within the deadline, force enable v2n
        self.delivery_mask[
            (self.delivery_deadline[requested_item] - self.steps) < 0, 0
        ] = 1

        # if vehicle is out of the road, force disable v2i
        self.delivery_mask[self.out == 1, 2] = -1
        self.delivery_mask[self.out == 1, 3] = -1

        # if vehicle not in wifi coverage, force disable v2i wifi
        self.delivery_mask[self.local_edge_distance > self.v2i_wifi_coverage, 3] = -1

        # if any nearby vehicle has the requested item and in communication range
        for vehicle_index in range(self.num_vehicles):
            for nearby_vehicle_index in range(self.num_vehicles):
                if (
                    vehicle_index != nearby_vehicle_index  # ignore self
                    and self.vehicle_distance[vehicle_index, nearby_vehicle_index]
                    < self.v2v_pc5_coverage  # check if in communication range
                    and self.cache[
                        self.num_edges + nearby_vehicle_index,
                        requested_item[vehicle_index],
                    ]
                    == 1  # check if the nearby vehicle has the requested item
                ):
                    # if v2v is available, break the loop
                    break

            # if v2v is not available, force disable v2v
            self.delivery_mask[vehicle_index, 1] = -1

    def reset_mobility(self, sigma: float = 0.5) -> None:
        """
        Reset and assign positions and velocities of vehicles.
        Args:
            sigma (float): Standard deviation for velocity distribution.
        """
        # Generate random x-coordinates for vehicles within the road length
        x = self.np_random.uniform(0, self.road_length, self.num_vehicles)

        # Calculate y-coordinates based on lane indices and lane spacing
        lane_indices = self.np_random.randint(0, self.num_lanes, size=self.num_vehicles)
        self.direction = np.where(lane_indices < self.num_lanes / 2, -1, 1)
        y = lane_indices * self.distance_between_lanes + self.distance_between_lanes / 2

        # Assign positions
        self.positions = np.column_stack((x, y))

        # Compute the mean velocity
        mu = (self.vmin + self.vmax) / 2

        # Generate random velocities using truncated Gaussian distribution
        a, b = (self.vmin - mu) / sigma, (self.vmax - mu) / sigma
        self.velocities = truncnorm.rvs(
            a,
            b,
            loc=mu,
            scale=sigma,
            size=self.num_vehicles,
            random_state=self.np_random,
        )

        # update the mobility status based on the initial positions
        self.update_mobility_status()

    def reset_request(self) -> None:
        """
        Reset and assign requests matrix.
        """
        # Update the Zipf alpha parameter
        self.alpha.step()

        # Sort content library by the requests frequency
        sorted_indices = np.argsort(self.requests_frequency)[::-1]

        # Generate requests probabilities from Zipf distribution
        self.requests_probs = zipf(self.num_items, self.alpha.value)[sorted_indices]

        # Sample the requests matrix
        self.requests_matrix = np.zeros((self.num_vehicles, self.num_items))
        for i in range(self.num_vehicles):
            self.requests_matrix[
                i, self.np_random.choice(self.num_items, size=1, p=self.requests_probs)
            ] = 1

        # Update the requests frequency
        self.requests_frequency = np.sum(self.requests_matrix, axis=0)
        self.cache = self.np_random.randint(
            0, 2, size=(self.num_edges + self.num_vehicles, self.num_items)
        )

        self.task_done = np.zeros(self.num_vehicles)
        self.collected = np.zeros(self.num_vehicles)
        self.delay = np.zeros(self.num_vehicles)
        self.cost = np.zeros(self.num_vehicles)

    def small_step(self, actions: np.ndarray) -> None:
        """
        Perform a small time step by updating vehicle velocities, positions, and managing content delivery.
        Args:
            actions (np.ndarray): Delivery actions taken by the system.
        """
        # increment the step counter
        self.steps += 1

        # create a dummy action
        actions = self.np_random.randint(0, 2, size=(self.num_vehicles, self.num_rats))

        for vehicle_index in range(self.num_vehicles):
            # get the requested item index
            requested_item = np.where(self.requests_matrix[vehicle_index] == 1)[0]

            # ignore if request has been satisfied
            if self.task_done[vehicle_index] == 1:
                continue

            # download with v2n
            if actions[vehicle_index, 0] == 1:
                # compute the distance from the vehicle to the BS
                distance = self.bs_distance[vehicle_index]

                # compute v2n data rate with macro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.v2n_bandwidth,
                    transmission_power=self.v2n_transmission_power,
                    noise_power=self.noise_power,
                    distance=distance,
                    path_loss_model="macro",
                )

                # compute the number of segments that can be transfered
                v2n_transfered_segment = np.ceil(data_rate * self.dt / self.code_size)

                # accumulate the collected segments
                self.collected[vehicle_index] += v2n_transfered_segment

                # accumulate the cost
                self.cost[vehicle_index] += (
                    self.v2n_cost * v2n_transfered_segment * self.code_size
                )

            # download with v2v
            if actions[vehicle_index, 1] == 1 and self.out[vehicle_index] == 0:
                nearby_vehicles = []
                potential_nearby_vehicles = []
                potential_distance = []

                # search all vehicles in communication range
                nearby_vehicles = [
                    (
                        nearby_vehicle_index,
                        self.vehicle_distance[vehicle_index, nearby_vehicle_index],
                    )
                    for nearby_vehicle_index in range(self.num_vehicles)
                    if vehicle_index != nearby_vehicle_index
                    and self.vehicle_distance[vehicle_index, nearby_vehicle_index]
                    < self.v2v_pc5_coverage
                ]

                # search nearby vehicles that have the requested item
                for nearby_vehicle_index, distance in nearby_vehicles:
                    # check if the nearby vehicle has the requested item
                    if (
                        self.cache[
                            self.num_edges + nearby_vehicle_index, requested_item
                        ]
                        == 1
                    ):
                        potential_nearby_vehicles.append(nearby_vehicle_index)
                        potential_distance.append(distance)

                # if there are potential nearby vehicles
                if len(potential_nearby_vehicles) > 0:
                    # find the closest vehicle
                    closest_vehicle_index = np.argmin(potential_distance)

                    # compute the distance to the closest vehicle
                    distance = potential_distance[closest_vehicle_index]

                    # compute the v2v data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.v2v_bandwidth,
                        transmission_power=self.v2v_transmission_power,
                        noise_power=self.noise_power,
                        distance=distance,
                        path_loss_model="micro",
                    )

                    # compute the number of segments that can be transfered
                    v2v_transfered_segment = np.ceil(
                        data_rate * self.dt / self.code_size
                    )

                    # accumulate the collected segments
                    self.collected[vehicle_index] += v2v_transfered_segment

                    # accumulate the cost
                    self.cost[vehicle_index] += (
                        self.v2n_cost * v2v_transfered_segment * self.code_size
                    )

            # download with v2i pc5 and vehicle is not out of the road
            if actions[vehicle_index, 2] == 1 and self.out[vehicle_index] == 0:
                # compute the distance from the vehicle to its local edge
                distance = self.local_edge_distance[vehicle_index]

                # compute v2i pc5 data rate with micro path loss model
                data_rate = compute_data_rate(
                    allocated_spectrum=self.v2i_pc5_bandwidth,
                    transmission_power=self.v2i_pc5_transmission_power,
                    noise_power=self.noise_power,
                    distance=distance,
                    path_loss_model="micro",
                )

                # check if the edge has the requested item
                if self.cache[int(self.local_of[vehicle_index]), requested_item] == 1:
                    # compute the number of segments that can be transfered directly from the local edge
                    v2i_pc5_transfered_segment = np.ceil(
                        data_rate * self.dt / self.code_size
                    )

                    # accumulate the collected segments
                    self.cost[vehicle_index] += (
                        self.v2i_pc5_cost * v2i_pc5_transfered_segment * self.code_size
                    )

                # if the edge does not have the requested item
                else:
                    # check for the nearest neighbor edge (by hop count) that has the requested item
                    hop_distance = 99
                    for edge_index in range(self.num_edges):
                        if self.cache[
                            edge_index, requested_item
                        ] == 1 and edge_index != int(self.local_of[vehicle_index]):
                            hop_distance = min(
                                hop_distance,
                                abs(edge_index - int(self.local_of[vehicle_index])),
                            )

                    # if there is a neighbor edge that has the requested item
                    if hop_distance < 99:
                        v2i_pc5_transfered_segment = np.ceil(
                            self.dt
                            * self.i2i_data_rate
                            / (
                                self.code_size
                                * (data_rate + self.i2i_data_rate * hop_distance)
                            )
                        )
                        # accumulate the cost
                        self.cost[vehicle_index] += (
                            self.i2i_cost * v2i_pc5_transfered_segment * self.code_size
                        )
                    # if there is no neighbor edge that has the requested item, use backhaul link
                    else:
                        v2i_pc5_transfered_segment = np.ceil(
                            self.dt
                            * self.i2n_data_rate
                            / (self.code_size * (data_rate + self.i2n_data_rate))
                        )
                        # accumulate the cost
                        self.cost[vehicle_index] += (
                            self.i2n_cost * v2i_pc5_transfered_segment * self.code_size
                        )

                # accumulate the collected segments
                self.collected[vehicle_index] += v2i_pc5_transfered_segment

            # download with v2i wifi and vehicle is not out of the road
            if actions[vehicle_index, 3] == 1 and self.out[vehicle_index] == 0:
                # check if the vehicle is within the coverage of the edge wifi
                distance = self.local_edge_distance[vehicle_index]

                if distance < self.v2i_wifi_coverage:
                    # compute v2i wifi data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.v2i_wifi_bandwidth,
                        transmission_power=self.v2i_wifi_transmission_power,
                        noise_power=self.noise_power,
                        distance=distance,
                        path_loss_model="micro",
                    )

                    # check if the edge has the requested item
                    if (
                        self.cache[int(self.local_of[vehicle_index]), requested_item]
                        == 1
                    ):
                        # compute the number of segments that can be transfered directly from the local edge
                        v2i_wifi_transfered_segment = np.ceil(
                            data_rate * self.dt / self.code_size
                        )

                        # accumulate the collected segments
                        self.cost[vehicle_index] += (
                            self.v2i_wifi_cost
                            * v2i_wifi_transfered_segment
                            * self.code_size
                        )

                    # if the edge does not have the requested item
                    else:
                        # check for the nearest neighbor edge (by hop count) that has the requested item
                        hop_distance = 99
                        for edge_index in range(self.num_edges):
                            if self.cache[
                                edge_index, requested_item
                            ] == 1 and edge_index != int(self.local_of[vehicle_index]):
                                hop_distance = min(
                                    hop_distance,
                                    abs(edge_index - int(self.local_of[vehicle_index])),
                                )

                        # if there is a neighbor edge that has the requested item
                        if hop_distance < 99:
                            v2i_wifi_transfered_segment = np.ceil(
                                self.dt
                                * self.i2i_data_rate
                                / (
                                    self.code_size
                                    * (data_rate + self.i2i_data_rate * hop_distance)
                                )
                            )
                            # accumulate the cost
                            self.cost[vehicle_index] += (
                                self.i2i_cost
                                * v2i_wifi_transfered_segment
                                * self.code_size
                            )
                        # if there is no neighbor edge that has the requested item, use backhaul link
                        else:
                            v2i_wifi_transfered_segment = np.ceil(
                                self.dt
                                * self.i2n_data_rate
                                / (self.code_size * (data_rate + self.i2n_data_rate))
                            )
                            # accumulate the cost
                            self.cost[vehicle_index] += (
                                self.i2n_cost
                                * v2i_wifi_transfered_segment
                                * self.code_size
                            )

            # check if the vehicle has collected all the requested items
            if self.collected[vehicle_index] > self.num_code_min[requested_item]:
                self.task_done[vehicle_index] = 1
                self.delay[vehicle_index] = self.steps * self.dt

        # check if all vehicles are done
        self.done = np.sum(self.task_done) == self.num_vehicles

        # update the mobility model
        self.step_velocity()
        self.step_position()
        self.set_states()
        self.update_mask()

        return None, None, self.done, None

    def large_step(self, actions: np.ndarray) -> None:
        """
        Perform a large time step by updating the cache status based on content caching actions.
        Args:
            actions (np.ndarray): Caching actions taken by the system.
        """
        # create a dummy action
        actions = self.np_random.randint(
            0, 2, size=(self.num_edges + self.num_vehicles, self.num_items)
        )

    def update_mobility_status(self) -> None:
        """
        Update mobility status based on the current positions of vehicles.
        """
        # Check if vehicles are out of the road
        self.out = np.where(
            (self.positions[:, 0] > self.road_length) | (self.positions[:, 0] < 0), 1, 0
        )

        # Extract x-coordinates
        vehicle_x = self.positions[:, 0]
        segment_length = self.road_length / self.num_edges

        # Compute edge indices
        edge_indices = (vehicle_x / segment_length).astype(int)
        edge_indices = np.clip(edge_indices, 0, self.num_edges - 1)  # ensure bounds

        # Initialize with -1 for all vehicles
        self.local_of = np.full(self.num_vehicles, -1, dtype=int)

        # Assign valid edge index only to in-bound vehicles
        in_bound_mask = self.out == 0
        self.local_of[in_bound_mask] = edge_indices[in_bound_mask]

        # Pre-compute local edge distances
        self.local_edge_distance = np.linalg.norm(
            self.positions - self.edge_positions[self.local_of.astype(int)], axis=1
        )

        # Pre-compute BS distances (broadcasted subtraction)
        self.bs_distance = np.linalg.norm(self.positions - self.bs_positions, axis=1)

        # Compute pairwise vehicle distances (efficient with broadcasting)
        diff = (
            self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        )  # shape: (N, N, 2)

        self.vehicle_distance = np.linalg.norm(diff, axis=2)

        # Set diagonal to zero (distance to self)
        np.fill_diagonal(self.vehicle_distance, 0.0)

    def step_velocity(self, sigma: float = 0.5) -> None:
        """
        Update the velocities of vehicles using a truncated Gaussian distribution.
        Args:
            sigma (float): Standard deviation for velocity distribution.
        """
        vt_next = np.zeros(self.num_vehicles)

        for i in range(self.num_vehicles):
            mu = self.velocities[i]
            a, b = (self.vmin - mu) / sigma, (self.vmax - mu) / sigma
            vt_next[i] = truncnorm.rvs(
                a, b, loc=mu, scale=sigma, random_state=self.np_random
            )

        self.velocities = vt_next

    def step_position(self) -> None:
        """
        Update the positions of vehicles based on their velocities and direction.
        """
        self.positions[:, 0] += self.velocities * self.dt * self.direction
        self.update_mobility_status()


if __name__ == "__main__":
    env = Environment(
        num_vehicles=300,
        num_edges=4,
        num_items=100,
        item_size_max=500,
        item_size_min=50,
        seed=123,
    )
    env.reset()

    for t in tqdm(range(3000)):
        actions = None  # You could have a policy here
        next_state, reward, done, _ = env.small_step(actions)
        if done:
            print(f"Stopped at timestep {t}")
            break

    print(
        "Avg Delay per segment (ms):",
        np.round(np.mean(env.delay * 1e3 / env.collected), 4),
    )
    print(
        "Avg Cost per bit (cost unit):",
        np.round(np.mean(env.cost / (env.collected * env.code_size)), 2),
    )
