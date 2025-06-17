import numpy as np
import torch
from scipy.stats import truncnorm

from environ.utils import compute_data_rate, zipf
from environ.markov import MarkovTransitionModel


class Environment:
    """
    Main environment class for simulating vehicle mobility and content delivery.
    """

    def __init__(
        self,
        # Simulation Meta-parameters
        dt: float = 0.1,
        seed: int = 42,
        num_vehicles: int = 30,
        num_edges: int = 4,
        num_items: int = 100,
        # Road and Mobility
        road_length: float = 2000.0,
        road_width: float = 15.0,
        num_lanes: int = 4,
        vmin: float = 8.0,
        vmax: float = 12.0,
        # Content & Coding
        item_size_max: float = 500,
        item_size_min: float = 50,
        code_size: float = 32 * 1024 * 8,
        delivery_deadline_max: float = 300,
        delivery_deadline_min: float = 100,
        # Storage
        edge_capacity: float = 2 * 1024,
        vehicle_capacity: float = 0.5 * 1024,
        edge_cost: float = 1,
        vehicle_cost: float = 3,
        # Coverage (in meters)
        v2i_pc5_coverage: float = 500,
        v2i_wifi_coverage: float = 100,
        v2v_pc5_coverage: float = 100,
        # Bandwidth (bps)
        v2n_bandwidth_max: float = 100e6,
        v2n_bandwidth: float = 1 * 1e6,
        v2v_bandwidth_max: float = 100e6,
        v2v_bandwidth: float = 1e6,
        v2i_pc5_bandwidth_max: float = 20e6,
        v2i_pc5_bandwidth: float = 1e6,
        v2i_wifi_bandwidth_max: float = 80e6,
        v2i_wifi_bandwidth: float = 5e6,
        # Transmission Cost
        v2n_cost: float = 8,
        v2i_pc5_cost: float = 1,
        v2i_wifi_cost: float = 0.8,
        v2v_cost: float = 0.8,
        # Transmission Power (dBm)
        v2n_transmission_power: float = 35,
        v2i_pc5_transmission_power: float = 33,
        v2i_wifi_transmission_power: float = 25,
        v2v_transmission_power: float = 30,
        # Noise & Rates
        noise_power: float = -174,
        i2i_data_rate: float = 100e6,
        i2n_data_rate: float = 150e6,
        i2i_cost: float = 0.3,
        i2n_cost: float = 30,
        # Cost and Delay Scaling
        storage_cost_scale: float = 1e-2,
        delay_scale: float = 1e10,
        cost_scale: float = 1e2,
        delay_weight: float = 1,
        cost_weight: float = 0,
    ):
        # Set core meta-parameters
        self.dt = dt
        self.seed = seed
        self.num_vehicles = num_vehicles
        self.num_edges = num_edges
        self.num_items = num_items
        self.np_random = np.random.RandomState(seed)
        self.num_rats = 4

        # Road configuration
        self.road_length = road_length
        self.road_width = road_width
        self.num_lanes = num_lanes
        self.distance_between_lanes = road_width / num_lanes
        self.vmin = vmin
        self.vmax = vmax

        # Edge and BS positions
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
        self.bs_positions = (self.road_length / 2, self.road_width / 2)

        # Content Coding
        self.code_size = code_size
        self.delivery_deadline = self.np_random.randint(
            delivery_deadline_min, delivery_deadline_max, size=self.num_items
        )
        self.item_size = (
            self.np_random.randint(item_size_min, item_size_max, size=self.num_items)
            * 1024
            * 1024
            * 8
        )
        self.num_code_min = np.ceil(self.item_size / self.code_size).astype(int)

        # Content Popularity
        self.alphas = [
            MarkovTransitionModel(
                states=[0.8, 0.9, 0.1, 0.2], random_state=self.np_random
            )
            for _ in range(self.num_edges)
        ]
        self.ranks = []
        for edge in range(self.num_edges):
            arr = np.arange(self.num_items)
            self.np_random.shuffle(arr)
            self.ranks.append(arr)

        # Storage capacities and costs
        self.edge_capacity = edge_capacity
        self.edge_cost = edge_cost
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_cost = vehicle_cost

        # Coverage parameters
        self.v2i_pc5_coverage = v2i_pc5_coverage
        self.v2i_wifi_coverage = v2i_wifi_coverage
        self.v2v_pc5_coverage = v2v_pc5_coverage

        # Bandwidth parameters
        self.v2n_bandwidth_max = v2n_bandwidth_max
        self.v2n_bandwidth = v2n_bandwidth
        self.v2v_bandwidth_max = v2v_bandwidth_max
        self.v2v_bandwidth = v2v_bandwidth
        self.v2i_pc5_bandwidth_max = v2i_pc5_bandwidth_max
        self.v2i_pc5_bandwidth = v2i_pc5_bandwidth
        self.v2i_wifi_bandwidth_max = v2i_wifi_bandwidth_max
        self.v2i_wifi_bandwidth = v2i_wifi_bandwidth

        # Communication costs
        self.v2n_cost = v2n_cost
        self.v2i_pc5_cost = v2i_pc5_cost
        self.v2i_wifi_cost = v2i_wifi_cost
        self.v2v_cost = v2v_cost
        self.i2i_cost = i2i_cost
        self.i2n_cost = i2n_cost

        # Transmission power
        self.v2n_transmission_power = v2n_transmission_power
        self.v2i_pc5_transmission_power = v2i_pc5_transmission_power
        self.v2i_wifi_transmission_power = v2i_wifi_transmission_power
        self.v2v_transmission_power = v2v_transmission_power
        self.noise_power = noise_power

        # Data rate
        self.i2i_data_rate = i2i_data_rate
        self.i2n_data_rate = i2n_data_rate

        # Cost and delay scaling
        self.delay_scale = delay_scale
        self.cost_scale = cost_scale
        self.delay_weight = delay_weight
        self.cost_weight = cost_weight
        self.storage_cost_scale = storage_cost_scale

        self.cache = np.zeros(
            (self.num_edges + self.num_vehicles, self.num_items), dtype=int
        )

    # Initialization Methods
    def reset(self) -> None:
        """
        Reset the environment, including content library and mobility model.
        """
        self.steps = 0
        self.utility = []
        self.hit_ratio = []
        self.reset_mobility()
        self.reset_request()
        self.set_states()

    def random_large_step(self, cache_vehicles) -> None:
        """
        Perform a large step with random actions for all edges and vehicles.
        """
        self.cache[: self.num_edges, :] = self.np_random.randint(
            0, 2, size=(self.num_edges, self.num_items)
        )
        self.cache[np.array(cache_vehicles) + self.num_edges, :] = (
            self.np_random.randint(0, 2, size=(len(cache_vehicles), self.num_items))
        )
        self.set_states()

    def all_large_step(self, cache_vehicles) -> None:
        """
        Perform a large step with all actions set to 1 for all edges and vehicles.
        """
        self.cache[: self.num_edges, :] = 1
        self.cache[np.array(cache_vehicles) + self.num_edges, :] = 1
        self.set_states()

    def large_step(self, actions, caching_vehicles_indices) -> None:
        # Update the cache with the provided actions
        self.cache[: self.num_edges, :] = actions.cpu()

        # Update the cache vehicles according to the corresponding edge indices
        self.update_vehicle_cache(caching_vehicles_indices)

        self.set_states()

    def update_vehicle_cache(self, caching_vehicles_indices):
        for vehicle_index in caching_vehicles_indices:
            edge_idx = self.local_of[vehicle_index]
            temp_capacity = self.vehicle_capacity * 1024 * 1024 * 8  # Convert to bits
            for item_idx in np.argsort(self.popularities[edge_idx])[::-1]:
                if (
                    self.cache[edge_idx, item_idx] == 1
                    and temp_capacity >= self.item_size[item_idx]
                ):
                    temp_capacity -= self.item_size[item_idx]
                    self.cache[vehicle_index + self.num_edges, item_idx] = 1
                if temp_capacity <= 0:
                    break

    def reset_request(self) -> None:
        """
        Reset and assign the requests matrix for all vehicles.
        """
        # Step 0: zero out vehicle cache
        self.cache[self.num_edges :, :] = 0

        # Step 1: Refresh the alpha values for Zipf distribution for each edge
        for alpha in self.alphas:
            alpha.value = alpha.step()

        # Step 2: Generate Popularity for each edge
        self.popularities = []
        for edge in range(self.num_edges):
            popularity = np.zeros(self.num_items)
            popularity[self.ranks[edge]] = zipf(self.num_items, self.alphas[edge].value)
            self.popularities.append(popularity)

        # Step 6: Generate request matrix (one request per vehicle)
        self.requests_matrix = np.zeros((self.num_vehicles, self.num_items))
        self.requests_edges = np.zeros((self.num_edges, self.num_items))
        for i in range(self.num_vehicles):
            requested_item = self.np_random.choice(
                self.num_items, size=1, p=self.popularities[self.local_of[i]]
            )
            self.requests_matrix[i, requested_item] = 1
            self.requests_edges[self.local_of[i], requested_item] = 1
        self.requested = np.argmax(self.requests_matrix, axis=1)

        # Step 7: Reset delivery status tracking
        self.delivery_done = np.zeros(self.num_vehicles)
        self.collected = np.zeros(self.num_vehicles)
        self.delay = np.zeros(self.num_vehicles)
        self.cost = np.zeros(self.num_vehicles)

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

        # Generate x coordinates based on the direction of the lanes
        # x = np.where(lane_indices < self.num_lanes / 2, self.road_length, 0)

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

    def greedy_projection(self, actions) -> None:  # num_vehicle x num_rats
        # bring action to cpu
        device = actions.device
        if device.type == "cuda":
            actions = actions.cpu()

        # check the shape of the actions
        joined = False
        if len(actions.shape) == 2:  # current shape is num_vehicles x num_rats
            pass
        elif (
            len(actions.shape) == 1
        ):  # current shape is num_vehicles * num_rats (joined)
            actions = actions.view(self.num_vehicles, self.num_rats)
            joined = True

        # compute the v2v action overload
        v2v_overload = max(
            0, torch.sum(actions[:, 1]) - self.v2v_bandwidth_max / self.v2v_bandwidth
        )

        # drop v2v action of low priority vehicles
        if v2v_overload > 0:
            v2v_index = torch.where(actions[:, 1] == 1)[0]
            priorities = torch.argsort(
                torch.tensor(
                    self.remaining_deadline[v2v_index]
                    / self.remaining_segments[v2v_index]
                ),
                dim=0,
            ).squeeze()

            while v2v_overload > 0 and priorities.numel() > 0:
                actions[priorities[0], 1] = 0
                v2v_overload -= 1
                priorities = priorities[1:]  # delete the first element

        # for each edge, check if the v2i pc5 and wifi actions are overloaded
        for edge_index in range(self.num_edges):
            # drop v2i pc5 action of low priority vehicles
            pc5_index = torch.where(
                (actions[:, 2] == 1) & (self.local_of == edge_index)
            )[0]
            v2i_pc5_overload = (
                torch.clamp(
                    torch.sum(actions[pc5_index, 2])
                    - self.v2i_pc5_bandwidth_max / self.v2i_pc5_bandwidth,
                    min=0,
                )
                / self.num_edges
            )
            if v2i_pc5_overload > 0:
                priorities = torch.argsort(
                    torch.tensor(
                        self.remaining_deadline[pc5_index]
                        / self.remaining_segments[pc5_index]
                    ),
                    dim=0,
                ).squeeze()

                while v2i_pc5_overload > 0 and priorities.numel() > 0:
                    actions[pc5_index[priorities[0]], 2] = 0
                    v2i_pc5_overload -= 1
                    priorities = priorities[1:]

            # drop v2i wifi action of low priority vehicles
            wifi_index = torch.where(
                (actions[:, 3] == 1) & (self.local_of == edge_index)
            )[0]
            v2i_wifi_overload = (
                torch.clamp(
                    torch.sum(actions[wifi_index, 3])
                    - self.v2i_wifi_bandwidth_max / self.v2i_wifi_bandwidth,
                    min=0,
                )
                / self.num_edges
            )
            if v2i_wifi_overload > 0:
                priorities = torch.argsort(
                    torch.tensor(
                        self.remaining_deadline[wifi_index]
                        / self.remaining_segments[wifi_index]
                    ),
                    dim=0,
                ).squeeze()
                while v2i_wifi_overload > 0 and priorities.numel() > 0:
                    actions[wifi_index[priorities[0]], 3] = 0
                    v2i_wifi_overload -= 1
                    priorities = priorities[1:]

        # bring action back to the original shape
        if joined:
            actions = actions.view(-1)

        return actions.to(device)

    def compute_violation(self):
        deadline_cost = np.where(
            (self.delay - self.delivery_deadline[self.requested]) > 0, 1, 0
        ).reshape(-1, 1)
        deadline_cost = deadline_cost * (1 - self.delivery_done.reshape(-1, 1))
        return deadline_cost

    def channel_utility_rate(self, actions: np.ndarray) -> np.ndarray:
        """
        For each type of link, count the number of vehicles that are using it to see which link vehicle prefers
        """

        # convert to numpy array if actions is a tensor
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return np.sum(actions, axis=0) / self.num_vehicles

    # State Management
    def set_states(self) -> None:
        """
        Set the states of the environment.
        """

        in_bound_mask = self.out == 0
        requested_items = self.requested[in_bound_mask]
        local_edges = self.local_of[in_bound_mask]

        # Initialize the delivery mask (no masks available - ==0)
        self.masks = np.zeros((self.num_vehicles, self.num_rats, 2))

        # if vehicle is out of the road, force disable v2i (mask 1)
        self.masks[self.out == 1, 2, 1] = 1
        self.masks[self.out == 1, 3, 1] = 1

        # if any nearby vehicle has the requested item and in communication range
        for vehicle_index in range(self.num_vehicles):
            any_car = False
            for nearby_vehicle_index in range(self.num_vehicles):
                if (
                    vehicle_index != nearby_vehicle_index  # ignore self
                    and self.vehicle_distance[vehicle_index, nearby_vehicle_index]
                    < self.v2v_pc5_coverage  # check if in communication range
                    and self.cache[
                        self.num_edges + nearby_vehicle_index,
                        self.requested[vehicle_index],
                    ]
                    == 1  # check if the nearby vehicle has the requested item
                ):
                    # if v2v is available, break the loop
                    any_car = True
                    break

            if not any_car:
                # if v2v is not available, force disable v2v (mask 1)
                self.masks[vehicle_index, 1, 1] = 1

        # if delivery is done, force disable all actions (mask 1)
        self.masks[self.delivery_done == 1, :, 1] = 1

        self.connection_status = np.zeros((self.num_vehicles, 6))
        """ 
        0: cache available in nearby vehicle
        1: cache available in local edge
        2: cache available in neighbor edge
        3: number of vehicle in local pc5 coverage
        4: number of vehicle in local wifi coverage
        5: cache available in the vehicle
        """

        # cache in nearby vehicle
        self.connection_status[:, 0] = 1 - self.masks[:, 1, 1]

        # cache in the vehicle
        for vehicle_index in range(self.num_vehicles):
            if (
                self.cache[
                    self.num_edges + vehicle_index, self.requested[vehicle_index]
                ]
                == 1
            ):
                self.connection_status[vehicle_index, 5] = 1

        # cache in local edge
        self.connection_status[in_bound_mask, 1] = (
            self.cache[local_edges, requested_items] == 1
        ).astype(int)

        # cache in neighbor edge
        self.connection_status[in_bound_mask, 2] = np.array(
            [
                np.any(
                    [
                        self.cache[edge_idx, item] == 1 and edge_idx != local_edge
                        for edge_idx in range(self.num_edges)
                    ]
                )
                for item, local_edge in zip(requested_items, local_edges)
            ]
        ).astype(int)

        # count number of vehicles in coverage of each edge
        counts = np.zeros((self.num_edges, 2))
        for vehicle_index in range(self.num_vehicles):
            # ignore if vehicle is out of the road or the delivery is done
            if (
                not in_bound_mask[vehicle_index]
                or self.delivery_done[vehicle_index] == 1
            ):
                continue

            edge_index = int(self.local_of[vehicle_index])

            counts[edge_index, 0] += 1
            if self.local_edge_distance[vehicle_index] < self.v2i_wifi_coverage:
                counts[edge_index, 1] += 1

            # if the vehicle is not in wifi coverage, force disable wifi (mask 1)
            else:
                self.masks[vehicle_index, 3, 1] = 1

        max_pc5 = self.v2i_wifi_bandwidth_max / self.v2i_pc5_bandwidth
        max_wifi = self.v2i_wifi_bandwidth_max / self.v2i_wifi_bandwidth

        # normalize the number of vehicles in coverage of each edge
        counts[:, 0] /= max_pc5
        counts[:, 1] /= max_wifi

        # assign the number of vehicles in coverage of each edge to the connection status
        for edge_index in range(self.num_edges):
            self.connection_status[(self.local_of == edge_index) & in_bound_mask, 3] = (
                counts[edge_index, 0]
            )
            self.connection_status[(self.local_of == edge_index) & in_bound_mask, 4] = (
                counts[edge_index, 1]
            )

        # remaining deadline of uncompleted tasks (0 if task is done) | shape: (num_vehicles,)
        self.remaining_deadline = np.where(
            self.delivery_done == 1,
            0,
            self.delivery_deadline[self.requested] - self.steps,
        ).reshape(-1, 1)

        # remaining segments of uncompleted tasks (0 if task is done) | shape: (num_vehicles,)
        self.remaining_segments = (
            (self.num_code_min[self.requested] - self.collected)
            .clip(min=1e-8)
            .reshape(-1, 1)
        )

        # how long the delivery is late
        self.current_late = np.where(
            self.delivery_done == 1,
            0,
            np.clip(self.delay - self.delivery_deadline[self.requested], 0, None),
        ).reshape(-1, 1)

        # if the delivery is not done and late, force enable v2n
        self.masks[
            (self.delivery_done == 0) & (self.current_late.reshape(-1) > 0), 0, 0
        ] = 1

        self.states = np.concatenate(
            [
                self.connection_status,
                self.remaining_deadline,
                self.remaining_segments,
                self.current_late,
            ],
            axis=1,
        )

        self.state_dim = self.states.shape[1]

    # Mobility Updates
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

        # Mask out-of-bounds vehicles
        self.local_edge_distance[self.out == 1] = 1e9

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

    def is_small_done(self) -> bool:
        """
        Check if all vehicles have completed their deliveries.
        Returns:
            bool: True if all deliveries are done, False otherwise.
        """
        return np.all(self.delivery_done == 1)

    def step_position(self) -> None:
        """
        Update the positions of vehicles based on their velocities and direction.
        """
        self.positions[:, 0] += self.velocities * self.dt * self.direction
        self.update_mobility_status()

    # Simulation Steps
    def small_step(self, actions: np.ndarray) -> None:
        """
        Perform a small time step by updating vehicle velocities, positions, and managing content delivery.
        Args:
            actions (np.ndarray): Delivery actions taken by the system.
        """
        # increment the step counter
        self.steps += 1

        # actions = np.ones((self.num_vehicles, self.num_rats))

        # initialize the cost and delay for the current step
        new_cost = np.zeros(self.num_vehicles)
        new_delay = np.zeros(self.num_vehicles)
        new_collected = np.zeros(self.num_vehicles)
        use_v2i = np.zeros(self.num_vehicles)
        hit_v2i = np.zeros(self.num_vehicles)
        self.old_done = self.delivery_done.sum()

        # convert torch tensor to numpy array
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        for vehicle_index in range(self.num_vehicles):
            # get the requested item index
            requested_item = np.where(self.requests_matrix[vehicle_index] == 1)[0]

            # ignore if request has been satisfied
            if self.delivery_done[vehicle_index] == 1:
                continue

            new_delay[vehicle_index] = self.dt

            # if requested item is inside the vehicle cache, skip the download
            if self.cache[self.num_edges + vehicle_index, requested_item] == 1:
                new_collected[vehicle_index] += self.num_code_min[requested_item] + 1
                new_cost[vehicle_index] = 0
                self.delivery_done[vehicle_index] = 1
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
                v2n_transfered_segment = np.floor(data_rate * self.dt / self.code_size)

                # accumulate the collected segments
                new_collected[vehicle_index] += v2n_transfered_segment

                # accumulate the cost
                new_cost[vehicle_index] = (
                    self.v2n_cost * v2n_transfered_segment * self.code_size
                )

            # download with v2v
            if actions[vehicle_index, 1] == 1:
                nearby_vehicles = []

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
                    and self.cache[
                        self.num_edges + nearby_vehicle_index, requested_item
                    ]
                    == 1
                ]

                # search nearby vehicles that have the requested item

                if len(nearby_vehicles) > 0:
                    min_distance = self.v2v_pc5_coverage

                    for nearby_vehicle_index, distance in nearby_vehicles:
                        if distance < min_distance:
                            min_distance = distance

                    # compute the v2v data rate with micro path loss model
                    data_rate = compute_data_rate(
                        allocated_spectrum=self.v2v_bandwidth,
                        transmission_power=self.v2v_transmission_power,
                        noise_power=self.noise_power,
                        distance=min_distance,
                        path_loss_model="micro",
                    )

                    # compute the number of segments that can be transfered
                    v2v_transfered_segment = np.floor(
                        data_rate * self.dt / self.code_size
                    )

                    # accumulate the collected segments
                    new_collected[vehicle_index] += v2v_transfered_segment

                    # accumulate the cost
                    new_cost[vehicle_index] = (
                        self.v2v_cost * v2v_transfered_segment * self.code_size
                    )

            # download with v2i pc5 and vehicle is not out of the road
            if actions[vehicle_index, 2] == 1 and self.out[vehicle_index] == 0:
                # count v2i
                use_v2i[vehicle_index] = 1

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
                    v2i_pc5_transfered_segment = np.floor(
                        data_rate * self.dt / self.code_size
                    )

                    # accumulate the collected segments
                    new_cost[vehicle_index] = (
                        self.v2i_pc5_cost * v2i_pc5_transfered_segment * self.code_size
                    )
                    hit_v2i[vehicle_index] = 1
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
                        v2i_pc5_transfered_segment = np.floor(
                            self.dt
                            * self.i2i_data_rate
                            * data_rate
                            / (
                                self.code_size
                                * (data_rate * hop_distance + self.i2i_data_rate)
                            )
                        )
                        # accumulate the cost
                        new_cost[vehicle_index] = (
                            v2i_pc5_transfered_segment
                            * self.code_size
                            * (self.i2i_cost + self.v2i_pc5_cost)
                        )
                        hit_v2i[vehicle_index] = 1

                    # if there is no neighbor edge that has the requested item, use backhaul link
                    else:
                        v2i_pc5_transfered_segment = np.floor(
                            self.dt
                            * self.i2n_data_rate
                            * data_rate
                            / (self.code_size * (data_rate + self.i2n_data_rate))
                        )
                        # accumulate the cost: i2n + v2i_pc5
                        new_cost[vehicle_index] = (
                            v2i_pc5_transfered_segment
                            * self.code_size
                            * (self.i2n_cost + self.v2i_pc5_cost)
                        )

                # accumulate the collected segments
                new_collected[vehicle_index] += v2i_pc5_transfered_segment

            # download with v2i wifi and vehicle is not out of the road
            if actions[vehicle_index, 3] == 1 and self.out[vehicle_index] == 0:
                # count v2i
                use_v2i[vehicle_index] = 1

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
                        v2i_wifi_transfered_segment = np.floor(
                            data_rate * self.dt / self.code_size
                        )

                        # accumulate the collected segments
                        new_cost[vehicle_index] = (
                            self.v2i_wifi_cost
                            * v2i_wifi_transfered_segment
                            * self.code_size
                        )
                        hit_v2i[vehicle_index] = 1
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
                            v2i_wifi_transfered_segment = np.floor(
                                self.dt
                                * self.i2i_data_rate
                                * data_rate
                                / (
                                    self.code_size
                                    * (data_rate + self.i2i_data_rate * hop_distance)
                                )
                            )
                            # accumulate the cost
                            new_cost[vehicle_index] = (
                                v2i_wifi_transfered_segment
                                * self.code_size
                                * (self.i2i_cost + self.v2i_wifi_cost)
                            )
                            hit_v2i[vehicle_index] = 1

                        # if there is no neighbor edge that has the requested item, use backhaul link
                        else:
                            v2i_wifi_transfered_segment = np.floor(
                                self.dt
                                * self.i2n_data_rate
                                * data_rate
                                / (self.code_size * (data_rate + self.i2n_data_rate))
                            )
                            # accumulate the cost
                            new_cost[vehicle_index] = (
                                self.i2n_cost
                                * v2i_wifi_transfered_segment
                                * self.code_size
                                + self.v2i_wifi_cost
                                * v2i_wifi_transfered_segment
                                * self.code_size
                            )

                    # accumulate the collected segments
                    new_collected[vehicle_index] += v2i_wifi_transfered_segment

            # check if the vehicle has collected all the requested items
            if self.collected[vehicle_index] > self.num_code_min[requested_item]:
                self.delivery_done[vehicle_index] = 1

        # accumulate the cost and delay
        self.cost += new_cost
        self.delay += new_delay
        self.collected += new_collected

        # compute cost and delay terms
        delay_term = (
            -self.delay_weight
            * self.delay_scale
            * new_delay
            / self.item_size[self.requested]
        )  # delay per bit
        cost_term = (
            -self.cost_weight
            * self.cost_scale
            * new_cost
            / self.item_size[self.requested]
        )  # cost per bit

        # compute the reward, dones, and violations
        self.rewards = (cost_term + delay_term).reshape(-1, 1)

        self.dones = self.delivery_done.astype(float).reshape(-1, 1)
        self.violations = self.compute_violation()
        self.utility.append(actions)
        hit_ratio = sum(hit_v2i) / sum(use_v2i) if sum(use_v2i) > 0 else -1
        self.hit_ratio.append(hit_ratio)

        # update env
        self.step_velocity()
        self.step_position()
        self.set_states()

        # return the next states, rewards, dones, and violations
        return (
            self.states,
            self.rewards,
            self.dones,
            self.violations,
        )
