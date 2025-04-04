from typing import Any, List, Literal, Union

import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm

from visualize import render


class MarkovTransitionModel:
    def __init__(self, states: List[Any]):
        self.states = states
        self.num_states = len(states)
        self.transition_matrix = np.full(
            (self.num_states, self.num_states), 1 / self.num_states
        )
        self.value = np.random.choice(self.states)

    def step(self) -> Any:
        current = self.states.index(self.value)
        next_state = np.random.choice(self.states, p=self.transition_matrix[current])
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
        path_loss = 128.1 + 37.6 * np.log10(distance * 1e-3)
    elif path_loss_model == "micro":
        path_loss = 140.7 + 36.7 * np.log10(distance * 1e-3)
    else:
        raise ValueError("Invalid path loss model")
    received_power = transmission_power - path_loss

    noise_power_linear = 10 ** ((noise_power - 30) / 10)
    received_power_linear = 10 ** ((received_power - 30) / 10)

    # Calculate the data rate using Shannon-Hartley theorem
    data_rate = allocated_spectrum * np.log2(
        1 + received_power_linear / noise_power_linear
    )
    return data_rate


class Environment:
    def __init__(
        self,
        dt: float = 0.1,
        num_vehicles: int = 30,
        num_edges: int = 4,
        num_items: int = 200,
        road_length: float = 2000.0,
        road_width: float = 15.0,
        num_lanes: int = 4,
        vmin: float = 8.0,
        vmax: float = 12.0,
        item_size_max: float = 500,
        item_size_min: float = 50,
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
    ):
        # Main parameters
        self.dt = dt
        self.num_vehicles = num_vehicles
        self.num_edges = num_edges
        self.num_items = num_items

        # Content model
        self.alpha = MarkovTransitionModel(states=[0.1, 0.8, 1.0, 1.2])
        self.requests_frequency = np.random.randint(
            1, 1000, size=self.num_items
        )  # Dummy requests frequency

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

    def reset(self) -> None:
        """
        Reset the environment, including content library and mobility model.
        """
        self.reset_request()
        self.reset_mobility()

    def reset_mobility(self, sigma: float = 0.5) -> None:
        """
        Reset and assign positions and velocities of vehicles.
        Args:
            sigma (float): Standard deviation for velocity distribution.
        """
        # Generate random x-coordinates for vehicles within the road length
        x = np.random.uniform(0, self.road_length, self.num_vehicles)

        # Calculate y-coordinates based on lane indices and lane spacing
        lane_indices = np.random.randint(0, self.num_lanes, size=self.num_vehicles)
        self.direction = np.where(lane_indices < self.num_lanes / 2, -1, 1)
        y = lane_indices * self.distance_between_lanes + self.distance_between_lanes / 2

        # Assign positions
        self.positions = np.column_stack((x, y))

        # Compute the mean velocity
        mu = (self.vmin + self.vmax) / 2

        # Generate random velocities using truncated Gaussian distribution
        a, b = (self.vmin - mu) / sigma, (self.vmax - mu) / sigma
        self.velocities = truncnorm.rvs(
            a, b, loc=mu, scale=sigma, size=self.num_vehicles
        )

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
                i, np.random.choice(self.num_items, size=1, p=self.requests_probs)
            ] = 1

        # Update the requests frequency
        self.requests_frequency = np.sum(self.requests_matrix, axis=0)

    def step(self, actions: np.ndarray) -> None:
        """
        Perform a simulation step by updating vehicle velocities and positions.
        Args:
            actions (np.ndarray): Actions taken by the system.
        """

        # create a dummy action
        delivery_action = np.zeros((self.num_vehicles, 4))

        # manage resource allocation

        self.step_velocity()
        self.step_position()

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
            vt_next[i] = truncnorm.rvs(a, b, loc=mu, scale=sigma)

        self.velocities = vt_next

    def step_position(self) -> None:
        """
        Update the positions of vehicles based on their velocities and direction.
        """
        self.positions[:, 0] += self.velocities * self.dt * self.direction


if __name__ == "__main__":
    env = Environment(num_vehicles=40, num_edges=4, num_items=100)
    env.reset()

    render(env, 1000, speed=1)
