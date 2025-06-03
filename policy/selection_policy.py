from typing import List

import numpy as np
from scipy.stats import truncnorm


def GTVS(
    env,
    min_vehicles: int = 3,
    DT: int = 10,
    T: int = 100,
) -> List[int]:
    """
    Greedy Vehicle Set Covering (GTVS) algorithm to select vehicles based on their coverage
    capabilities over a time horizon.
    Args:
        init_positions (np.ndarray): Initial positions of vehicles (shape: [num_vehicles, 2]).
        init_velocities (np.ndarray): Initial velocities of vehicles (shape: [num_vehicles,]).
        direction (np.ndarray): Direction vector for vehicle movement (shape: [2,]).
        vmin (float): Minimum velocity.
        vmax (float): Maximum velocity.
        dt (float): Time step size.
        coverage_radius (float): Radius within which a vehicle can cover a position.
        sigma (float): Standard deviation for the truncated normal distribution.
        DT (int): Downsampling factor for the time steps.
        T (int): Total number of time steps to simulate.
        min_vehicles (int): Minimum number of vehicles to select based on coverage.
        random_seed (int): Random seed for reproducibility.
    Returns:
        List[int]: Indices of selected vehicles that cover the positions over the time horizon.
    """
    init_positions = env.positions.copy()
    velocities = env.velocities.copy()
    direction = env.direction.copy()
    vmin = float(env.vmin)
    vmax = float(env.vmax)
    dt = float(env.dt)
    coverage_radius = float(env.v2v_pc5_coverage)
    sigma = 0.5
    seed = 42

    np_random = np.random.RandomState(seed)
    num_vehicles = init_positions.shape[0]
    positions = [init_positions.copy()]

    for t in range(1, T):
        vt_next = np.zeros(num_vehicles)

        for i in range(num_vehicles):
            mu = velocities[i]
            a, b = (vmin - mu) / sigma, (vmax - mu) / sigma
            vt_next[i] = truncnorm.rvs(
                a, b, loc=mu, scale=sigma, random_state=np_random
            )
        velocities = vt_next

        positions.append(
            positions[-1] + velocities[:, np.newaxis] * direction[:, np.newaxis] * dt
        )

    experiment_positions = np.stack(positions)[np.arange(0, T, DT)]

    selected_vehicles = []
    covered_by_selected = []

    while True:
        # count the number of vehicles that cover each position for each timestep
        coverage_counts = np.zeros((len(experiment_positions), num_vehicles))
        coverage_vehicles = [
            [[] for _ in range(len(experiment_positions))] for __ in range(num_vehicles)
        ]
        for t, period in enumerate(experiment_positions):
            if len(selected_vehicles) != 0:
                ignore_set = set(
                    np.concatenate([v[t] for v in covered_by_selected])
                    .flatten()
                    .tolist()
                )
            else:
                ignore_set = set()

            for i, vehicle in enumerate(period):
                distances = np.linalg.norm(vehicle - period, axis=1)
                coverage_vehicles[i][t].extend(
                    set(np.where(distances <= coverage_radius)[0].tolist()) - ignore_set
                )
                coverage_counts[t, i] += len(coverage_vehicles[i][t])

        avg_counts = np.mean(coverage_counts, axis=0)
        best_vehicle = np.argmax(avg_counts)
        if avg_counts[best_vehicle] < min_vehicles:
            break

        if best_vehicle not in selected_vehicles:
            selected_vehicles.append(best_vehicle.item())
            covered_by_selected.append(coverage_vehicles[best_vehicle])

    return selected_vehicles
