from typing import List
import numpy as np
from scipy.stats import truncnorm


def GTVS(
    env,
    min_vehicles: int = 1,
    DT: int = 5,
    T: int = 100,
) -> List[int]:
    # Extract environment variables
    positions = env.positions.copy()
    velocities = env.velocities.copy()
    direction = env.direction.copy()
    vmin, vmax = float(env.vmin), float(env.vmax)
    dt = float(env.dt)
    coverage_radius = float(env.v2v_pc5_coverage)
    seed = env.seed
    sigma = 0.5
    rng = np.random.RandomState(seed)
    num_vehicles = positions.shape[0]

    # Simulate vehicle movement
    all_positions = [positions.copy()]
    for _ in range(1, T):
        new_velocities = np.zeros(num_vehicles)
        for i in range(num_vehicles):
            mu = velocities[i]
            a, b = (vmin - mu) / sigma, (vmax - mu) / sigma
            new_velocities[i] = truncnorm.rvs(
                a, b, loc=mu, scale=sigma, random_state=rng
            )
        velocities = new_velocities

        dx = velocities * direction * dt
        dy = np.zeros_like(dx)  # assuming 1D road
        positions = all_positions[-1] + np.column_stack((dx, dy))
        all_positions.append(positions)

    # Sample positions every DT steps
    sampled_positions = np.stack(all_positions)[np.arange(0, T, DT)]

    selected_vehicles = []
    covered_history = []  # stores coverage sets of selected vehicles at each time

    while True:
        coverage_counts = np.zeros((len(sampled_positions), num_vehicles))
        coverage_sets = [
            [set() for _ in range(len(sampled_positions))] for _ in range(num_vehicles)
        ]

        for t, snapshot in enumerate(sampled_positions):
            ignore_set = set()
            for covered in covered_history:
                ignore_set.update(covered[t])

            for i, vehicle_pos in enumerate(snapshot):
                distances = np.linalg.norm(snapshot - vehicle_pos, axis=1)
                covered_indices = (
                    set(np.where(distances <= coverage_radius)[0]) - ignore_set
                )
                coverage_sets[i][t] = covered_indices
                coverage_counts[t, i] = len(covered_indices)

        avg_coverage = coverage_counts.mean(axis=0)
        best_vehicle = int(np.argmax(avg_coverage))

        if avg_coverage[best_vehicle] < min_vehicles:
            break

        if best_vehicle not in selected_vehicles:
            selected_vehicles.append(best_vehicle)
            covered_history.append(coverage_sets[best_vehicle])

    return selected_vehicles
