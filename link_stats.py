"""
Link statistics run: drives the real content-delivery simulation (caching,
requests, AllLinkDeliveryPolicy) so link availability naturally reflects cache
state (in particular V2V, which is only "available" when a nearby vehicle has
the requested item cached) - but reports only physical-layer link statistics,
not delivery outcome (no cost/deadline/hit-rate/fairness reporting).

Reports two things:
  - availability (%): fraction of vehicle-steps a link was usable, computed
    from physical conditions (out-of-road, coverage distance, disable flags)
    rather than env.masks (which also disables everything once a vehicle's
    request is already complete - see compute_availability's docstring).
    V2V and the two "(direct, cached)" rows additionally require an actual
    cache hit (V2V: a nearby vehicle has the item; PC5/WiFi direct: the
    local edge has the item) - mirroring how content availability, not just
    radio range, gates real usage.
  - potential data rate (Mbps): the throughput each of the 8 delivery paths
    (V2N, V2V, V2I-PC5 x{direct,remote-to-local,backhaul-to-local},
    V2I-WiFi x{direct,remote-to-local,backhaul-to-local}) would achieve.
    Direct-path rates are computed only over the cache-hit vehicle-steps
    above; remote-to-local/backhaul rates are computed unconditionally
    whenever the parent link is in range, regardless of where the content
    actually sits.

Vehicles that drive off the road are teleported back to the start of the
road from the opposite direction (a looping/circular road), so link
statistics don't decay over a long run as vehicles permanently exit.

Usage:
    python link_stats.py --name my_run --num_vehicles 30 --num_episodes 20
"""

import argparse
import csv
import datetime
import os

import numpy as np
import torch

from environ.env import Environment
from environ.utils import compute_data_rate
from policy.cache_policy import (
    heuristic_cache_placement,
    no_cache_placement,
    random_cache_placement,
)
from policy.delivery_policy import AllLinkDeliveryPolicy
from policy.selection_policy import GTVS, no_vehicle_selection, random_vehicle_selection

RATS = ("v2n", "v2v", "v2i_pc5", "v2i_wifi")
RAT_LABELS = {
    "v2n": "V2N",
    "v2v": "V2V",
    "v2i_pc5": "V2I-PC5",
    "v2i_wifi": "V2I-WiFi",
}

# Direct-link availability that also folds in the chance the local edge
# actually has the requested item cached - the same treatment V2V gets
# (nearby_vehicles_of only counts a neighbor that has the item cached).
DIRECT_CACHE_LINKS = ("v2i_pc5_direct", "v2i_wifi_direct")
DIRECT_CACHE_LABELS = {
    "v2i_pc5_direct": "V2I-PC5 (direct, cached)",
    "v2i_wifi_direct": "V2I-WiFi (direct, cached)",
}

# Remote-to-local availability additionally requires a NEIGHBOR edge (not the
# local one) to have the item cached - unlike backhaul-to-local, which is a
# fallback to the core network and is therefore always reachable.
REMOTE_CACHE_LINKS = ("v2i_pc5_remote", "v2i_wifi_remote")
REMOTE_CACHE_LABELS = {
    "v2i_pc5_remote": "V2I-PC5 (remote, neighbor-cached)",
    "v2i_wifi_remote": "V2I-WiFi (remote, neighbor-cached)",
}

CASES = (
    "v2n",
    "v2v",
    "v2i_pc5_direct",
    "v2i_pc5_remote",
    "v2i_pc5_bs",
    "v2i_wifi_direct",
    "v2i_wifi_remote",
    "v2i_wifi_bs",
)
CASE_LABELS = {
    "v2n": "V2N",
    "v2v": "V2V",
    "v2i_pc5_direct": "V2I-PC5 (direct)",
    "v2i_pc5_remote": "V2I-PC5 (remote-to-local)",
    "v2i_pc5_bs": "V2I-PC5 (bs-to-edge-to-local)",
    "v2i_wifi_direct": "V2I-WiFi (direct)",
    "v2i_wifi_remote": "V2I-WiFi (remote-to-local)",
    "v2i_wifi_bs": "V2I-WiFi (bs-to-edge-to-local)",
}

RELAY_HOP_DISTANCE = 1  # assumed hop count for the "remote-to-local" cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report link availability and potential data rate, driven by the real content-delivery simulation."
    )
    parser.add_argument("--name", type=str, default="link_stats", help="Run name")
    parser.add_argument("--num_vehicles", type=int, default=30)
    parser.add_argument("--num_edges", type=int, default=4)
    parser.add_argument("--num_items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=1, help="Time step size")
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="Number of caching/delivery episodes"
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=30,
        help=(
            "Cap on small-steps per episode (also a safety net in case delivery "
            "never completes). Defaults to delivery_deadline_max so episodes stay "
            "close to a realistic delivery window."
        ),
    )

    parser.add_argument("--delivery_deadline_min", type=int, default=10)
    parser.add_argument("--delivery_deadline_max", type=int, default=30)
    parser.add_argument("--item_size_min", type=int, default=50)
    parser.add_argument("--item_size_max", type=int, default=100)

    parser.add_argument(
        "--cache_policy",
        type=str,
        default="heuristic",
        choices=["heuristic", "random", "none"],
    )
    parser.add_argument(
        "--vehicle_selection_policy",
        type=str,
        default="gtvs_min2",
        choices=["gtvs_min2", "random", "none"],
    )

    # Ablation
    parser.add_argument("--remove_v2v", action="store_true")
    parser.add_argument("--remove_wifi", action="store_true")
    parser.add_argument("--remove_pc5", action="store_true")
    parser.add_argument("--remove_v2n", action="store_true")

    return parser.parse_args()


def get_environment(args) -> Environment:
    env = Environment(
        num_vehicles=args.num_vehicles,
        num_edges=args.num_edges,
        num_items=args.num_items,
        seed=args.seed,
        dt=args.dt,
        delivery_deadline_min=args.delivery_deadline_min,
        delivery_deadline_max=args.delivery_deadline_max,
        item_size_min=args.item_size_min,
        item_size_max=args.item_size_max,
        disable_v2v=args.remove_v2v,
        disable_wifi=args.remove_wifi,
        disable_pc5=args.remove_pc5,
        disable_v2n=args.remove_v2n,
    )
    env.reset()
    _patch_wraparound(env)
    return env


def _patch_wraparound(env: Environment) -> None:
    """
    Replace Environment.update_position (called internally by small_step) with a
    version that teleports out-of-bound vehicles to the start of the road from the
    opposite side, instead of leaving them permanently "out". This keeps V2I/V2N
    availability from decaying over a long run without altering environ/env.py.
    """

    def wrapped_update_position(self: Environment) -> None:
        self.positions[:, 0] = (
            self.positions[:, 0] + self.velocities * self.dt * self.direction
        ) % self.road_length
        self.update_mobility_status()

    env.update_position = wrapped_update_position.__get__(env, Environment)


def get_caching_vehicles(env: Environment, args):
    if args.vehicle_selection_policy == "gtvs_min2":
        return GTVS(env, min_vehicles=2)
    elif args.vehicle_selection_policy == "random":
        return random_vehicle_selection(env, num_vehicles=6)
    else:
        return no_vehicle_selection(env)


def get_cache_actions(env: Environment, args):
    if args.cache_policy == "heuristic":
        return heuristic_cache_placement(env)
    elif args.cache_policy == "random":
        return random_cache_placement(env)
    else:
        return no_cache_placement(env)


def compute_availability(env: Environment):
    """
    Pure physical-layer availability per link, independent of whether a vehicle
    still has an active request. `env.masks` is NOT used here because it also
    disables every link once delivery_done==1 (env.py:561) - that reflects "no
    longer needs it", not "can't physically use it", and would otherwise make
    V2N/V2I-PC5 (which have no other gating condition) look unavailable for any
    vehicle that already finished downloading.
    """
    num_vehicles = env.num_vehicles

    v2n_avail = np.full(num_vehicles, not env.disable_v2n)

    v2v_avail = (env.nearby_vehicles_of != -1) & (not env.disable_v2v)

    # v2i_pc5 has no coverage-distance gating in the real sim - only "not out of
    # road" (env.py:525). With the wraparound patch, out is always 0, so this is
    # ~100% unless explicitly disabled.
    pc5_avail = (env.out == 0) & (not env.disable_pc5)

    wifi_avail = (
        (env.out == 0)
        & (env.local_edge_distance < env.v2i_wifi_coverage)
        & (not env.disable_wifi)
    )

    # Cache-hit-gated direct availability: physically reachable AND the local
    # edge actually has the vehicle's requested item cached (mirrors V2V's
    # cache-gated availability, applied to the direct-to-local-edge path).
    cache_hit_local = env.cache[env.local_of, env.requested] == 1
    pc5_direct_avail = pc5_avail & cache_hit_local
    wifi_direct_avail = wifi_avail & cache_hit_local

    # Cache-hit-gated remote-to-local availability: local edge misses, but some
    # OTHER edge has the item cached (env.py's "hop_distance < 99" check) - this
    # is not always true, unlike the backhaul-to-local case below, which is a
    # fallback to the core network and is therefore always reachable whenever
    # the parent link itself is in range (no cache condition to satisfy).
    edge_cache = env.cache[: env.num_edges, :]  # (num_edges, num_items)
    has_item_per_edge = edge_cache[:, env.requested]  # (num_edges, num_vehicles)
    edges_with_item = has_item_per_edge.sum(axis=0) - cache_hit_local.astype(int)
    neighbor_has_item = (edges_with_item > 0) & (not env.remove_edge_cooperation)

    pc5_remote_avail = pc5_avail & (~cache_hit_local) & neighbor_has_item
    wifi_remote_avail = wifi_avail & (~cache_hit_local) & neighbor_has_item

    return (
        v2n_avail,
        v2v_avail,
        pc5_avail,
        wifi_avail,
        pc5_direct_avail,
        wifi_direct_avail,
        pc5_remote_avail,
        wifi_remote_avail,
    )


def collect_step_stats(env: Environment, avail_slots: dict, avail_total: dict, rate_samples: dict):
    num_vehicles = env.num_vehicles
    for rat in RATS + DIRECT_CACHE_LINKS + REMOTE_CACHE_LINKS:
        avail_total[rat] += num_vehicles

    (
        v2n_avail,
        v2v_avail,
        pc5_avail,
        wifi_avail,
        pc5_direct_avail,
        wifi_direct_avail,
        pc5_remote_avail,
        wifi_remote_avail,
    ) = compute_availability(env)

    avail_slots["v2n"] += int(v2n_avail.sum())
    avail_slots["v2v"] += int(v2v_avail.sum())
    avail_slots["v2i_pc5"] += int(pc5_avail.sum())
    avail_slots["v2i_wifi"] += int(wifi_avail.sum())
    avail_slots["v2i_pc5_direct"] += int(pc5_direct_avail.sum())
    avail_slots["v2i_wifi_direct"] += int(wifi_direct_avail.sum())
    avail_slots["v2i_pc5_remote"] += int(pc5_remote_avail.sum())
    avail_slots["v2i_wifi_remote"] += int(wifi_remote_avail.sum())

    # --- V2N: potential rate, ignoring cache status ---
    if v2n_avail.any():
        rate = (
            compute_data_rate(
                allocated_spectrum=env.v2n_bandwidth,
                transmission_power=env.v2n_transmission_power,
                noise_power=env.noise_power,
                distance=env.bs_distance[v2n_avail],
                path_loss_model="macro",
            )
            / 1e6
        )
        rate_samples["v2n"].extend(rate.tolist())

    # --- V2V: cache status already gates availability (nearby_vehicles_of only
    # points at a nearby vehicle that has the requested item cached); the rate
    # itself is purely distance-based ---
    if v2v_avail.any():
        idx = np.where(v2v_avail)[0]
        neighbor_idx = env.nearby_vehicles_of[idx]
        distance = env.vehicle_distance[idx, neighbor_idx]
        rate = (
            compute_data_rate(
                allocated_spectrum=env.v2v_bandwidth,
                transmission_power=env.v2v_transmission_power,
                noise_power=env.noise_power,
                distance=distance,
                path_loss_model="micro",
            )
            / 1e6
        )
        rate_samples["v2v"].extend(rate.tolist())

    # --- V2I-PC5: backhaul-to-local is always reachable whenever the parent link
    # is in range (fallback to the core network, no cache condition); direct and
    # remote-to-local are gated by an actual cache hit, locally or at a neighbor
    # edge respectively (see pc5_direct_avail / pc5_remote_avail) ---
    if pc5_avail.any():
        base_rate = compute_data_rate(
            allocated_spectrum=env.v2i_pc5_bandwidth,
            transmission_power=env.v2i_pc5_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[pc5_avail],
            path_loss_model="micro",
        )
        backhaul = env.i2n_data_rate * base_rate / (base_rate + env.i2n_data_rate)
        rate_samples["v2i_pc5_bs"].extend((backhaul / 1e6).tolist())

    if pc5_direct_avail.any():
        direct = compute_data_rate(
            allocated_spectrum=env.v2i_pc5_bandwidth,
            transmission_power=env.v2i_pc5_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[pc5_direct_avail],
            path_loss_model="micro",
        )
        rate_samples["v2i_pc5_direct"].extend((direct / 1e6).tolist())

    if pc5_remote_avail.any():
        base_rate = compute_data_rate(
            allocated_spectrum=env.v2i_pc5_bandwidth,
            transmission_power=env.v2i_pc5_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[pc5_remote_avail],
            path_loss_model="micro",
        )
        remote = (
            env.i2i_data_rate
            * base_rate
            / (base_rate * RELAY_HOP_DISTANCE + env.i2i_data_rate)
        )
        rate_samples["v2i_pc5_remote"].extend((remote / 1e6).tolist())

    # --- V2I-WiFi: same split as V2I-PC5 above ---
    if wifi_avail.any():
        base_rate = compute_data_rate(
            allocated_spectrum=env.v2i_wifi_bandwidth,
            transmission_power=env.v2i_wifi_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[wifi_avail],
            path_loss_model="micro",
        )
        backhaul = env.i2n_data_rate * base_rate / (base_rate + env.i2n_data_rate)
        rate_samples["v2i_wifi_bs"].extend((backhaul / 1e6).tolist())

    if wifi_direct_avail.any():
        direct = compute_data_rate(
            allocated_spectrum=env.v2i_wifi_bandwidth,
            transmission_power=env.v2i_wifi_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[wifi_direct_avail],
            path_loss_model="micro",
        )
        rate_samples["v2i_wifi_direct"].extend((direct / 1e6).tolist())

    if wifi_remote_avail.any():
        base_rate = compute_data_rate(
            allocated_spectrum=env.v2i_wifi_bandwidth,
            transmission_power=env.v2i_wifi_transmission_power,
            noise_power=env.noise_power,
            distance=env.local_edge_distance[wifi_remote_avail],
            path_loss_model="micro",
        )
        remote = (
            env.i2i_data_rate
            * base_rate
            / (base_rate + env.i2i_data_rate * RELAY_HOP_DISTANCE)
        )
        rate_samples["v2i_wifi_remote"].extend((remote / 1e6).tolist())


def run_simulation(env: Environment, args):
    delivery_model = AllLinkDeliveryPolicy()

    avail_slots = {rat: 0 for rat in RATS + DIRECT_CACHE_LINKS + REMOTE_CACHE_LINKS}
    avail_total = {rat: 0 for rat in RATS + DIRECT_CACHE_LINKS + REMOTE_CACHE_LINKS}
    rate_samples = {case: [] for case in CASES}

    for _ in range(args.num_episodes):
        caching_vehicle = get_caching_vehicles(env, args)
        cache_actions = get_cache_actions(env, args)
        env.large_step(cache_actions, caching_vehicle)

        steps = 0
        while not env.is_small_done() and steps < args.max_steps_per_episode:
            state_tensor = torch.tensor(env.states, dtype=torch.float32)
            mask_tensor = torch.tensor(env.masks, dtype=torch.float32)
            actions, _ = delivery_model.act(
                state_tensor, mask_tensor, projection=env.bandwidth_constraints_handler
            )
            reshaped_actions = actions.view(env.num_vehicles, env.num_rats)

            collect_step_stats(env, avail_slots, avail_total, rate_samples)

            env.small_step(reshaped_actions)
            steps += 1

        env.reset()

    availability = {
        rat: 100.0 * avail_slots[rat] / avail_total[rat] if avail_total[rat] else 0.0
        for rat in RATS + DIRECT_CACHE_LINKS + REMOTE_CACHE_LINKS
    }

    data_rate = {}
    for case in CASES:
        values = np.array(rate_samples[case])
        data_rate[case] = {
            "avg_data_rate_mbps": float(values.mean()) if values.size else 0.0,
            "std_data_rate_mbps": float(values.std()) if values.size else 0.0,
        }

    return availability, data_rate


def format_report(args, current, availability, data_rate) -> str:
    lines = []
    lines.append(
        f"[{current}] Link Statistics {args.name} "
        f"(num_vehicles={args.num_vehicles}, num_episodes={args.num_episodes}, seed={args.seed}) "
        "==========================="
    )

    lines.append("")
    lines.append("Link availability:")
    lines.append(f"{'RAT':<32}{'Avail(%)':>12}")
    for rat in RATS:
        lines.append(f"{RAT_LABELS[rat]:<32}{availability[rat]:>12.2f}")
    for rat in DIRECT_CACHE_LINKS:
        lines.append(f"{DIRECT_CACHE_LABELS[rat]:<32}{availability[rat]:>12.2f}")
    for rat in REMOTE_CACHE_LINKS:
        lines.append(f"{REMOTE_CACHE_LABELS[rat]:<32}{availability[rat]:>12.2f}")

    lines.append("")
    lines.append(
        "Potential data rate (direct/remote require an actual cache hit locally/at "
        "a neighbor edge; backhaul is always reachable when the parent link is):"
    )
    lines.append(f"{'Case':<28}{'AvgRate(Mbps)':>16}{'StdRate(Mbps)':>16}")
    for case in CASES:
        s = data_rate[case]
        lines.append(
            f"{CASE_LABELS[case]:<28}"
            f"{s['avg_data_rate_mbps']:>16.4f}"
            f"{s['std_data_rate_mbps']:>16.4f}"
        )
    return "\n".join(lines)


def write_csv(args, current, availability, data_rate) -> None:
    os.makedirs("./.output", exist_ok=True)

    avail_path = "./.output/link_availability.csv"
    avail_exists = os.path.exists(avail_path)
    with open(avail_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not avail_exists:
            writer.writerow(
                ["run", "timestamp", "seed", "num_vehicles", "link", "availability_pct"]
            )
        for rat in RATS:
            writer.writerow(
                [args.name, current, args.seed, args.num_vehicles, RAT_LABELS[rat], availability[rat]]
            )
        for rat in DIRECT_CACHE_LINKS:
            writer.writerow(
                [
                    args.name,
                    current,
                    args.seed,
                    args.num_vehicles,
                    DIRECT_CACHE_LABELS[rat],
                    availability[rat],
                ]
            )
        for rat in REMOTE_CACHE_LINKS:
            writer.writerow(
                [
                    args.name,
                    current,
                    args.seed,
                    args.num_vehicles,
                    REMOTE_CACHE_LABELS[rat],
                    availability[rat],
                ]
            )

    rate_path = "./.output/link_data_rate.csv"
    rate_exists = os.path.exists(rate_path)
    with open(rate_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not rate_exists:
            writer.writerow(
                [
                    "run",
                    "timestamp",
                    "seed",
                    "num_vehicles",
                    "case",
                    "avg_data_rate_mbps",
                    "std_data_rate_mbps",
                ]
            )
        for case in CASES:
            s = data_rate[case]
            writer.writerow(
                [
                    args.name,
                    current,
                    args.seed,
                    args.num_vehicles,
                    CASE_LABELS[case],
                    s["avg_data_rate_mbps"],
                    s["std_data_rate_mbps"],
                ]
            )


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with args: {args}")

    env = get_environment(args)
    availability, data_rate = run_simulation(env, args)

    current = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = f"./.output/runs/{current}_{args.name}"
    os.makedirs(run_dir, exist_ok=True)

    report = format_report(args, current, availability, data_rate)
    print(report)

    with open(f"{run_dir}/link_stats.out", "a") as f:
        f.write(report + "\n\n")

    write_csv(args, current, availability, data_rate)
