import argparse
import os
import torch
from tqdm import tqdm

from policy.cache_policy import (
    heuristic_cache_placement,
    no_cache_placement,
    non_redundant_cache_placement,
    random_cache_placement,
)

from policy.delivery_policy import (
    AllLinkDeliveryPolicy,
    GreedyDeliveryPolicy,
    MAPPODeliveryPolicy,
    RandomDeliveryPolicy,
    RATSelection,
    TrueAllLink,
    CheapSel,
    GA,
)

from policy.selection_policy import (
    GTVS,
    clustering_vehicle_selection,
    no_vehicle_selection,
    random_vehicle_selection,
)

from utils import (
    aggregate_metrics,
    get_environment,
    log_and_collect,
)


def build_delivery_model(args, env):

    if args.delivery_policy == "mappo":
        return MAPPODeliveryPolicy(args, env)

    elif args.delivery_policy == "drl_selective":
        return RATSelection(args, env)

    elif args.delivery_policy == "all":
        return AllLinkDeliveryPolicy()

    elif args.delivery_policy == "true_all_link":
        return TrueAllLink(
            args,
            env,
        )

    elif args.delivery_policy == "cheapselect":
        return CheapSel(
            args,
            env,
        )

    elif args.delivery_policy == "greedy":
        return GreedyDeliveryPolicy(
            args,
            env,
        )

    elif args.delivery_policy == "ga":
        return GA(
            args,
            env,
        )

    elif args.delivery_policy == "random":
        return RandomDeliveryPolicy(
            num_agents=args.num_vehicles,
            num_actions=env.num_rats,
            action_dim=2,
        )

    raise ValueError(f"Unknown delivery policy: {args.delivery_policy}")


def select_caching_vehicles(args, env):

    if args.vehicle_selection_policy == "gtvs_min1":
        return GTVS(env, min_vehicles=1)

    elif args.vehicle_selection_policy == "gtvs_min2":
        return GTVS(env, min_vehicles=2)

    elif args.vehicle_selection_policy == "gtvs_min3":
        return GTVS(env, min_vehicles=3)

    elif args.vehicle_selection_policy == "clustering":
        return clustering_vehicle_selection(
            env,
            num_clusters=6,
        )

    elif args.vehicle_selection_policy == "random":
        return random_vehicle_selection(
            env,
            num_vehicles=6,
        )

    return no_vehicle_selection(env)


def cache_placement(args, env, caching_vehicle):

    if args.cache_policy == "heuristic":

        return (
            heuristic_cache_placement(env),
            None,
        )

    elif args.cache_policy == "heuristic_no_deadline":

        return (
            heuristic_cache_placement(
                env,
                use_deadline=False,
                use_popularity=True,
                use_size=True,
            ),
            None,
        )

    elif args.cache_policy == "heuristic_no_popularity":

        return (
            heuristic_cache_placement(
                env,
                use_deadline=True,
                use_popularity=False,
                use_size=True,
            ),
            None,
        )

    elif args.cache_policy == "heuristic_no_size":

        return (
            heuristic_cache_placement(
                env,
                use_deadline=True,
                use_popularity=True,
                use_size=False,
            ),
            None,
        )

    elif args.cache_policy == "random":

        return (
            random_cache_placement(env),
            None,
        )

    elif args.cache_policy == "none":

        return (
            no_cache_placement(env),
            None,
        )

    elif args.cache_policy == "split_non_redundant":

        vehicle_cache_actions, cache_actions = non_redundant_cache_placement(
            env,
            caching_vehicle,
        )

        return (
            cache_actions,
            vehicle_cache_actions,
        )
    elif args.cache_policy == "split_non_redundant_rsu":
        vehicle_cache_actions, cache_actions = non_redundant_cache_placement(
            env,
            caching_vehicle,
            priority="rsu",
        )

        return (
            cache_actions,
            vehicle_cache_actions,
        )

    raise ValueError(f"Unknown cache policy: {args.cache_policy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument("--remove_pc5", action="store_true")
    parser.add_argument("--remove_wifi", action="store_true")
    parser.add_argument("--remove_v2v", action="store_true")
    parser.add_argument("--remove_v2n", action="store_true")
    parser.add_argument("--remove_edge_cooperation", action="store_true")
    parser.add_argument("--item_size", type=int, default=None)
    parser.add_argument("--veh_cache_capacity", type=int, default=None)
    parser.add_argument("--rsu_cache_capacity", type=int, default=None)
    parser.add_argument("--cache_policy", type=str, default=None)
    parser.add_argument("--name", type=str, default="evaluation")
    parser.add_argument("--deadline", type=int, default=None)
    parser.add_argument("--num_vehicles", type=int, default=None)

    args_eval = parser.parse_args()

    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    model_path = Path(args_eval.model_path)

    train_folder = model_path.parent.name
    eval_folder = f"eval_{train_folder}"

    writer = SummaryWriter(log_dir=f"./.output/runs/{eval_folder}_{args_eval.name}")

    checkpoint = torch.load(
        args_eval.model_path,
        map_location="cpu",
        weights_only=False,
    )

    args = checkpoint["args"]
    args.name = args_eval.name

    print(f"Evaluation logs: ./.output/runs/{eval_folder}_{args_eval.name}")

    # Override training args with evaluation args
    args.remove_pc5 = args_eval.remove_pc5
    args.remove_wifi = args_eval.remove_wifi
    args.remove_v2v = args_eval.remove_v2v
    args.remove_v2n = args_eval.remove_v2n
    args.remove_edge_cooperation = args_eval.remove_edge_cooperation

    if args_eval.cache_policy is not None:
        args.cache_policy = args_eval.cache_policy

    if args_eval.item_size is not None:
        args.item_size_max = args_eval.item_size + 1
        args.item_size_min = args_eval.item_size

    if args_eval.deadline is not None:
        args.delivery_deadline_min = args_eval.deadline
        args.delivery_deadline_max = args_eval.deadline + 1

    # Override the number of vehicles for scalability evaluation. The MAPPO
    # actor is parameter-shared per-vehicle, so a model trained at one vehicle
    # count can be evaluated at any other count without retraining.
    if args_eval.num_vehicles is not None:
        args.num_vehicles = args_eval.num_vehicles

    args.training_episodes = 0
    args.evaluation_episodes = args_eval.episodes

    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    print(f"Loaded experiment: {args.name}")
    print(f"Running {args_eval.episodes} evaluation episodes")

    torch.manual_seed(args.seed)

    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = get_environment(args)

    if args_eval.veh_cache_capacity is not None:
        env.vehicle_capacity = args_eval.veh_cache_capacity

    if args_eval.rsu_cache_capacity is not None:
        env.edge_capacity = args_eval.rsu_cache_capacity

    delivery_model = build_delivery_model(
        args,
        env,
    )

    #
    # Load actor weights
    #
    if args.delivery_policy in [
        "mappo",
        "drl_selective",
    ]:

        delivery_model.agent.actor.load_state_dict(checkpoint["delivery_model"])

        delivery_model.agent.actor.eval()

    infos = []

    action_track = {}
    workload = {}

    for episode in tqdm(
        range(args.evaluation_episodes),
        desc="Evaluating",
        unit="episode",
    ):

        #
        # Large timescale
        #
        caching_vehicle = select_caching_vehicles(
            args,
            env,
        )

        cache_actions, vehicle_cache_actions = cache_placement(
            args,
            env,
            caching_vehicle,
        )

        if args.cache_policy == "split_non_redundant":

            env.large_step(
                cache_actions,
                caching_vehicle,
                vehicle_cache_actions,
            )

        else:

            env.large_step(
                cache_actions,
                caching_vehicle,
            )

        #
        # Small timescale
        #
        while not env.is_small_done():
            state_tensor = torch.tensor(
                env.states,
                dtype=torch.float32,
                device=args.device,
            )

            mask_tensor = torch.tensor(
                env.masks,
                dtype=torch.float32,
                device=args.device,
            )

            with torch.no_grad():

                actions, _ = delivery_model.act(
                    state_tensor,
                    mask_tensor,
                    projection=env.bandwidth_constraints_handler,
                )

            reshaped_actions = actions.view(
                args.num_vehicles,
                env.num_rats,
            )

            env.small_step(reshaped_actions)

        workload[episode] = env.load_ratios_track
        action_track[episode] = env.action_track

        infos.append(
            log_and_collect(
                writer=writer,
                env=env,
                episode=episode,
            )
        )

        infos[-1]["num_caching_vehicles"] = len(caching_vehicle)

        env.reset()

    results = aggregate_metrics(infos)

    print("\n")
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for key, value in results.items():

        mean = value["value"]
        std = value["std"]

        print(f"{key}: " f"{mean:.4f} ± {std:.4f}")

    os.makedirs(f"./.output/runs/{eval_folder}_{args_eval.name}", exist_ok=True)

    torch.save(
        {
            "args": args,
            "evaluate": results,
            "infos": infos,
            "workload": workload,
            "action_track": action_track,
            "name": args_eval.name,
        },
        f"./.output/runs/{eval_folder}_{args_eval.name}/model.pth",
    )

    print(
        f"\nSaved evaluation results to:\n"
        f"{args_eval.model_path.replace('model.pth', 'evaluation.pth')}"
    )
