import torch
from tqdm import tqdm

from config import parse_args
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
from utils import aggregate_metrics, get_environment, get_logger, log_and_collect

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")

    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    # set random seed
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Setup logger
    current, writer = get_logger(args)

    # Get the environment
    env = get_environment(args)

    # Initialize delivery policies
    if args.delivery_policy == "mappo":
        delivery_model = MAPPODeliveryPolicy(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "drl_selective":
        delivery_model = RATSelection(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "all":
        delivery_model = AllLinkDeliveryPolicy()
    elif args.delivery_policy == "true_all_link":
        delivery_model = TrueAllLink(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "cheapselect":
        delivery_model = CheapSel(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "greedy":
        delivery_model = GreedyDeliveryPolicy(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "ga":
        delivery_model = GA(
            args,
            env,
            writer=writer,
        )
    elif args.delivery_policy == "random":
        delivery_model = RandomDeliveryPolicy(
            num_agents=args.num_vehicles,
            num_actions=env.num_rats,
            action_dim=2,  # Assuming action_dim is 2 for the delivery policy
        )
    else:
        raise ValueError(f"Unknown delivery policy: {args.delivery_policy}")

    # Compute the total episodes: include training only for learning policies
    total_episodes = (
        (args.training_episodes + args.evaluation_episodes)
        if args.delivery_policy in ["mappo", "drl_selective"]
        else args.evaluation_episodes
    )

    # evaluation metrics tracking
    infos = []
    workload = {}

    # Begin training loop
    for episode in tqdm(range(total_episodes), desc="Running", unit="episode"):
        # At Large time-scale:
        # Step 1: Run the vehicle selection policy here
        if args.vehicle_selection_policy == "gtvs_min1":
            caching_vehicle = GTVS(env, min_vehicles=1)
        elif args.vehicle_selection_policy == "gtvs_min2":
            caching_vehicle = GTVS(env, min_vehicles=2)
        elif args.vehicle_selection_policy == "gtvs_min3":
            caching_vehicle = GTVS(env, min_vehicles=3)
        elif args.vehicle_selection_policy == "clustering":
            caching_vehicle = clustering_vehicle_selection(env, num_clusters=6)
        elif args.vehicle_selection_policy == "random":
            caching_vehicle = random_vehicle_selection(env, num_vehicles=6)
        else:
            caching_vehicle = no_vehicle_selection(env)

        # Step 2: Make caching decisions
        # initialize caching policy
        if args.cache_policy == "heuristic":
            cache_actions = heuristic_cache_placement(env)
        elif args.cache_policy == "heuristic_no_deadline":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=False, use_popularity=True, use_size=True
            )
        elif args.cache_policy == "heuristic_no_popularity":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=True, use_popularity=False, use_size=True
            )
        elif args.cache_policy == "heuristic_no_size":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=True, use_popularity=True, use_size=False
            )
        elif args.cache_policy == "heuristic_no_deadline_popularity":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=False, use_popularity=False, use_size=True
            )
        elif args.cache_policy == "heuristic_no_deadline_size":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=False, use_popularity=True, use_size=False
            )
        elif args.cache_policy == "heuristic_no_popularity_size":
            cache_actions = heuristic_cache_placement(
                env, use_deadline=True, use_popularity=False, use_size=False
            )
        elif args.cache_policy == "random":
            cache_actions = random_cache_placement(env)
        elif args.cache_policy == "none":
            cache_actions = no_cache_placement(env)
        elif args.cache_policy == "split_non_redundant":
            vehicle_cache_actions, cache_actions = non_redundant_cache_placement(
                env,
                caching_vehicle,
            )
        else:
            raise ValueError(f"Unknown cache policy: {args.cache_policy}")

        # Overwrite the cache states in the environment before performing the small step
        if args.cache_policy == "split_non_redundant":
            env.large_step(cache_actions, caching_vehicle, vehicle_cache_actions)
        else:
            env.large_step(cache_actions, caching_vehicle)

        # Small time-scale:
        # Run the multi-agent delivery policy here
        while not env.is_small_done():
            # Convert to tensor
            state_tensor = torch.tensor(
                env.states, dtype=torch.float32, device=args.device
            )
            mask_tensor = torch.tensor(
                env.masks, dtype=torch.float32, device=args.device
            )

            # MAPPO Policy action selection
            actions, log_probs = delivery_model.act(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions
                projection=env.bandwidth_constraints_handler,
            )

            # Reshape action to match the environment
            reshaped_actions = actions.view(args.num_vehicles, env.num_rats)

            # Step the environment
            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

            # Convert to tensor
            reward_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=args.device
            ).view(-1, 1)
            done_tensor = torch.tensor(
                dones, dtype=torch.float32, device=args.device
            ).view(-1, 1)
            violation_tensor = torch.tensor(
                violations, dtype=torch.float32, device=args.device
            ).view(-1, 1)
            next_state_tensor = torch.tensor(
                next_states, dtype=torch.float32, device=args.device
            )

            # Store the transition in the delivery model
            delivery_model.store_transition(
                state_tensor,  # num_agents x state_dim
                mask_tensor,  # num_agents x num_actions
                reshaped_actions,  # num_agents x num_actions
                log_probs,  # num_agents x num_actions
                reward_tensor,  # num_agents x 1
                next_state_tensor,  # num_agents x state_dim
                done_tensor,  # num_agents x 1
                violation_tensor,  # num_agents x 1
            )

            # Update the delivery model only for learning policies
            if (
                args.delivery_policy in ["mappo", "drl_selective"]
                and episode > 0
                and delivery_model.steps % args.small_train_per_n_steps == 0
            ):
                if episode < args.training_episodes:
                    delivery_model.train()

        # update workload
        workload.update({episode: env.load_ratios_track})

        # Collect episode information
        infos.append(
            log_and_collect(
                writer,
                env,
                episode,
            )
        )
        infos[-1]["num_caching_vehicles"] = len(caching_vehicle)
        # print(infos[-1]["episode_length"])
        # print(infos[-1]["violation_ratio"])

        # Reset the environment
        env.reset()

    # Aggregate the evaluation metrics
    evaluate = aggregate_metrics(infos[-args.evaluation_episodes :])
    evaluate["num_vehicles"] = args.num_vehicles
    evaluate["num_edges"] = args.num_edges
    evaluate["num_items"] = args.num_items
    evaluate["name"] = args.name

    # Save the model and metrics
    torch.save(
        {
            "args": args,
            "delivery_model": delivery_model.model(),
            "evaluate": evaluate,
            "infos": infos,
            "workload": workload,
        },
        f"runs/{current}_{args.name}/model.pth",  # Save the model with the current time and name
    )

    # Print the evaluation metrics
    print(f"[{current}] Evaluation Metrics {args.name} ===========================")
    for key, value in evaluate.items():
        print(f"{key}: {value}")

    # Write evaluation metrics to ./out.out
    with open("./out.out", "a") as f:
        f.write(
            f"[{current}] Evaluation Metrics {args.name} ===========================\n"
        )
        for key, value in evaluate.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
