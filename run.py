import numpy as np
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
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")

    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    current, writer = get_logger(args)
    env = get_environment(args)

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
            action_dim=2,
        )
    else:
        raise ValueError(f"Unknown delivery policy: {args.delivery_policy}")

    total_episodes = (
        (args.training_episodes + args.evaluation_episodes)
        if args.delivery_policy in ["mappo", "drl_selective"]
        else args.evaluation_episodes
    )

    infos = []
    workload = {}
    accumulate_reward_track = []
    activated_links_track = []

    for episode in tqdm(range(total_episodes), desc="Running", unit="episode"):
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

        if args.cache_policy == "split_non_redundant":
            env.large_step(cache_actions, caching_vehicle, vehicle_cache_actions)
        else:
            env.large_step(cache_actions, caching_vehicle)

        actions_track = []
        while not env.is_small_done():
            active_mask = getattr(
                env, "active_vehicle_mask", np.ones(args.num_vehicles, dtype=bool)
            )

            state_tensor = torch.tensor(
                env.states, dtype=torch.float32, device=args.device
            )
            mask_tensor = torch.tensor(
                env.masks, dtype=torch.float32, device=args.device
            )

            actions, log_probs = delivery_model.act(state_tensor, mask_tensor)
            reshaped_actions = actions.view(args.num_vehicles, env.num_rats)

            active_indices = np.where(active_mask)[0]

            actions_track.append(reshaped_actions[active_indices].cpu().numpy())

            next_states, rewards, dones, violations = env.small_step(reshaped_actions)

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
            active_mask_tensor = torch.tensor(
                active_mask, dtype=torch.float32, device=args.device
            ).view(-1, 1)

            delivery_model.store_transition(
                state_tensor,
                mask_tensor,
                reshaped_actions,
                log_probs,
                reward_tensor,
                next_state_tensor,
                done_tensor,
                violation_tensor,
                active_mask_tensor,
            )

            if (
                args.delivery_policy in ["mappo", "drl_selective"]
                and episode > 0
                and delivery_model.steps % args.small_train_per_n_steps == 0
            ):
                if episode < args.training_episodes:
                    delivery_model.train()

        workload.update({episode: env.load_ratios_track})

        # compute mean used links for each vehicle:
        # for each timestep, for each vehicle, sum the action
        # then for each vehicle sum across timesteps and divide by the vehicle delay (which is the number of timesteps it was active)

        actions_track = np.array(
            actions_track
        )  # shape: (timesteps, num_active_vehicles, num_rats)

        mean_ep_activated_links = []
        for vehicle_id in active_indices:
            mean_vehicle_links = np.sum(
                actions_track[: int(env.delay[vehicle_id]), vehicle_id, :], axis=-1
            )
            mean_ep_activated_links.append(
                np.mean(mean_vehicle_links) if len(mean_vehicle_links) > 0 else 0.0
            )
        activated_links_track.append(np.mean(mean_ep_activated_links))

        infos.append(
            log_and_collect(
                writer,
                env,
                episode,
            )
        )
        infos[-1]["num_caching_vehicles"] = len(caching_vehicle)

        accumulate_reward_track.append(
            infos[-1]["cumulative_reward"] / env.active_num_vehicles
        )

        try:
            window = 100
            if len(accumulate_reward_track) > window:
                moving_avg = float(np.mean(accumulate_reward_track[-window:]))
            else:
                moving_avg = 0.0
        except Exception:
            moving_avg = 0.0

        if writer is not None:
            if moving_avg != 0.0:
                writer.add_scalar(f"log/reward_moving_avg", moving_avg, episode)

            writer.add_scalar(
                "log/episode_avg_cumulative_reward",
                (
                    accumulate_reward_track[-1]
                    if len(accumulate_reward_track) > 0
                    else infos[-1]["cumulative_reward"] / env.num_vehicles
                ),
                episode,
            )

        if writer is not None:
            writer.add_scalar(
                f"log/episode_avg_activated_links",
                activated_links_track[-1] if len(activated_links_track) > 0 else 0.0,
                episode,
            )

        env.reset()

    evaluate = aggregate_metrics(infos[-args.evaluation_episodes :])
    evaluate["num_vehicles"] = args.num_vehicles
    evaluate["num_edges"] = args.num_edges
    evaluate["num_items"] = args.num_items
    evaluate["name"] = args.name
    evaluate["avg_activated_links"] = np.mean(
        activated_links_track[-args.evaluation_episodes :]
    )

    torch.save(
        {
            "args": args,
            "delivery_model": delivery_model.model(),
            "evaluate": evaluate,
            "infos": infos,
            "workload": workload,
            "links": activated_links_track,
        },
        f"runs/{current}_{args.name}/model.pth",
    )

    print(f"[{current}] Evaluation Metrics {args.name} ===========================")
    for key, value in evaluate.items():
        print(f"{key}: {value}")

    with open("./out.out", "a") as f:
        f.write(
            f"[{current}] Evaluation Metrics {args.name} ===========================\n"
        )
        for key, value in evaluate.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
