import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the application.")
    parser.add_argument(
        "--name", type=str, default="default", help="Name of the run for logging"
    )
    parser.add_argument(
        "--num_vehicles", type=int, default=30, help="Number of vehicles"
    )
    parser.add_argument("--num_edges", type=int, default=4, help="Number of edges")
    parser.add_argument("--num_items", type=int, default=500, help="Number of items")
    parser.add_argument(
        "--training_episodes",
        type=int,
        default=5000,
        help="Training episodes",
    )
    parser.add_argument(
        "--evaluation_episodes", type=int, default=100, help="Evaluation episodes"
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=128, help="Mini batch size"
    )
    parser.add_argument(
        "--large_train_per_n_eps",
        type=int,
        default=20,
        help="Steps per batch for the delivery",
    )
    parser.add_argument(
        "--small_train_per_n_steps",
        type=int,
        default=512,
        help="Steps per batch for the delivery",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="PPO epsilon (exploration constraint)",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--entropy_coeff", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--penalty_coeff", type=float, default=1.0, help="Penalty coefficient"
    )
    parser.add_argument("--gae_lambda", type=float, default=0.97, help="GAE lambda")
    parser.add_argument(
        "--tau", type=float, default=1e-2, help="Soft update coefficient"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--delivery_deadline_min",
        type=int,
        default=100,
        help="Minimum delivery deadline",
    )
    parser.add_argument(
        "--delivery_deadline_max",
        type=int,
        default=300,
        help="Maximum delivery deadline",
    )
    parser.add_argument(
        "--item_size_max", type=int, default=100, help="Maximum item size"
    )
    parser.add_argument(
        "--item_size_min", type=int, default=50, help="Minimum item size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dt", type=int, default=1, help="Time step size")
    parser.add_argument("--cost_weight", type=float, default=0.5, help="Cost weight")
    parser.add_argument("--delay_weight", type=float, default=0.5, help="Delay weight")
    parser.add_argument(
        "--remove_v2v", action="store_true", help="Remove V2V communication"
    )
    parser.add_argument(
        "--remove_wifi", action="store_true", help="Remove WiFi communication"
    )
    parser.add_argument(
        "--remove_pc5", action="store_true", help="Remove PC5 communication"
    )
    parser.add_argument(
        "--cache_policy",
        type=str,
        default="heuristic",
        choices=[
            "random",
            "none",
            "heuristic",
            "heuristic_no_deadline",
            "heuristic_no_popularity",
            "heuristic_no_size",
            "heuristic_no_deadline_popularity",
            "heuristic_no_deadline_size",
            "heuristic_no_popularity_size",
        ],
    )
    parser.add_argument(
        "--delivery_policy",
        type=str,
        default="mappo",
        choices=["greedy", "random", "all", "mappo", "selective"],
    )
    parser.add_argument(
        "--vehicle_selection_policy",
        type=str,
        default="gtvs_min2",
        choices=["gtvs_min1", "gtvs_min2", "gtvs_min3", "random", "none", "clustering"],
        help="Policy for selecting vehicles",
    )
    parser.add_argument(
        "--remove_edge_cooperation",
        action="store_true",
        help="Remove edge cooperation for caching decisions",
    )
    return parser.parse_args()


# python run.py --cache_policy random --delivery_policy random --vehicle_selection_policy gtvs --training_episodes 1 --evaluation_episodes 10
