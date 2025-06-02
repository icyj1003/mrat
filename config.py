import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the application.")
    parser.add_argument(
        "--num_vehicles", type=int, default=20, help="Number of vehicles"
    )
    parser.add_argument("--num_edges", type=int, default=4, help="Number of edges")
    parser.add_argument("--num_items", type=int, default=50, help="Number of items")
    parser.add_argument("--episode", type=int, default=5000, help="Number of episodes")
    parser.add_argument(
        "--evaluation_episodes", type=int, default=1000, help="Evaluation episodes"
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=32, help="Mini batch size"
    )
    parser.add_argument(
        "--steps_per_batch",
        type=int,
        default=1024,
        help="Steps per batch for the delivery",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size"
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
        default=50,
        help="Minimum delivery deadline",
    )
    parser.add_argument(
        "--delivery_deadline_max",
        type=int,
        default=100,
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
    parser.add_argument("--cost_weight", type=float, default=0.3, help="Cost weight")
    parser.add_argument("--delay_weight", type=float, default=0.7, help="Delay weight")

    return parser.parse_args()
