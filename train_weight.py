import os

WEIGHTS = [
    (0.5, 0.5),
    (0, 1),
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (1, 0),
]

WEIGHTS_point5 = [
    (0.05, 0.95),
    (0.15, 0.85),
    (0.25, 0.75),
    (0.35, 0.65),
    (0.45, 0.55),
    (0.55, 0.45),
    (0.65, 0.35),
    (0.75, 0.25),
    (0.85, 0.15),
    (0.95, 0.05),
]

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--from_bottom",
    action="store_true",
    help="Whether to start from the bottom of the list of weights",
)
parser.add_argument(
    "--use_single",
    action="store_true",
    help="Enable single link transmission (i.e., only one link can be used for delivery)",
)

parser.add_argument(
    "--use_point5_weights",
    action="store_true",
    help="Whether to use weights with 0.05 increments (e.g., (0.05, 0.95), (0.15, 0.85), etc.)",
)

parser.add_argument(
    "--all_weights",
    action="store_true",
    help="Whether to use weights with 0.05 increments (e.g., (0.05, 0.95), (0.15, 0.85), etc.)",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.use_point5_weights:

        WEIGHTS = WEIGHTS_point5

    if args.from_bottom:
        WEIGHTS = WEIGHTS[::-1]

    if args.all_weights:
        WEIGHTS = WEIGHTS + WEIGHTS_point5

    for cost_weight, delay_weight in WEIGHTS:
        run_name = (
            f"w1_{str(cost_weight * 10).replace('.', '')}_w2_{str(delay_weight * 10).replace('.', '')}"
            + ("_single" if args.use_single else "")
        )
        cmd = (
            f"python run.py --name {run_name} --cost_weight {cost_weight} --delay_weight {delay_weight} --cuda"
            + (" --delivery_policy drl_selective" if args.use_single else "")
        )
        print(f"Running command: {cmd}")
        os.system(cmd)
