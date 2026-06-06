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

if __name__ == "__main__":
    args = parser.parse_args()

    if args.from_bottom:
        WEIGHTS = WEIGHTS[::-1]

    for cost_weight, delay_weight in WEIGHTS:
        run_name = f"w1_{int(cost_weight * 10)}_w2_{int(delay_weight * 10)}" + (
            "_single" if args.use_single else ""
        )
        os.system(
            f"python run.py --name {run_name} --cost_weight {cost_weight} --delay_weight {delay_weight} --cuda"
            + (" --delivery_policy drl_selective" if args.use_single else "")
        )
