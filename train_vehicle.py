import os

VEH = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


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
        VEH = VEH[::-1]

    for num_vehicles in VEH:
        run_name = f"num_vehicles_{num_vehicles}" + (
            "_single" if args.use_single else ""
        )
        cmd = (
            f"python run.py --name {run_name} --num_vehicles {num_vehicles} --cuda"
            + (" --delivery_policy drl_selective" if args.use_single else "")
        )
        print(f"Running command: {cmd}")
        os.system(cmd)
