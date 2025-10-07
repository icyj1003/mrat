import os
from argparse import ArgumentParser


def v_scaling():
    num_vehicles_list = [10, 20, 30, 40, 50]
    for num_vehicles in num_vehicles_list:
        cmd = f"python run.py --num_vehicles {num_vehicles} --name vehicle_scale_{num_vehicles}"
        os.system(cmd)


def l_removal():
    cmds = [
        "python run.py --remove_v2v --name no_v2v",
        "python run.py --remove_wifi --name no_wifi",
        "python run.py --remove_pc5 --name no_pc5",
    ]
    for cmd in cmds:
        os.system(cmd)


args = ArgumentParser()
args.add_argument(
    "--v_scaling", action="store_true", help="Run vehicle scaling experiments"
)
args.add_argument(
    "--l_removal", action="store_true", help="Run link removal experiments"
)
opts = args.parse_args()

if opts.v_scaling:
    v_scaling()

if opts.l_removal:
    l_removal()
