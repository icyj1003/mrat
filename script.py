import os
from argparse import ArgumentParser


def get_candidate(code):
    if code == "0":
        cache = "heuristic"
        delivery = "mappo"
    elif code == "1":
        cache = "random"
        delivery = "mappo"
    elif code == "2":
        cache = "none"
        delivery = "mappo"
    elif code == "3":
        cache = "heuristic"
        delivery = "random"
    elif code == "4":
        cache = "heuristic"
        delivery = "all"
    elif code == "5":
        cache = "heuristic"
        delivery = "drl_selective"
    else:
        raise ValueError("Invalid code")
    return cache, delivery


def v_scaling():
    num_vehicles_list = [10, 20, 30, 40, 50]
    for num_vehicles in num_vehicles_list:
        cmd = f"python run.py --num_vehicles {num_vehicles} --name vehicle_scale_{num_vehicles}"
        os.system(cmd)


def l_removal():
    cache, delivery = get_candidate(opts.code)
    cmds = [
        "python run.py --remove_v2v --name no_v2v-{cache}-{delivery} --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --remove_wifi --name no_wifi-{cache}-{delivery} --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --remove_pc5 --name no_pc5-{cache}-{delivery} --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
    ]
    for cmd in cmds:
        os.system(cmd)


def deadline():
    cache, delivery = get_candidate(opts.code)
    cmds = [
        "python run.py --delivery_deadline_min 50 --delivery_deadline_max 150 --name dl50-150-{cache}-{delivery} --cost_weight 0.3 --delay_weight 0.7 --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --delivery_deadline_min 25 --delivery_deadline_max 75 --name dl25-75-{cache}-{delivery} --cost_weight 0.3 --delay_weight 0.7 --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --delivery_deadline_min 10 --delivery_deadline_max 20 --name dl10-20-{cache}-{delivery} --cost_weight 0.3 --delay_weight 0.7 --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --delivery_deadline_min 10 --delivery_deadline_max 11 --name dl10-11-{cache}-{delivery} --cost_weight 0.3 --delay_weight 0.7 --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
    ]
    for cmd in cmds:
        os.system(cmd)


def item_size():
    cache, delivery = get_candidate(opts.code)
    cmds = [
        "python run.py --item_size_min 100 --item_size_max 200 --name is100-200-{cache}-{delivery} --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
        "python run.py --item_size_min 200 --item_size_max 400 --name is200-400-{cache}-{delivery} --cache_policy {cache} --delivery_policy {delivery}".format(
            cache=cache, delivery=delivery
        ),
    ]
    for cmd in cmds:
        os.system(cmd)


def cache_policy():
    cache_policy = [
        "random",
        "none",
        "heuristic",
        "heuristic_no_deadline",
        "heuristic_no_popularity",
        "heuristic_no_size",
        "heuristic_no_deadline_popularity",
        "heuristic_no_deadline_size",
        "heuristic_no_popularity_size",
    ]

    for policy in cache_policy:
        cmd = f"python run.py --cache_policy {policy} --name cache_{policy}"
        os.system(cmd)


args = ArgumentParser()
args.add_argument(
    "--v_scaling", action="store_true", help="Run vehicle scaling experiments"
)
args.add_argument(
    "--l_removal", action="store_true", help="Run link removal experiments"
)
args.add_argument(
    "--cache_policy", action="store_true", help="Run cache policy experiments"
)

args.add_argument(
    "--deadline", action="store_true", help="Run delivery deadline experiments"
)
args.add_argument("--item_size", action="store_true", help="Run item size experiments")

args.add_argument(
    "--code",
    type=str,
    help="Code for specific cache/delivery policy combination",
    default="0",
)
opts = args.parse_args()

if opts.v_scaling:
    v_scaling()

if opts.cache_policy:
    cache_policy()

if opts.l_removal:
    l_removal()

if opts.deadline:
    deadline()

if opts.item_size:
    item_size()
