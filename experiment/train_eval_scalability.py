"""Train a single model at one vehicle count, then evaluate it across many
vehicle counts (scalability study).

The MAPPO/RAT-selection actor is parameter-shared per vehicle, so a model
trained at one count can be evaluated at any other count without retraining
(see eval.py --num_vehicles).

Run from the repo root, e.g.:
    python experiment/train_eval_scalability.py --name myrun
    python experiment/train_eval_scalability.py --use_single --episodes 200
    python experiment/train_eval_scalability.py --skip_train   # reuse latest model
"""

import argparse
import glob
import os
import subprocess
import sys
import time

TRAIN_VEH = 30
EVAL_VEH = [5, 10, 15, 20, 25, 30, 35, 40]

multi_weight = (0.55, 0.45)  # cost - delay
single_weight = (0.5, 0.5)  # cost - delay


def run(cmd):
    print(f"\n>>> {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {cmd}")


def newest_model(name, min_mtime=0.0):
    """Return the most recent .output/runs/*_{name}/model.pth, or None."""
    candidates = [
        p
        for p in glob.glob(f"./.output/runs/*_{name}/model.pth")
        if os.path.getmtime(p) >= min_mtime
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="scalability", help="Base run name")
    parser.add_argument("--train_vehicles", type=int, default=TRAIN_VEH)
    parser.add_argument(
        "--eval_vehicles",
        type=int,
        nargs="+",
        default=EVAL_VEH,
        help="Vehicle counts to evaluate the trained model on",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Evaluation episodes per count"
    )
    parser.add_argument(
        "--use_single",
        action="store_true",
        help="Single-link transmission (delivery_policy=drl_selective)",
    )
    parser.add_argument(
        "--model_path", default=None, help="Use this model.pth and skip training"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Reuse the latest matching trained model instead of training",
    )
    args = parser.parse_args()

    policy = "drl_selective" if args.use_single else "mappo"
    weight = single_weight if args.use_single else multi_weight
    train_name = args.name + ("_single" if args.use_single else "")

    # 1) Train once (unless a model is supplied / reuse requested)
    model_path = args.model_path
    if model_path is None:
        if args.skip_train:
            model_path = newest_model(train_name)
            if model_path is None:
                sys.exit(f"No existing model for '{train_name}'; drop --skip_train")
        else:
            start = time.time()
            run(
                f"python run.py --name {train_name}"
                f" --num_vehicles {args.train_vehicles} --cuda"
                f" --delivery_policy {policy}"
                f" --cost_weight {weight[0]} --delay_weight {weight[1]}"
            )
            model_path = newest_model(train_name, min_mtime=start)
            if model_path is None:
                sys.exit("Training finished but model.pth was not found")

    print(f"\nUsing model: {model_path}")

    # 2) Evaluate it across the requested vehicle counts
    for veh in args.eval_vehicles:
        run(
            f"python eval.py --model_path {model_path}"
            f" --num_vehicles {veh}"
            f" --episodes {args.episodes}"
            f" --name {train_name}_eval_veh{veh}"
        )

    print("\nDone. Eval results saved under .output/runs/eval_* (run aggregate.py).")
