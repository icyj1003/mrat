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

import os


for policy in cache_policy:
    cmd = f"python run.py --cache_policy {policy} --name cache_{policy}"
    os.system(cmd)
    # print(cmd)
