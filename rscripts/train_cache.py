import os

cache_capacity = [x * 1024 for x in [2, 3, 4, 5, 6, 7, 8, 9]]
item_size = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
cache_policies = ["heuristic", "none", "split_non_redundant", "random"]
model_path = (
    "/media/anda-network/hdd/khoi/mrat/runs/2026-06-08-16-38-41_w1_55_w2_45/model.pth"
)

for policy in cache_policies:
    for cache in cache_capacity:
        cmd = f"""
        python eval.py
            --model_path {model_path}
            --episodes 100
            --rsu_cache_capacity {cache}
            --name cachecapa_{cache}_{policy}
            --cache_policy {policy}
        """.replace("\n", " ")
        print(cmd)
        os.system(cmd)

    for item in item_size:
        cmd = f"""
        python eval.py
            --model_path {model_path}
            --episodes 100
            --item_size {item}
            --name itemsize_{item}_{policy}
            --cache_policy {policy}
        """.replace("\n", " ")
        print(cmd)
        os.system(cmd)
