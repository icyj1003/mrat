delivery_policy = ["random", "all"]

import os


for policy in delivery_policy:
    cmd = f"python run.py --delivery_policy {policy} --name delivery_{policy}"
    os.system(cmd)
    # print(cmd)
