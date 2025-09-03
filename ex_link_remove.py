cmd = [
    "python run.py --remove_v2v --name remove_v2v",
    "python run.py --remove_wifi --name remove_wifi",
    "python run.py --remove_pc5 --name remove_pc5",
]

import os

for command in cmd:
    os.system(command)
    # print(command)

# python run.py --remove_edge_cooperation --name no_edge_cooperation
