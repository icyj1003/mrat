trade_off = [
    (0, 1),
    (1, 0),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.7, 0.3),
]

import os


for pair in trade_off:
    cmd = f"python run.py --cost_weight {pair[0]} --delay_weight {pair[1]} --name weighting_{pair[0]}_{pair[1]}"
    # os.system(cmd)
    print(cmd)
