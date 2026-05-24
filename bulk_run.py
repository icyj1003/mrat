import os

code = "1"

for c in code:
    os.system(f"python script.py --code {c} --v_scaling")
    # os.system(f"python script.py --code {c} --v_scaling --deadline --item_size")
