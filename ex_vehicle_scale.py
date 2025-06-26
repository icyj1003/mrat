num_vehicles_list = [10, 20, 30, 40, 50]

import os


for num_vehicles in num_vehicles_list:
    cmd = f"python run.py --num_vehicles {num_vehicles} --name vehicle_scale_{num_vehicles}"
    os.system(cmd)
