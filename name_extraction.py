import numpy as np
import os

#scenario_name = input('scenario name: ')
root_dir = 'data/'


for root, dirs, files in os.walk(root_dir):
    for scenario_name in dirs:
        path = os.path.join(root_dir, scenario_name, "GT", "cross_walk")
        for inner_root, inner_dirs, inner_files in os.walk(path):
            pass
        # Extract cross_walk files' name 
        num_list = []
        for name in inner_files:
            num_list.append(int(name.split('.npy')[0]))
        first_num = min(num_list)
        last_num = max(num_list)
        # Print lines
        cmd = scenario_name.split('_')[2]
        print("        samples += self.add_scenarios('data/{}', 'ClearNoon_', {}, {}, {})".format(scenario_name, str(first_num), str(last_num), cmd))

    break   # Only traverse the first layer