# compare two pickle files

import pickle
import os

root_1 = r'/home/oli/data/results/oli/run_folder/eval_100_4/'
name_1 = 'final_loc'
root_2 = r'/home/oli/data/temp_folder/'
name_2 = 'xyz?'

with open(os.path.join(root_1, name_1) + '.pkl', 'rb') as f:
        dict_1 = pickle.load(f)

with open(os.path.join(root_2, name_2) + '.pkl', 'rb') as f:
        dict_2 = pickle.load(f)

print(dict_1)
print(dict_2)