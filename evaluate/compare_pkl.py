# compare two pickle files

import pickle
import os
import numpy as np

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

landmarks = [1,2,3,4,5,6,7,8,9,10]

x_dev = {}
y_dev = {}
z_dev = {}

for l in landmarks:
    x_dev[l] = []

for k in dict_1.keys():
    for l in landmarks:
        x = dict_1[k][l]['x'] - dict_2[k][l]['x']
        x_dev[l].append(x)
        y = dict_1[k][l]['y'] - dict_2[k][l]['y']
        y_dev[l].append(y)
        z = dict_1[k][l]['z'] - dict_2[k][l]['z']
        z_dev[l].append(z)
    
for l in landmarks:
    print(np.mean(x_dev[l]))
