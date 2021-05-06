# compare two pickle files

import pickle
import os
import numpy as np
import csv

csv_root = r'/home/oli/data/paed_dataset/test/'

def pat_to_mm(patient):
    data = csv.reader(open(os.path.join(csv_root, 'image_dimensions.csv')),delimiter=',')
    next(data) # skip first line
    list_img = list(data)#, key=operator.itemgetter(0))
    # sortedlist[img_number][0 = name, 1 = x/y, 2 = z]
    #image_idx = int(image_idx)
    pat_ind = patient.replace('.npy','')
    index = 0 
    for i in range(len(list_img)):
        if list_img[i][0] == pat_ind:
            index = i
    pixel_mm_x = list_img[index][1] # 1 pixel = pixel_mm_x * mm
    pixel_mm_y = list_img[index][1]
    pixel_mm_z = list_img[index][2]
    return pixel_mm_x, pixel_mm_y, pixel_mm_z

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
p2p = {}

for l in landmarks:
    x_dev[l] = []
    y_dev[l] = []
    z_dev[l] = [] 
    p2p[l] = []

for k in dict_1.keys():
    for l in landmarks:
        x_mm, y_mm, z_mm = pat_to_mm(k)
        x = dict_1[k][l]['x'] - dict_2[k][l]['x']
        x = x.cpu().numpy()
        x_mod = x * float(x_mm)
        x_dev[l].append(x_mod)
        y = dict_1[k][l]['y'] - dict_2[k][l]['y']
        y = y.cpu().numpy()
        y_mod = y * float(y_mm)
        y_dev[l].append(y_mod)
        z = dict_1[k][l]['z'] - dict_2[k][l]['z']
        z = z.cpu().numpy()
        z_mod = z * float(z_mm)
        z_dev[l].append(z_mod)
        p2p_mod = (x_mod**2+y_mod**2+z_mod**2)**0.5
        p2p[l].append(p2p_mod)
    
for l in landmarks:
    print('landmark', l)
    print(np.mean(p2p[l]))
    

