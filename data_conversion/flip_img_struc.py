# flip upside down cts and structures

import os
import pickle
import numpy as np

from data_loading import numpy_loc

    
def save_obj(obj, path, name):
    with open(os.path.join(path, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    with open(os.path.join(path, name) + '.pkl', 'rb') as f:
        return pickle.load(f)
    
root = r'/home/oli/data/paed_dataset/test'

struc_list = list(sorted(os.listdir(os.path.join(root, "Structures"))))
structure_path = os.path.join(root, "Structures")
print(structure_path)
print(struc_list)


coordinates = {}

# if AMl (1) is above SOl (9)
landmarks = [1,9]
landmarks_loc = {1:'com', 9:'com'}

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    for k in landmarks:
        coordinates[i][k] = {'x':0, 'y':0, 'z':0}


for i in struc_list:
    structure = np.load(os.path.join(structure_path, i))
    for l in landmarks:
        coord_calc =  numpy_loc.landmark_loc_np(landmarks_loc[l],structure,l, i)
        coords, coords_present = coord_calc[0], coord_calc[1]
        if sum(coords) != 0 :
            x, y, z = coords[0], coords[1], coords[2]
            coordinates[i][l]['x'], coordinates[i][l]['y'], coordinates[i][l]['z'] = x,y,z
            coordinates[i][l]['present'] = 1 # 1 
        else:
            coordinates[i][l]['present'] = 0
            
# upside down
"""
ct_flips = os.path.join(root, "CTs_flip")
struc_flips = os.path.join(root, "Structures_flip")
try: 
    os.mkdir(ct_flips)
    os.mkdir(struc_flips)
except OSError as error:
    print(error)
"""

for i in struc_list:
    print(i)
    if coordinates[i][1]['z'] > coordinates[i][9]['z']:
        print('upside down')
        print(i)
        # flip and save ct
    #    img_path = os.path.join(root, "CTs", i) 
    #    struc_path = os.path.join(root, "Structures", i)
    #    img_save_path = os.path.join(ct_flips, i) 
    #    struc_save_path = os.path.join(struc_flips, i)
    #    img = np.load(img_path)
    #    struc = np.load(struc_path)
    #    img_flip = np.flip(img, axis=0)
    #    struc_flip = np.flip(struc, axis=0)
    #    np.save(img_save_path, img_flip) 
    #    np.save(struc_save_path, struc_flip)


