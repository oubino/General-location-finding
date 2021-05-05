# flip upside down cts and structures

# scrape coords and save

import os
import pickle
import numpy as np

import settings as S
from data_loading import numpy_loc
S.init(False)

root = r'/home/oli/data/paed_dataset'

    
def save_obj(obj, name):
    with open(os.path.join(struc_path, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(struc_path, name) + '.pkl', 'rb') as f:
        return pickle.load(f)


struc_list = list(sorted(os.listdir(os.path.join(root, "Structures"))))
structure_path = os.path.join(root, "Structures")
print(structure_path)
print(struc_list)


coordinates = {}

# if bottom of spinal chord is above cochleas flip!
landmarks = [4,5]
landmarks_loc = {4:'bot', 5:'com'}

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    for k in landmarks:
        coordinates[i][k] = {'x':0, 'y':0, 'z':0}


for i in struc_list:
    structure = np.load(os.path.join(structure_path, i))
    for l in landmarks:
        # structure is z, y, x
        # need it in y, x, z
        coord_calc =  numpy_loc.landmark_loc_np(landmarks_loc[l],structure,l, i)
        coords, coords_present = coord_calc[0], coord_calc[1]
        if sum(coords) != 0 :
            x, y, z = coords[0], coords[1], coords[2]
            #structure_mod[z][y][x] = l
            coordinates[i][l]['x'], coordinates[i][l]['y'], coordinates[i][l]['z'] = x,y,z
            coordinates[i][l]['present'] = 1 # 1 
        else:
            #coordinates[i][l]['locat'] = 'absent'
            coordinates[i][l]['present'] = 0
            
# upside down

ct_flips = os.path.join(root, "CTs_flip")
struc_flips = os.path.join(root, "Structures_flip")
try: 
    os.mkdir(ct_flips)
    os.mkdir(struc_flips)
except OSError as error:
    print(error)


for i in struc_list:
    if coordinates[i][4]['z'] > coordinates[i][5]['z']:
        print('upside down')
        print(i)
        # flip and save ct
        img_path = os.path.join(root, "CTs", i) 
        struc_path = os.path.join(root, "Structures", i)
        img_save_path = os.path.join(ct_flips, i) 
        struc_save_path = os.path.join(struc_flips, i)
        img = np.load(img_path)
        struc = np.load(struc_path)
        img_flip = np.flip(img, axis=0)
        struc_flip = np.flip(struc, axis=0)
        np.save(img_save_path, img_flip) 
        np.save(struc_save_path, struc_flip)


