# crop coords create

import os
import pickle
import numpy as np

import settings as S
S.init()

# pickle file location
pickle_struc_path = r'/home/olive/data/Facial_asymmetry_reclicks'

def save_obj(root, obj, name):
    with open(os.path.join(root, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(root, name):
    with open(os.path.join(root, name) + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# load in pickle file
file = load_obj(pickle_struc_path, 'coords_' + S.clicker)

# average of HMl and HMr location
patients = list(file.keys())
crop_coords = {}
for i in patients:
    crop_coords[i] = {}
    x_hml, y_hml, z_hml = file[i][3]['x'], file[i][3]['y'], file[i][3]['z']
    x_hmr, y_hmr, z_hmr = file[i][4]['x'], file[i][4]['y'], file[i][4]['z']
    x = (x_hml + x_hmr)/2
    y = (y_hml + y_hmr)/2
    z = (z_hml + z_hmr)/2
    crop_coords[i]['x'], crop_coords[i]['y'], crop_coords[i]['z'] = x,y,z
    
# save pickel file as crop_coords_clicker
save_obj(pickle_struc_path, crop_coords,'crop_coords_%s' % S.clicker)

a = load_obj(pickle_struc_path, 'crop_coords_%s' % S.clicker)
print(a)
