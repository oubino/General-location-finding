# scrape coords and save
import pickle
import os
import numpy as np

coding_path = r'C:\Users\olive\OneDrive\Documents\GitHub\General-location-finding'

os.chdir(coding_path)

import settings as S
from data_loading import numpy_loc

S.init()
    
test = False
#clicker = 'Oli'


def save_obj(obj, name):
    with open(os.path.join(S.root, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(S.root, name) + '.pkl', 'rb') as f:
        return pickle.load(f)


struc_list = list(sorted(os.listdir(os.path.join(S.root, "Structures"))))
structure_path = os.path.join(S.root, "Structures")
print(struc_list)
"""
coordinates = {}

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    for k in S.landmarks_total:
        coordinates[i][k] = {'x':0, 'y':0, 'z':0}


for i in struc_list:
    structure = np.load(os.path.join(structure_path, i))
    for l in S.landmarks_total:
        # structure is z, y, x
        # need it in y, x, z
        coord_calc =  numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, i, test)
        coords, coords_present = coord_calc[0], coord_calc[1]
        if sum(coords) != 0 :
            x, y, z = coords[0], coords[1], coords[2]
            #structure_mod[z][y][x] = l
            coordinates[i][l]['x'], coordinates[i][l]['y'], coordinates[i][l]['z'] = x,y,z
            coordinates[i][l]['locat'] = S.landmarks_total_loc[l]
            coordinates[i][l]['present'] = True
        else:
            coordinates[i][l]['present'] = False
            coordinates[i][l]['locat'] = 'absent'
            
            

save_obj(coordinates, 'coords_%s' % S.clicker)
"""
a = load_obj('coords_Oli')
print(a)

for i in struc_list:
    for l in S.landmarks_total:
        print(a[i][l]['present'])