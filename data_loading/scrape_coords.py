# scrape coords and save

import os
import pickle
import numpy as np

from data_loading import numpy_loc

train_or_test = input("Scraping for train(y) or test(n) ")
if train_or_test == "y":
    clicker = "Oli"
    struc_path = r'/home/oli/data/paed_dataset/train'
elif train_or_test == "n":
    clicker = "Abby"
    struc_path =  r'/home/oli/data/paed_dataset/test' 
    
def save_obj(obj, name):
    with open(os.path.join(struc_path, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(struc_path, name) + '.pkl', 'rb') as f:
        return pickle.load(f)


struc_list = list(sorted(os.listdir(os.path.join(struc_path, "Structures"))))
structure_path = os.path.join(struc_path, "Structures")
print(structure_path)
print(struc_list)

landmarks_total = [1,2,3,4,5,6,7,8,9,10]
landmarks_total_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7:'com',8:'com', 9:'com',10:'com', } 

coordinates = {}

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    for k in landmarks_total:
        coordinates[i][k] = {'x':0, 'y':0, 'z':0}


for i in struc_list:
    structure = np.load(os.path.join(structure_path, i))
    for l in landmarks_total:
        # structure is z, y, x
        # need it in y, x, z
        coord_calc =  numpy_loc.landmark_loc_np(landmarks_total_loc[l],structure,l, i)
        coords, coords_present = coord_calc[0], coord_calc[1]
        if sum(coords) != 0 :
            x, y, z = coords[0], coords[1], coords[2]
            coordinates[i][l]['x'], coordinates[i][l]['y'], coordinates[i][l]['z'] = x,y,z
            #coordinates[i][l]['locat'] = S.landmarks_total_loc[l]
            coordinates[i][l]['present'] = 1 # 1 
        else:
            coordinates[i][l]['present'] = 0

            
save_obj(coordinates, 'coords_%s' % clicker)

a = load_obj('coords_%s' % clicker)
print(a)

