# combine coords
import os
import numpy as np

from useful_functs import functions

pat_hnscc = []
pat_facial = []

# load in pickle file for 1 to 6 brainstem
root_hnscc = r'/home/olive/data/HNSCC_deepmind_cropped'
# hnscc_coords
hnscc_coords = functions.load_obj_pickle(root_hnscc, 'coords_' + 'Oli')
for k in hnscc_coords.keys():
    pat_hnscc.append(k)

# load in pickle file for 1 to 10 facial
root_fac = r'/home/oliver_umney/data/Facial_asymmetry_reclicks'
# facial_coords
facial_coords_oli = functions.load_obj_pickle(root_fac, 'coords_' + 'Oli')
facial_coords_aaron = functions.load_obj_pickle(root_fac, 'coords_' + 'Aaron')
for k in facial_coords_oli.keys():
    pat_facial.append(k)

# if patient in both add to list
patients = [x for x in pat_hnscc if x in pat_facial]

# create new dict
coords = {}
crop_coords = {}

landmarks = [1,2,3,4,5,6,7,8,9,10]

# mean of oli and aaron
for patient in patients:
    coords[patient] = {}
    for l in landmarks:
        x = (facial_coords_oli[patient][l]['x'] + facial_coords_aaron[patient][l]['x'])/2
        y = (facial_coords_oli[patient][l]['y'] + facial_coords_aaron[patient][l]['y'])/2
        z = (facial_coords_oli[patient][l]['z'] + facial_coords_aaron[patient][l]['z'])/2
        coords[patient][l] = {'x':x, 'y': y, 'z':z}
    # check if upside down
    if coords[patient][3]['z'] < coords[patient][1]['z']: # head lower than angle
        print('upside down facial')
        print(patient)
        
landmarks_hnscc = [1,2,3,4,5,6]
landmarks_hnscc_left = [1,5]
landmarks_hnscc_right = [2,6]

for patient in patients:
    for l in landmarks_hnscc:
        coords[patient][l+10] = {'x':hnscc_coords[patient][l]['x'], 'y': hnscc_coords[patient][l]['y'], 'z': hnscc_coords[patient][l]['z'] }
    
for patient in patients:
    if coords[patient][15]['z'] < coords[patient][14]['z']:
        print('upside down hnscc')
        print(patient)
            
print(coords)
    
landmarks_tot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# bottom of brainstem crop around
for patient in patients:
    crop_coords[patient] ={'x':coords[patient][13]['x'],'y':coords[patient][13]['y'], 'z':coords[patient][13]['z']}
        
# save both pickle files in new folder (n)
save_root = r'/home/oliver_umney/data/HNSCC_&_facial'
try: 
    os.mkdir(save_root)
except OSError as error:
    print(error)

# 1->10 is facial; 11->16 is hnscc

# save pickle files
functions.save_obj_pickle(coords, save_root, 'coords_Oli')
functions.save_obj_pickle(crop_coords, save_root, 'crop_coords_Oli')

# save CTs in new folder
cts = os.path.join(save_root, "CTs")
try:
    os.mkdir(cts)
except OSError as error:
    print(error)

for patient in patients:
    img_path = os.path.join(root_fac, "CTs", patient) # image path is combination of root and index 
    img = np.load(img_path) # image read in as numpy array
    ct_path = os.path.join(cts, patient)
    np.save(ct_path, img)
    
        

