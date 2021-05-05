# work out slide coords
import numpy as np
import os

import settings as S

def coords(patient, in_x, in_y, in_z, sliding_points):
    
    # return ct for patient
    #patients = list(sorted(os.listdir(os.path.join(S.root, "CTs"))))
    img_path = os.path.join(S.root, "CTs", patient) # image path is combination of root and index 
    img = np.load(img_path) # image read in as numpy array
    
    # width, height, depth
    depth, height, width = img.shape[0], img.shape[1], img.shape[2]
    
    
    # calculate locations for crops
    coords = {}
    for s in range(sliding_points):
        coords[s] = {}
    #coords['x'], coords['y'], coords['z'] = in_x/2, in_y/2, in_z/2
    
    index = 0
    for x in range(5):
        for y in range(5):
            for z in range(3):
                coords[index]['x'], coords[index]['y'], coords[index]['z'] = np.clip((in_x/2)*(1+x),0,width), np.clip((in_y/2)*(1+y),0,height), np.clip((in_z/2)*(1+z), 0, depth)   
                index += 1
    
    # return correct dict
    return coords

