import torch
import numpy as np
from scipy.optimize import curve_fit
import csv, operator
import settings as S
import os

def landmark_loc_np(locat, structure, landmark, patient):
    if locat == 'com':
        return com_structure_np(structure,landmark, patient)
    elif locat == 'top':
        return top_structure_np(structure,landmark, patient)
    elif locat == 'bot':
        return bot_structure_np(structure,landmark, patient)
        
def com_structure_np(structure, landmark, patient): # assumes 1 channel
  # structure is (D x H x W)
  # output is x,y,z
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  locations = np.nonzero(np.round(structure) == landmark)
  
  if (len(locations[0]) == 0): # if no landmarks detected for structure
    print('no structure found using np for %1.0f for image % s' % (landmark,patient))
    landmark_present.append(False)
    x_com = 0
    y_com = 0
    z_com = 0
  else:
    landmark_present.append(True)
    x_com = 0
    y_com = 0
    z_com = 0
    for k in range(len(locations[0])): # number of landmarks
      x_com += locations[2][k]
      y_com += locations[1][k]
      z_com += locations[0][k]
    x_com /= len(locations[0])
    y_com /= len(locations[0])
    z_com /= len(locations[0])
  coords = [int(x_com),int(y_com),int(z_com)]
  return coords, landmark_present 

def top_structure_np(structure, landmark, patient): # assumes 1 channel
  # structure is (D x H x W)
  # output is x,y,z
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  locations = np.nonzero(np.round(structure) == landmark)
  
  if (len(locations[0]) == 0): # if no landmarks detected for structure
    print('no structure found using np for %1.0f for image % s' % (landmark,patient))
    landmark_present.append(False)
    x_top = 0
    y_top = 0
    z_top = 0
  else:
    landmark_present.append(True)
    z_coords = locations[0]
    #print('here')
    #top_coords = locations[np.argmax(z_coords)]
    x_top = locations[2][np.argmax(z_coords)]#top_coords[2]
    y_top = locations[1][np.argmax(z_coords)]
    z_top = locations[0][np.argmax(z_coords)]
  coords = [x_top, y_top, z_top]
  return coords, landmark_present   

def bot_structure_np(structure, landmark, patient): # assumes 1 channel
  # structure is (D x H x W)
  # output is x,y,z
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  locations = np.nonzero(np.round(structure) == landmark)
  
  if (len(locations[0]) == 0): # if no landmarks detected for structure
    print('no structure found using np for %1.0f for image % s' % (landmark,patient))
    landmark_present.append(False)
    x_bot = 0
    y_bot = 0
    z_bot = 0
  else:
    landmark_present.append(True)
    z_coords = locations[0]
    #print('here')
    #top_coords = locations[np.argmax(z_coords)]
    x_bot = locations[2][np.argmin(z_coords)]#top_coords[2]
    y_bot = locations[1][np.argmin(z_coords)]
    z_bot = locations[0][np.argmin(z_coords)]
  coords = [x_bot, y_bot, z_bot]
  return coords, landmark_present  


