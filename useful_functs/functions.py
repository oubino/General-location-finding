import torch
import numpy as np
from scipy.optimize import curve_fit
import csv
import os
import math
import pickle
import random

import settings as S

#os.chdir(S.root) # change to data path and change back at end
#print('Data directory: ')
#print(os.getcwd())

# takes in heatmap of B x C x H x W x D
# i.e. batch x 6 channels x H x W x D
# needs to workout max (x,y,z) for specified landmark
def pred_max(heatmap, landmark, landmarks): 
  a = torch.max(heatmap, dim = 4, keepdim = True)
  b = torch.max(a[0],dim =3,keepdim = True)
  c = torch.max(b[0],dim = 2,keepdim = True)
  batch_size = (heatmap.size()[0])
  for i in range(batch_size):
    index = landmarks.index(landmark)
    y = c[1][i][index][0][0][0]
    x = b[1][i][index][y][0][0]
    z = a[1][i][index][y][x][0]
    if i == 0:
      coords = torch.tensor([[x,y,z]]).to(S.device)
    else: 
      coords_temp = torch.tensor([[x,y,z]]).to(S.device)
      coords = torch.cat((coords,coords_temp),dim = 0)

  return coords   

# pred max gives maximum for each z layer between certain range of z 

def gauss_max(heatmap, landmark, height_trained, sigma_trained, in_x, in_y, in_z, landmarks): # this should be current sigma for landmark
    # heatmap is batch of masks or images of dim (B x C x H x W x D)
    # output is (B x coord)
    # i.e. x,y of 3rd image = coords[3][0], coords[3][1] only 1 channel
    pred_coords = pred_max(heatmap, landmark, S.landmarks).cpu().numpy()
    batch_size = (heatmap.size()[0])
    heatmap = heatmap.detach().cpu().numpy()
    index = landmarks.index(landmark)
    for i in range(batch_size):
        t = np.indices((in_y,in_x,in_z)).astype(float)  ## will be ~100,256,256 in your case
        def f(t, mu_i, mu_j, mu_k, height, sigma):                 ## function to be fitted
            pos = np.array([mu_i, mu_j, mu_k])      ## central position of the gaussian peak (using initial estimate of the argmax)
            t = t.reshape((3,in_y,in_x,in_z))           ## will be ~100,256,256 in your case
            dist_map = np.sqrt(np.sum([np.power((t[0] - pos[0]), 2), np.power((t[1] - pos[1]), 2), np.power((in_y/in_z * (t[2] - pos[2])), 2)], axis=0))  ## generate euclidean distance map to sample gauss fn on
            gauss = height * np.exp( - dist_map/ (2.*sigma*sigma) )
            return np.array(gauss, dtype=float).ravel()  ## generate gaussian using the estimated central point (your gauss scale will be different here too)
        #argmax_pred = np.unravel_index(torch.argmax(model_output[0]), model_output.size()[2:])  ## get initial guess of gauss peak (prediction argmax)
        #optimum_pos = curve_fit(f, t.ravel(), model_output[0].numpy().ravel(), p0=argmax_pred) ## optimisation process, compare predicted heatmap with generated gaussians (centre point shifted around to optimum)
        p0 = [pred_coords[i][0], pred_coords[i][1],pred_coords[i][2], height_trained, sigma_trained]
        
        # catch error
        try:
            params = curve_fit(f, t.ravel() , heatmap[i][index].ravel(), p0 = p0)          
            x = params[0][1]
            y = params[0][0]
            z = params[0][2]  

        except RuntimeError:
            print("Error - curve_fit failed")
            x = torch.tensor(0) # change !!!
            y = torch.tensor(0)
            z = torch.tensor(0)
        
        #print(params[0])
        #print(params[1])
        #print(params[2])
        

        x_max = heatmap.shape[3]
        y_max = heatmap.shape[2]
        z_max = heatmap.shape[4]
        x = max(0,min(x,x_max)) # keep x & y in [0,224]
        y = max(0,min(y,y_max))
        z = max(0,min(z,z_max))
        if i == 0:
            coords = torch.tensor([[x,y,z]])
        else: 
            coords_temp = torch.tensor([[x,y,z]])
            coords = torch.cat((coords,coords_temp),dim = 0)
        
    return coords  


def point_to_point(mask_x, mask_y, mask_z, pred_x, pred_y, pred_z):
  # calculates point to point error for image
  point_to_point = ((pred_x - mask_x)**2 + (pred_y - mask_y)**2 + (pred_z - mask_z)**2)**0.5 
  return point_to_point

def point_to_point_mm(mask_x, mask_y, mask_z, pred_x, pred_y, pred_z, patient):
  # calculates point to point in mm
  data = csv.reader(open(os.path.join(S.root, 'image_dimensions.csv')),delimiter=',')
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
  
  #print('pixel x, y, z')
  #print(pixel_mm_x, pixel_mm_y, pixel_mm_z)
  
  pixel_mm_x = torch.tensor(float(pixel_mm_x)).to(S.device)
  pixel_mm_y = torch.tensor(float(pixel_mm_y)).to(S.device)
  pixel_mm_z = torch.tensor(float(pixel_mm_z)).to(S.device)
  
  point_to_point = (((pred_x - mask_x)*pixel_mm_x)**2 + ((pred_y - mask_y)*pixel_mm_y)**2 + ((pred_z - mask_z)*pixel_mm_z)**2)**0.5 
  return point_to_point

def gaussian(x,y,z, targ_coords, sigma, gamma, dimension = 3): # assumes in 2d space
  # x, y are general coords and targ_coords define mean of gaussian
  x_targ, y_targ, z_targ = targ_coords[0], targ_coords[1], targ_coords[2]
  l2_dif = torch.tensor(-((x-x_targ)**2 + (y- y_targ)**2 + (z-z_targ)**2)/(2*sigma**2)).to(S.device)
  gauss = torch.tensor((gamma) * (2*np.pi)**(-dimension/2) * sigma ** (-dimension) ** torch.exp(l2_dif)).to(S.device)
  return gauss

def gaussian_map(peak_x,peak_y,peak_z, sigma,gamma,x_size,y_size, z_size, output,dimension = 3): # 2D gaussian 5x5 image
  if output == False: # expands gaussian map to 448x448 to ensure proper normalisation ? not working
    y,x,z = np.ogrid[0:2*y_size,0:2*x_size, 0:2*z_size]
    peak_x = peak_x + x_size/2
    peak_y = peak_y + y_size/2
    peak_z = peak_z + z_size/2
    pre_factor = ((gamma) * (2*np.pi)**(-dimension/2) * sigma ** (-dimension)) 
    h = pre_factor * torch.exp( -((torch.tensor(x).to(S.device)-peak_x)**2 + (torch.tensor(y).to(S.device)-peak_y)**2 + (torch.tensor(z).to(S.device)-peak_z)**2) / (2.*sigma*sigma) )
  if output == True:
    y,x,z = np.ogrid[0:y_size,0:x_size, 0:z_size]
    pre_factor = ((gamma) * (2*np.pi)**(-dimension/2) * sigma ** (-dimension)) 
    h = pre_factor * torch.exp( -((torch.tensor(x).to(S.device)-peak_x)**2 + (torch.tensor(y).to(S.device)-peak_y)**2 + ((torch.tensor(z).to(S.device)-peak_z)*x_size/z_size)**2) / (2.*sigma*sigma) )
    # note multiply to ensure spherical
  return h

def print_metrics(metrics,  epoch_samples, phase): # not sure??
    print('Phase: Landmark, Value')
    for l in metrics.keys():
        output_per_landmark = []
        for k in metrics[l].keys():
            output_per_landmark.append("{}: {:4f}".format((l,k), metrics[l][k]/epoch_samples))
        print("{}: {}".format(phase, ", ".join(output_per_landmark)))

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') # change if switch to ReLU
        torch.nn.init.zeros_(m.bias)
    
def string(s):
    if s is None:
        return ''
    return '_' + str(s)

def rotate(x_coord, y_coord, z_coord, x_size, y_size, z_size, angle, axis):
    
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))
    
    x_offset = (x_size - 1)/2 #- math.cos(math.radians(angle)) * ((x_size - 1)/2) - math.sin(math.radians(angle)) * ((z_size- 1)/2)
    y_offset = (y_size - 1)/2
    z_offset = (z_size - 1)/2
    
    x,y, z = x_coord - x_offset, y_coord - y_offset, z_coord - z_offset
  
    if axis == [1,0]:
        y_new = s * z + c * y
        z_new = c * z - s * y 
        x_new = x
    elif axis == [1,2]:
        x_new = s * y + c * x
        y_new = c * y - s * x
        z_new = z
    elif axis == [2,0]:                  
        x_new = s * z + c * x
        z_new = c * z - s * x
        y_new = y      

    return x_new + x_offset, y_new + y_offset, z_new + z_offset

def save_obj_pickle(obj, root, name):
    with open(os.path.join(root, name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_pickle(root, name):
    with open(os.path.join(root, name) + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def line_learn_loc(coords_1, coords_2):
    coords = {}
    for k in S.landmarks_total:
        coords[k] = {}
    for k in S.landmarks_total:
        x_1, y_1, z_1 = coords_1[k]['x'], coords_1[k]['y'], coords_1[k]['z']
        x_2, y_2, z_2 = coords_2[k]['x'], coords_2[k]['y'], coords_2[k]['z']
        t = random.random()
        coords[k]['x'] = x_2 + t*(x_2 - x_1)
        coords[k]['y'] = y_2 + t*(y_2 - y_1)
        coords[k]['z'] = z_2 + t*(z_2 - z_1)
        #if coords_1[k]['locat'] != coords_2[k]['locat']:
        #    print('ERROR DIFFERENT TYPES OF POINTS')
        #else:
        #    coords[k]['locat'] = coords_1[k]['locat']
        if coords_1[k]['present'] == 0 or coords_2[k]['present'] == 0:
            coords[k]['present'] = 0
        else:
            coords[k]['present'] = 1
    return coords

def line_learn_crop(coords_1, coords_2):
    coords = {}
    x_1, y_1, z_1 = coords_1['x'], coords_1['y'], coords_1['z']
    x_2, y_2, z_2 = coords_2['x'], coords_2['y'], coords_2['z']
    x = (x_1 + x_2)/2
    y = (y_1 + y_2)/2
    z = (z_1 + z_2)/2
    coords['x'] = x
    coords['y'] = y
    coords['z'] = z
    return coords

def mean_from_clickers(coords_1, coords_2): 
    coords = {}
    for key in coords_1.keys():
        coords[key] = {}
        for k in S.landmarks_total:
            coords[key][k] = {}
    for key in coords_1.keys():
        for k in S.landmarks_total:
            x_1, y_1, z_1 = coords_1[key][k]['x'], coords_1[key][k]['y'], coords_1[key][k]['z']
            x_2, y_2, z_2 = coords_2[key][k]['x'], coords_2[key][k]['y'], coords_2[key][k]['z']
            coords[key][k]['x'] = (x_1 + x_2)/2
            coords[key][k]['y'] = (y_1 + y_2)/2
            coords[key][k]['z'] = (z_1 + z_2)/2
            #if coords_1[key][k]['locat'] != coords_2[key][k]['locat']:
            #    print('ERROR DIFFERENT TYPES OF POINTS')
            #else:
            #    coords[key][k]['locat'] = coords_1[key][k]['locat']
            if coords_1[key][k]['present'] == 0 or coords_2[key][k]['present'] == 0:
                coords[key][k]['present'] = 0
            else:
                coords[key][k]['present'] = 1
    return coords
    

def aug_to_orig(pred_max_x, pred_max_y, pred_max_z, downsample, patient):
    if downsample == True: # convert a downsample
        # convert pred max to location in full size image
          if patient in S.downsample_ratio_list:
            pred_max_x = pred_max_x * S.downsample_ratio_list[patient]['w']
            pred_max_y = pred_max_y * S.downsample_ratio_list[patient]['h']
            pred_max_z = pred_max_z * S.downsample_ratio_list[patient]['d']   
          else:
              print('aug to orig error')    
              exit() # deliberately crash if gets here
              
    elif downsample == False: # convert a crop
        # convert pred max to location in full size image
        if patient in S.crop_list:
            pred_max_x = pred_max_x + S.crop_list[patient]['x_left']
            pred_max_y = pred_max_y + S.crop_list[patient]['y_left']
            pred_max_z = pred_max_z + S.crop_list[patient]['z_left']
        else:
            print('aug to orig error')
            exit()
    
    return pred_max_x, pred_max_y, pred_max_z
    
        
        
        
#os.chdir(S.coding_path) # change to data path and change back at end
#print('Coding directory: ')
#print(os.getcwd())

"""
def landmark_loc(heatmap, landmark):
    batch_size = (heatmap.size()[0])
    landmark_present = []
    landmark = float(landmark) # ensure that comparison is made properly
    for i in range(batch_size):
      # heatmap shape [1, 1, 256, 256, 100] i.e. B x C x H x W x D
      # add in round here to see if picks up landmarks its missing
      locations = (torch.nonzero((torch.round(heatmap[i][0]) == landmark),as_tuple = False)).to(S.device)
      
      if (locations.size(0) == 0): # if no landmarks detected for image in batch
        landmark_present.append(False)
        x = torch.tensor(0, dtype = torch.float64).to(S.device) # in theory this should not be used
        y = torch.tensor(0, dtype = torch.float64).to(S.device)
        z = torch.tensor(0, dtype = torch.float64).to(S.device)
      else:
        landmark_present.append(True)
        x = locations[0][1]
        y = locations[0][0]
        z = locations[0][2]
      if i == 0:
        coords = torch.tensor([[x,y,z]]).to(S.device)
      else: 
        coords_temp = torch.tensor([[x,y,z]]).to(S.device)
        coords = torch.cat((coords,coords_temp),dim = 0)
    # returns arrays of B x coord
    # returns array of B x True/False
    # if True then for that coord it is COM, if false then need to produce heatmap of zeros
    return coords, landmark_present   
    

def landmark_loc(locat, heatmap, landmark):
    if locat == 'com':
        return com_structure(heatmap,landmark)
    elif locat == 'top':
        return top_structure(heatmap,landmark)
    elif locat == 'bot':
        return bot_structure(heatmap,landmark)


def com_structure(heatmap, landmark): # assumes 1 channel
  # heatmap is batch of either masks or image of 
  # heatmap dim is (B x C x H x W x D)
  # output is (B x coord)
  # i.e. x,y,z of 3rd image = coords[3][0],coords[3coo][1], coords[3][2] only 1 channel
  batch_size = (heatmap.size()[0])
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  for i in range(batch_size):
    # heatmap shape [1, 1, 256, 256, 100] i.e. B x C x H x W x D
    # add in round here to see if picks up landmarks its missing
    #locations = (torch.round(heatmap[i][0]) == landmark).nonzero().to(S.device)
    locations = (torch.nonzero((torch.round(heatmap[i][0]) == landmark),as_tuple = False)).to(S.device)
    
    if (locations.size(0) == 0): # if no landmarks detected for image in batch
      #print('no structure for %1.0f' % landmark)
      #print('heatmap maximum value %5.2f' % heatmap[i][0].max())
      #print('heatmap minimum value %5.2f' % heatmap[i][0].min())
      landmark_present.append(False)
      x_com = torch.tensor(0, dtype = torch.float64).to(S.device) # in theory this should not be used
      y_com = torch.tensor(0, dtype = torch.float64).to(S.device)
      z_com = torch.tensor(0, dtype = torch.float64).to(S.device)
    else:
      landmark_present.append(True)
      x_com = torch.tensor(0, dtype = torch.float64).to(S.device)
      y_com = torch.tensor(0, dtype = torch.float64).to(S.device)
      z_com = torch.tensor(0, dtype = torch.float64).to(S.device)
      for k in range(locations.size(0)): # number of landmarks
        x_com += locations[k][1]
        y_com += locations[k][0]
        z_com += locations[k][2]
      x_com /= locations.size(0)
      y_com /= locations.size(0)
      z_com /= locations.size(0)


    if i == 0:
      coords = torch.tensor([[x_com,y_com,z_com]]).to(S.device)
    else: 
      coords_temp = torch.tensor([[x_com,y_com,z_com]]).to(S.device)
      coords = torch.cat((coords,coords_temp),dim = 0)
  # returns arrays of B x coord
  # returns array of B x True/False
  # if True then for that coord it is COM, if false then need to produce heatmap of zeros
  return coords, landmark_present   


def top_structure(heatmap, landmark): # assumes 1 channel
  # heatmap is batch of either masks or image of 
  # heatmap dim is (B x C x H x W x D)
  # output is (B x coord)
  # i.e. x,y,z of 3rd image = coords[3][0],coords[3][1], coords[3][2] only 1 channel
  batch_size = (heatmap.size()[0])
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  for i in range(batch_size):
    # heatmap shape [1, 1, 256, 256, 100] i.e. B x C x H x W x D
    # add in round here to see if picks up landmarks its missing
    locations = (torch.nonzero((torch.round(heatmap[i][0]) == landmark),as_tuple = False)).to(S.device)
    #counter = torch.tensor(1, dtype = torch.float64).to(S.device) # counter defines number of points at top per image
    if (locations.size(0) == 0): # if no landmarks detected for image in batch
      #print('no structure for %1.0f' % landmark)
      #print('heatmap maximum value %5.2f' % heatmap[i][0].max())
      #print('heatmap minimum value %5.2f' % heatmap[i][0].min())
      landmark_present.append(False)
      x_top = torch.tensor(0, dtype = torch.float64).to(S.device) # in theory this should not be used
      y_top = torch.tensor(0, dtype = torch.float64).to(S.device)
      z_top = torch.tensor(0, dtype = torch.float64).to(S.device)
    else:
      landmark_present.append(True)
      z_coords = locations[:,2]
      top_coords = locations[torch.max(z_coords, dim=0)[1].item()]
      x_top = top_coords[1]
      y_top = top_coords[0]
      z_top = top_coords[2]
       
    if i == 0:
      coords = torch.tensor([[x_top,y_top,z_top]]).to(S.device)
    else: 
      coords_temp = torch.tensor([[x_top,y_top,z_top]]).to(S.device)
      coords = torch.cat((coords,coords_temp),dim = 0)
  # returns arrays of B x coord
  # returns array of B x True/False
  # if True then for that coord it is TOP, if false then need to produce heatmap of zeros
  return coords, landmark_present  

def bot_structure(heatmap, landmark): # assumes 1 channel
  # heatmap is batch of either masks or image of 
  # heatmap dim is (B x C x H x W x D)
  # output is (B x coord)
  # i.e. x,y,z of 3rd image = coords[3][0],coords[3][1], coords[3][2] only 1 channel
  batch_size = (heatmap.size()[0])
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  for i in range(batch_size):
    # heatmap shape [1, 1, 256, 256, 100] i.e. B x C x H x W x D
    # add in round here to see if picks up landmarks its missing
    locations = (torch.nonzero((torch.round(heatmap[i][0]) == landmark),as_tuple = False)).to(S.device)
    #counter = torch.tensor(1, dtype = torch.float64).to(S.device) # counter defines number of points at top per image
    if (locations.size(0) == 0): # if no landmarks detected for image in batch
      #print('no structure for %1.0f' % landmark)
      #print('heatmap maximum value %5.2f' % heatmap[i][0].max())
      #print('heatmap minimum value %5.2f' % heatmap[i][0].min())
      landmark_present.append(False)
      x_bot = torch.tensor(0, dtype = torch.float64).to(S.device) # in theory this should not be used
      y_bot = torch.tensor(0, dtype = torch.float64).to(S.device)
      z_bot = torch.tensor(0, dtype = torch.float64).to(S.device)
    else:
      landmark_present.append(True)
      z_coords = locations[:,2]
      bot_coords = locations[torch.min(z_coords, dim=0)[1].item()]
      x_bot = bot_coords[1]
      y_bot = bot_coords[0]
      z_bot = bot_coords[2]
       
    if i == 0:
      coords = torch.tensor([[x_bot,y_bot,z_bot]]).to(S.device)
    else: 
      coords_temp = torch.tensor([[x_bot,y_bot,z_bot]]).to(S.device)
      coords = torch.cat((coords,coords_temp),dim = 0)
  # returns arrays of B x coord
  # returns array of B x True/False
  # if True then for that coord it is TOP, if false then need to produce heatmap of zeros
  return coords, landmark_present


def landmarks_coords(heatmap, landmark):
  # heatmap is batch of either masks or image of 
  # heatmap dim is (B x C x H x W x D)
  # output is (B x coords)
  # i.e. first (x,y,z) of 3rd image = coords[3][0]
  # i.e. second (x,y,z) of 3rd image = coords[3][1]
  batch_size = (heatmap.size()[0])
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  for i in range(batch_size):
    locations = (heatmap[i][0] == landmark).nonzero()
    if (locations.size(0) == 0): # if no landmarks detected for image in batch
      #print('no structure for %1.0f' % landmark)
      #print('heatmap maximum value %5.2f' % heatmap[i][0].max())
      landmark_present.append(False)
      x_com = torch.tensor(0, dtype = torch.float64).to(S.device) # in theory this should not be used
      y_com = torch.tensor(0, dtype = torch.float64).to(S.device)
      z_com = torch.tensor(0, dtype = torch.float64).to(S.device)
    else:
      landmark_present.append(True)
      # --- rewrite ---
      locations = locations.transpose(1,0,2) # not sure if this will work but need to change so have x,y,z
      if i == 0:
        coords = torch.tensor(locations).to(S.device) # coords should be array type object of x,y,z for each i (== image)
      else:
        coords_temp = torch.tensor(locations).to(S.device)
        coords = torch.cat((coords,coords_temp),dim = 0) # coords by the end will be B x coords_array_per_image
  # returns array of B x True/False
  # if True then has x,y,z of all posn of that landmark, if False then need to produce heatmap of zeros
  return coords, landmark_present   
 
def point_to_point_mm_old(mask_x, mask_y, mask_z, pred_x, pred_y, pred_z, patient):
  # calculates point to point in mm
  data = csv.reader(open(os.path.join(S.root, 'image_dimensions.csv')),delimiter=',')
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
  print('orig pixel x, y, z')
  print(pixel_mm_x, pixel_mm_y, pixel_mm_z)
  if len(S.downsample_ratio_list):
        # array not emtpy
        index = 0       
        if patient in S.downsample_ratio_list:
        #index = S.downsample_idx_list.index(patient)
        #print('patient')
        #print(patient)
        #print('pre amendment pixel sizes', pixel_mm_x, pixel_mm_y, pixel_mm_z)
            pixel_mm_x = float(pixel_mm_x) * S.downsample_ratio_list[patient]['w']
            pixel_mm_y = float(pixel_mm_y) * S.downsample_ratio_list[patient]['h']
            pixel_mm_z = float(pixel_mm_z) * S.downsample_ratio_list[patient]['d']
            pixel_mm_x = torch.tensor((pixel_mm_x)).to(S.device)
            pixel_mm_y = torch.tensor((pixel_mm_y)).to(S.device)
            pixel_mm_z = torch.tensor((pixel_mm_z)).to(S.device)
        #print('downsample ratio x,y,z')
        #print(S.downsample_ratio_w[index], S.downsample_ratio_h[index], S.downsample_ratio_d[index])
        #print('post amendment pixel sizes', pixel_mm_x, pixel_mm_y, pixel_mm_z)
  else:
      pixel_mm_x = torch.tensor(float(pixel_mm_x)).to(S.device)
      pixel_mm_y = torch.tensor(float(pixel_mm_y)).to(S.device)
      pixel_mm_z = torch.tensor(float(pixel_mm_z)).to(S.device)
  point_to_point = (((pred_x - mask_x)*pixel_mm_x)**2 + ((pred_y - mask_y)*pixel_mm_y)**2 + ((pred_z - mask_z)*pixel_mm_z)**2)**0.5 
  return point_to_point
  
# new gaussian map function produces gaussian target based on every location of landmark in target
def gaussian_map_expansive(landmarks, landmarks_size, sigma, gamma, x_size, y_size, z_size, dimension =3):
  # x/y/z_landmarks gives multiple locations of landmark
  y,x,z = np.ogrid[0:y_size,0:x_size, 0:z_size]
  pre_factor = ((gamma) * (2*np.pi)**(-dimension/2) * sigma ** (-dimension)) 
  h = torch.tensor().to(S.device) # not sure what size it is can work out from gaussian_map
  for i in range(landmarks_size):
    peak_x, peak_y, peak_z = landmarks[i][0], landmarks[i][1], landmarks[i][2]
    h += pre_factor * torch.exp( -((torch.tensor(x).to(S.device)-peak_x)**2 + (torch.tensor(y).to(S.device)-peak_y)**2 + (torch.tensor(z).to(S.device)-peak_z)**2) / (2.*sigma*sigma) )
  return h

"""
