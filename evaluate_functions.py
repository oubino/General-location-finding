# evaluate model functions
import torch
import settings as S
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.cm as cm
import functions
from collections import defaultdict
import numpy as np
import data_loaders
import os
import csv
import time

def extract_landmark_for_structure(structure, landmark):
  landmark = float(landmark)
  zero_tensor = torch.zeros(structure.size(), dtype = torch.float).to(S.device)

  min = landmark - 0.1
  max = landmark + 0.1 # small range around int

  min = float(min)
  max = float(max)

  #print(structure.type())
  #print(zero_tensor.type())

  a = torch.where(structure > min, structure, zero_tensor)
  b = torch.where(a < max, a, zero_tensor)


  return b


# for pred/structure & image
def plot_3d_pred_img_struc(image, structure, pred, threshold_img, eval_path):
    
    verts_structure, faces_structure = measure.marching_cubes_classic(structure)#, threshold_structure)
    verts_img, faces_img = measure.marching_cubes_classic(image, threshold_img)
    verts_pred, faces_pred = measure.marching_cubes_classic(pred)#, threshold_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh_img = Poly3DCollection(verts_img[faces_img], alpha=0.1)
    mesh_structure = Poly3DCollection(verts_structure[faces_structure], alpha=0.6)
    mesh_pred = Poly3DCollection(verts_pred[faces_pred], alpha=0.2)

    #face_color_img = [0.5, 0.5, 1]
    face_color_img = ['tab:gray']
    mesh_img.set_facecolor(face_color_img)

    face_color_structure = ['r', 'b', 'g']
    mesh_structure.set_facecolor(face_color_structure)

    face_color_pred = ['y', 'm', 'c']
    mesh_pred.set_facecolor(face_color_pred)


    ax.add_collection3d(mesh_img)
    ax.add_collection3d(mesh_structure)
    #ax.add_collection3d(mesh_pred)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])

    ax.invert_xaxis()

    # rotate the axes and update
    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    img_name = os.path.join(eval_path, "%1.0f.png" % S.img_counter_1)
    S.img_counter_1 += 1
    plt.savefig(img_name)
    
def plot_3d_pred_img_struc_no_img(structure, pred, eval_path):
    
    verts_structure, faces_structure = measure.marching_cubes_classic(structure)#, threshold_structure)
    verts_pred, faces_pred = measure.marching_cubes_classic(pred)#, threshold_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
   
    mesh_structure = Poly3DCollection(verts_structure[faces_structure], alpha=0.6)
    mesh_pred = Poly3DCollection(verts_pred[faces_pred], alpha=0.2)

   

    face_color_structure = ['r', 'b', 'g']
    mesh_structure.set_facecolor(face_color_structure)

    face_color_pred = ['y', 'm', 'c']
    mesh_pred.set_facecolor(face_color_pred)


  
    ax.add_collection3d(mesh_structure)
    ax.add_collection3d(mesh_pred)

    ax.set_xlim(0, structure.shape[1])
    ax.set_ylim(0, structure.shape[0])
    ax.set_zlim(0, structure.shape[2])

    ax.invert_xaxis()

    # rotate the axes and update
    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    img_name = os.path.join(eval_path, "no_struc_%1.0f.png" % S.img_counter_2)
    S.img_counter_2 += 1
    plt.savefig(img_name)
    
def print_2D_slice(image, structure, pred, landmark, pred_z, eval_path):
    
    # image
    # - C x H x W x D needs to be cut down to H x W x D
    # structure
    # - C x H x W x D needs to be cut down to H x W x D
    # - currently has all landmarks in but need to plot only 1 landmark - l
    # pred
    # - C x H x W x D needs to be cut down to H x W x D
    # - not sure what values of this heatmap will be so not sure what threshold should be
    image = image.squeeze(0).cpu().numpy()
    index = S.landmarks.index(landmark)
    pred = pred[index].detach().cpu().numpy() # chooses relevant channel for landmark - might need to be squeezed
    structure = structure.squeeze(0)
    structure_l = extract_landmark_for_structure(structure, landmark).cpu().numpy() # edit
    structure = structure.cpu().numpy()
    
    fig = plt.figure(figsize=(7, 7))
    
    print('image and predicted heatmap')
    
    pred_z = int(pred_z) # convert to nearest int

    image = image[:, :, pred_z]
    structure_l = structure_l[:, :, pred_z]
    pred = pred[:, :, pred_z]

    plt.imshow(image,cmap = 'Greys_r', alpha = 0.5)
    #plt.imshow(structure_l, cmap = 'Reds', alpha = 0.8 )
    plt.imshow(pred, cmap = cm.jet, alpha = 0.5)
    
    print('image and structure')
    
    fig = plt.figure(figsize=(7, 7))

    plt.imshow(image,cmap = 'Greys_r', alpha = 1)
    cmap = matplotlib.colors.ListedColormap(['0','r'])
    # create a normalize object the describes the limits of
    # each color
    bounds = [0,0.5,6]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(structure_l, cmap = cmap, alpha = 0.5)

    img_name = os.path.join(eval_path, "2d_slice_%1.0f.png" % S.img_counter_3)
    S.img_counter_3 += 1
    plt.savefig(img_name)

    
    
def print_3D_heatmap(image, structure, pred, landmark, eval_path):
  # image
  # - C x H x W x D needs to be cut down to H x W x D
  # structure
  # - C x H x W x D needs to be cut down to H x W x D
  # - currently has all landmarks in but need to plot only 1 landmark - l
  # pred
  # - C x H x W x D needs to be cut down to H x W x D
  # -t not sure what values of this heatmap will be so not sure what threshold should be

  structure_max = torch.max(structure).item()

  image = image.squeeze(0).cpu().numpy()
  index = S.landmarks.index(landmark)
  pred = pred[index].detach().cpu().numpy() # chooses relevant channel for landmark - might need to be squeezed
  structure = structure.squeeze(0)
  structure_1 = extract_landmark_for_structure(structure, landmark).cpu().numpy() # edit
  structure = structure.cpu().numpy()

  threshold_img = S.threshold_img_print
  #threshold_structure = structure_max # unused
  #threshold_pred = threshold_pred_print # unused


  plot_3d_pred_img_struc(image, structure_1, pred, threshold_img, eval_path)

def print_3D_heatmap_no_img(structure, pred, landmark):
  # - C x H x W x D needs to be cut down to H x W x D
  # - currently has all landmarks in but need to plot only 1 landmark - l
  # pred
  # - C x H x W x D needs to be cut down to H x W x D
  # - not sure what values of this heatmap will be so not sure what threshold should be

  structure_max = torch.max(structure).item()
    
  index = S.landmarks.index(landmark)
  pred = pred[index].detach().cpu().numpy() # chooses relevant channel for landmark - might need to be squeezed
  structure = structure.squeeze(0)
  structure_1 = extract_landmark_for_structure(structure, landmark).cpu().numpy() # edit
  structure = structure.cpu().numpy()

  #threshold_structure = structure_max # unused
  #threshold_pred = threshold_pred_print # unused


  plot_3d_pred_img_struc_no_img(structure_1, pred)

# print structure as 3D gauss and then print prediction heatmap and the max of it 
def print_3D_gauss_heatmap(image, structure_com_x, structure_com_y, structure_com_z, pred, landmark, sigma, eval_path):
  # image
  # - C x H x W x D needs to be cut down to H x W x D
  # structure_com
  # - use gaussian_map to make gaussian map from structure
  # - this needs to be only for the desired landmark
  # pred
  # - print

  image = image.squeeze(0).cpu().numpy()
  x_size = image.shape[1]
  y_size = image.shape[0]
  z_size = image.shape[2]
  structure_gauss = functions.gaussian_map(structure_com_x,structure_com_y,structure_com_z, sigma,S.gamma,x_size,y_size, z_size, output = True,dimension = 3).detach().cpu()
  index = S.landmarks.index(landmark)
  pred = pred[index].detach().cpu().numpy() # chooses relevant channel for landmark - might need to be squeezed


  threshold_img = S.threshold_img_print
  #threshold_structure = landmark # unuused
  #threshold_pred = threshold_pred_print # unused

  plot_3d_pred_img_struc(image, structure_gauss, pred, threshold_img, eval_path)



def performance_metrics(model,sigmas,gamma, epochs_completed):
  
  # create directory for this eval
  epochs_completed_string = str(epochs_completed)
  file_name = "eval_" + epochs_completed_string
  eval_path = os.path.join(S.run_path, file_name) # directory labelled with epochs_completed
  try: 
      os.mkdir(eval_path)
  except OSError as error:
      print(error)
      
  p2p_landmarks = defaultdict(float)
  outliers_landmarks = defaultdict(float)
  for l in S.landmarks:
    p2p_landmarks[l] = np.empty((0), float)
    outliers_landmarks[l] = np.empty((0), float)


  for batch in data_loaders.dataloaders['test']:
    image = batch['image'].to(S.device)
    structure = batch['structure'].to(S.device)
    idx = batch['idx'].to(S.device)
    pred = model(image)
  
    batch_number = 0
    
    for l in S.landmarks: # cycle over all landmarks
      
      for i in range(S.batch_size):
        
        structure_loc = functions.landmark_loc(S.landmarks_loc[l], structure, l)[0]
        #structure_com = functions.com_structure(structure, l)[0]# [0] ensures extracts coords rather than True/False
        # change to top structure
        #if functions.com_structure(structure,1)[1][i] == True:
        if functions.landmark_loc(S.landmarks_loc[l],structure,l)[1][i] == True:
        # change to top structure
          dimension = 3
          height_guess = ((gamma) * (2*np.pi)**(-dimension/2) * sigmas[l].item() ** (-dimension)) 
          
          if S.pred_max == True:
              pred_coords_max = functions.pred_max(pred, l, S.landmarks) # change to gauss fit
          else:
              pred_coords_max = functions.gauss_max(pred,l,height_guess,sigmas[l].item(), S.in_x, S.in_y, S.in_z, S.landmarks)  

          #print(pred.shape)
          
          structure_max_x, structure_max_y, structure_max_z = structure_loc[i][0],structure_loc[i][1], structure_loc[i][2] 
          pred_max_x, pred_max_y, pred_max_z =  pred_coords_max[i][0], pred_coords_max[i][1], pred_coords_max[i][2] 


          # print out 3D images for first one in batch
          if batch_number == 0 and i == 0: # for first batch 
            # now need to choose first in batch i.e. # image[0]
            #print('3D plots for landmark %1.0f' % l)
            print_3D_heatmap(image[i], structure[i], pred[i], l, eval_path)
            print_3D_gauss_heatmap(image[i], structure_max_x, structure_max_y, structure_max_z, pred[i], l, sigmas[l], eval_path)
            print('\n')
            print('Structure LOC for landmark %1.0f:' % l)
            print(structure_max_x, structure_max_y, structure_max_z)
            print('Predicted LOC for landmark %1.0f:' % l)
            print(pred_max_x, pred_max_y, pred_max_z)
            print('\n')
            # print 2D slice
            #print('2D slice for landmark %1.0f' % l)
            #print_2D_slice(image[i], structure[i], pred[i], l, pred_max_z, eval_path)
            

          #img_landmark_point_to_point = point_to_point(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z)
          img_landmark_point_to_point = functions.point_to_point_mm(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z, idx[i].item())
          p2p_landmarks[l] = np.append(p2p_landmarks[l],img_landmark_point_to_point.cpu())
          # if img_point_to_point > 20mm is an outlier
          if img_landmark_point_to_point > 20:
            outliers_landmarks[l] = np.append(outliers_landmarks[l],1)

    batch_number += 1 # not sure where to put
    
  print('\n')
  print('Results summary')    
  print('---------------')
  
  for l in S.landmarks:
    print('\n')
    print('Landmark %1.0f' % l)
    mean = np.mean(p2p_landmarks[l])
    std_mean = np.std(p2p_landmarks[l],ddof =1)*(len(p2p_landmarks[l]))**-0.5
    median = np.median(p2p_landmarks[l])
    outliers_perc = outliers_landmarks[l].sum()/len(p2p_landmarks[l]) * 100
    print('    mean point to point error is ' + str(mean) + '+/-' + str(std_mean))
    print('    median point to point error is ' + str(median))
    print('    percentage of images which were outliers is ' + str(outliers_perc) + '%')
    print('    sigma is ' + str(sigmas[l]))
    print('    trained for ' + str(epochs_completed) + ' epochs')
    print('    pred max used = %s' % S.pred_max)
    print('\n')
  
    name_of_file = os.path.join(eval_path, "results.txt")
    file = open(name_of_file, "a")
    L = ['\n','Landmark %1.0f' % l, '\n', 
         '  mean point to point error is ' + str(mean) + '+/-' + str(std_mean), '\n',
         '  median point to point error is ' + str(median), '\n', 
         '  percentage of images which were outliers is ' + str(outliers_perc) + '%', '\n',
         '  sigma is ' + str(sigmas[l]), '\n', 
         '  pred max used = ' + str(S.pred_max), '\n',
         '  trained for ' + str(epochs_completed) + ' epochs\n']
    file.writelines(L)
    file.close()
    
    # add to csv file
    csv_name = os.path.join(S.save_data_path, 'results_summary.csv')
    with open(csv_name, 'a', newline = '') as file:
        writer = csv.writer(file)
        sigma_string = str(sigmas[l])
        writer.writerow(['%s' % S.run_folder, '%s' % epochs_completed_string, 'Landmark %1.0f' % l, 
             str(mean), str(std_mean),str(median),str(outliers_perc) + '%', sigma_string.replace("\n", " "), time.strftime("%Y%m%d-%H%M%S"), 'pred max used = %s' % S.pred_max])
    