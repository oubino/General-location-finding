# evaluate model functions
import torch
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.cm as cm
from collections import defaultdict
import numpy as np
import os
import csv
import time

from data_loading import data_loaders
import settings as S
from useful_functs import functions


def performance_metrics(model,sigmas,gamma, epochs_completed, fold): 
    
  # so can print out test ids for each fold at end
  S.k_fold_ids.append(data_loaders.test_set_ids)
  
  # create directory for this eval
  epochs_completed_string = str(epochs_completed)
  file_name = "eval_" + epochs_completed_string + functions.string(fold)
  eval_path = os.path.join(S.run_path, file_name) # directory labelled with epochs_completed
  try: 
      os.mkdir(eval_path)
  except OSError as error:
      print(error)
      
  p2p_landmarks = defaultdict(float)
  outliers_landmarks = defaultdict(float)
  x_axis_err, x_axis_err_mm = defaultdict(float), defaultdict(float)
  y_axis_err, y_axis_err_mm = defaultdict(float), defaultdict(float)
  z_axis_err, z_axis_err_mm = defaultdict(float), defaultdict(float)
  for l in S.landmarks:
    p2p_landmarks[l] = np.empty((0), float)
    outliers_landmarks[l] = np.empty((0), float)
    x_axis_err[l], x_axis_err_mm[l] = np.empty((0), float),  np.empty((0), float)
    y_axis_err[l], y_axis_err_mm[l] = np.empty((0), float),  np.empty((0), float)
    z_axis_err[l], z_axis_err_mm[l] = np.empty((0), float),  np.empty((0), float)
    
  # load in struc_coord  
  struc_coord = functions.load_obj_pickle(S.root, 'coords_' + S.clicker) 
  
  # initiate max val as 0 for all patients - sliding window stuff
  # patients needs to be = ['0003.npy', '0004.npy', etc.]
  patients = data_loaders.test_set_ids
  val_max_list = {}
  coord_list = {}
  pat_index = {}
  for p in patients:
      val_max_list[p] = {}
      coord_list[p] = {}
      pat_index[p] = {}
      for l in S.landmarks:
          val_max_list[p][l] = 0
          coord_list[p][l] = {'x':0, 'y':0, 'z':0}
          pat_index[p][l] = 0
  
  for slide_index in range(S.sliding_points):
    
      for batch in data_loaders.dataloaders['test']:        
        patient = batch['patient']    
        image = batch['image'].to(S.device)        
        pred = model(image)
            
      
        #batch_number = 0
        
        for l in S.landmarks: # cycle over all landmarks
          
          for i in range(image.size()[0]): # batch size
            
            #struc_loc = struc_coord[patient[i]]
    
            #if struc_loc[l]['present'] == 1:
            # change to top structure
              dimension = 3
              height_guess = ((gamma) * (2*np.pi)**(-dimension/2) * sigmas[l].item() ** (-dimension)) 
              
              if S.pred_max == True:
                  pred_coords_max, val_max = functions.pred_max(pred, l, S.landmarks)[0], functions.pred_max(pred, l, S.landmarks)[1] # change to gauss fit
              else:
                  pred_coords_max = functions.gauss_max(pred,l,height_guess,sigmas[l].item(), S.in_x, S.in_y, S.in_z, S.landmarks)  
            
              # if max value is greatest for this patient then save the predicted coord for this landmark
              if val_max[i] > val_max_list[patient[i]][l]:
                  val_max_list[patient[i]][l] = val_max[i] # update max val
                  coord_list[patient[i]][l]['x'], coord_list[patient[i]][l]['y'], coord_list[patient[i]][l]['z'] = pred_coords_max[i][0], pred_coords_max[i][1], pred_coords_max[i][2]                  
                  pat_index[patient[i]][l] = slide_index
              
      S.slide_index += 1
  S.slide_index = 0
  
  # final locations dict
  final_loc = {}
  for p in patients:
      final_loc[p] ={}
      for l in S.landmarks:
          final_loc[p][l]= {'x':0, 'y':0, 'z':0}

  for p in patients:
      
      for l in S.landmarks:
            
              pred_max_x, pred_max_y, pred_max_z =  coord_list[p][l]['x'], coord_list[p][l]['y'], coord_list[p][l]['z']
              
              # convert pred to location in orig img
              pred_max_x, pred_max_y, pred_max_z = functions.aug_to_orig(pred_max_x, pred_max_y, pred_max_z, S.downsample_user, p, pat_index[p][l])
              
              # final location add
              final_loc[p][l]['x'], final_loc[p][l]['y'], final_loc[p][l]['z'] = pred_max_x, pred_max_y, pred_max_z
              
              struc_loc = struc_coord[p]
              
              if struc_loc[l]['present'] == 1:
                  
                  structure_max_x, structure_max_y, structure_max_z = struc_loc[l]['x'],struc_loc[l]['y'], struc_loc[l]['z']      
                      
              
                  # print out 3D images for first one in batch
                  #if batch_number == 0 and i == 0: # for first batch 
                  # now need to choose first in batch i.e. # image[0]
                  #print('3D plots for landmark %1.0f' % l)
                  #print_3D_heatmap(image[i], structure[i], pred[i], l, eval_path, patient[i])
                  #print_3D_gauss_heatmap(image[i], structure_max_x, structure_max_y, structure_max_z, pred[i], l, sigmas[l], eval_path, patient[i])
                  #print('\n')
                  #print('Structure LOC for landmark %1.0f:' % l)
                    #print(structure_max_x, structure_max_y, structure_max_z)
                    #print('Predicted LOC for landmark %1.0f:' % l)
                    #print(pred_max_x, pred_max_y, pred_max_z)
                    #print('\n')
                    # print 2D slice
                    #print('2D slice for landmark %1.0f' % l)
                  print_2D_slice(l, pred_max_x, pred_max_y, pred_max_z, structure_max_x, structure_max_y, structure_max_z ,eval_path, p)
                          
                  # point to point takes in original structure location!!
                  img_landmark_point_to_point = functions.point_to_point_mm(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z, p)
                  p2p_landmarks[l] = np.append(p2p_landmarks[l],img_landmark_point_to_point.cpu())
                  x_p2p, x_p2p_mm, y_p2p, y_p2p_mm, z_p2p, z_p2p_mm = functions.axis_p2p_err(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z, p)
                  x_axis_err[l] = np.append(x_axis_err[l], x_p2p.cpu())
                  x_axis_err_mm[l] = np.append(x_axis_err_mm[l], x_p2p_mm.cpu())
                  y_axis_err[l] = np.append(y_axis_err[l], y_p2p.cpu())
                  y_axis_err_mm[l] = np.append(y_axis_err_mm[l], y_p2p_mm.cpu())
                  z_axis_err[l] = np.append(z_axis_err[l], z_p2p.cpu())
                  z_axis_err_mm[l] = np.append(z_axis_err_mm[l], z_p2p_mm.cpu())
                  
                 # x_axis_err[l] = np.append(x_axis_err[l], )
                  # if img_point_to_point > 20mm is an outlier
                  if img_landmark_point_to_point > 10:
                    outliers_landmarks[l] = np.append(outliers_landmarks[l],1)
                
        #batch_number += 1 # not sure where to put
    
  print('\n')
  print('Results summary')    
  print('---------------')
  
  latex_line = []
  csv_line = []
  if S.rts == True:
        name_of_file = os.path.join(eval_path, S.clicker + "results_rts_final.txt")
  elif S.rts == False:
        name_of_file = os.path.join(eval_path, S.clicker + "results_new.txt")
  txt_file = open(name_of_file, "a")
  
  for l in S.landmarks:
    print('\n')
    print('Landmark %1.0f' % l)
    mean = np.mean(p2p_landmarks[l])
    std_mean = np.std(p2p_landmarks[l],ddof =1)*(len(p2p_landmarks[l]))**-0.5
    median = np.median(p2p_landmarks[l])
    outliers_perc = outliers_landmarks[l].sum()/len(p2p_landmarks[l]) * 100
    upper_perc = np.percentile(p2p_landmarks[l], 75)
    lower_perc = np.percentile(p2p_landmarks[l], 25)
    error_min = np.amin(p2p_landmarks[l])
    error_max = np.amax(p2p_landmarks[l])
    mean_x_err = np.mean(x_axis_err[l])
    mean_x_err_mm = np.mean(x_axis_err_mm[l])
    mean_y_err = np.mean(y_axis_err[l])
    mean_y_err_mm = np.mean(y_axis_err_mm[l])
    mean_z_err = np.mean(z_axis_err[l])
    mean_z_err_mm = np.mean(z_axis_err_mm[l])
    
    print('    mean point to point error is ' + str(mean) + '+/-' + str(std_mean))
    print('    median point to point error is ' + str(median))
    print('    75th percentile is: ' + str(upper_perc))
    print('    25th percentile is  ' + str(lower_perc))
    print('    minimum point to point error is: ' + str(error_min))
    print('    maximum point to point error is: ' + str(error_max))
    print('    mean error in x axis is: ' + str(mean_x_err) + ' (' + str(mean_x_err_mm) + ' mm)')
    print('    mean error in y axis is: ' + str(mean_y_err) + ' (' + str(mean_y_err_mm) + ' mm)')
    print('    mean error in z axis is: ' + str(mean_z_err) + ' (' + str(mean_z_err_mm) + ' mm)')
    print('    percentage of images which were outliers is ' + str(outliers_perc) + '%')
    print('    sigma is ' + str(sigmas[l]))
    print('    trained for ' + str(epochs_completed) + ' epochs')
    print('    pred max used = %s' % S.pred_max)
    print('\n')
  
    L = ['\n','Landmark %1.0f' % l, '\n', 
         '  mean point to point error is ' + str(mean) + '+/-' + str(std_mean), '\n',
         '  median point to point error is ' + str(median), '\n', 
         '  75th percentile is: ' + str(upper_perc), '\n',
         '  25th percentile is  ' + str(lower_perc), '\n',
         '  minimum point to point error is: ' + str(error_min), '\n',
         '  maximum point to point error is: ' + str(error_max), '\n',
         '  mean error in x axis is: ' + str(mean_x_err) + ' (' + str(mean_x_err_mm) + ' mm)', '\n',
         '  mean error in y axis is: ' + str(mean_y_err) + ' (' + str(mean_y_err_mm) + ' mm)', '\n',
         '  mean error in z axis is: ' + str(mean_z_err) + ' (' + str(mean_z_err_mm) + ' mm)', '\n',
         '  percentage of images which were outliers is ' + str(outliers_perc) + '%', '\n',
         '  sigma is ' + str(sigmas[l]), '\n', 
         '  pred max used = ' + str(S.pred_max), '\n',
         '  trained for ' + str(epochs_completed) + ' epochs\n']
    txt_file.writelines(L)
       
    # write in latex table format
    latex_line_temp = [' & ' + str(round(mean,1)) + '$\pm$' + str(round(std_mean,1))]  
    latex_line = latex_line + latex_line_temp     
    # write in excel format for easy to calc folds 
    csv_line_temp = [str(round(mean,1)) + ',' + str(round(std_mean,1)) + ',']  
    csv_line = csv_line + csv_line_temp   
    
    # add to csv file
    csv_name = os.path.join(S.save_data_path, 'results_summary.csv')
    with open(csv_name, 'a', newline = '') as file:
        writer = csv.writer(file)
        sigma_string = str(sigmas[l])
        writer.writerow(['%s' % S.run_folder, '%s' % epochs_completed_string, 'Landmark %1.0f' % l, 
             str(mean), str(std_mean),str(median),str(outliers_perc) + '%', sigma_string.replace("\n", " "), time.strftime("%Y%m%d-%H%M%S"), 'pred max used = %s' % S.pred_max])


  # write in latex/csv form
  txt_file.writelines(latex_line)
  txt_file.writelines(['\n'])
  txt_file.writelines(csv_line)
  txt_file.close()
  
  print('final locations')
  print(final_loc)
  
  functions.save_obj_pickle(final_loc, eval_path, 'final_coords')
       

def performance_metrics_line(model,sigmas,gamma, epochs_completed, fold): 
  
  # so can print out test ids for each fold at end
  S.k_fold_ids.append(data_loaders.test_set_ids)
  
  # create directory for this eval
  epochs_completed_string = str(epochs_completed)
  file_name = "eval_" + epochs_completed_string + functions.string(fold)
  eval_path = os.path.join(S.run_path, file_name) # directory labelled with epochs_completed
  try: 
      os.mkdir(eval_path)
  except OSError as error:
      print(error)
      
  keys = ('clicker_1', 'clicker_2', 'mean')
  p2p_landmarks = {}
  outliers_landmarks = {}
  x_axis_err, x_axis_err_mm = {}, {}
  y_axis_err, y_axis_err_mm = {}, {}
  z_axis_err, z_axis_err_mm = {}, {}
  for i in keys:
      p2p_landmarks[i] = {} 
      outliers_landmarks[i] = {} 
      x_axis_err[i] = {}
      x_axis_err_mm[i] = {}
      y_axis_err[i] = {}
      y_axis_err_mm[i] = {}
      z_axis_err[i] = {}
      z_axis_err_mm[i] = {}
      
  for i in keys:
      for l in S.landmarks:
        p2p_landmarks[i][l] = np.empty((0), float)
        outliers_landmarks[i][l] = np.empty((0), float)
        x_axis_err[i][l] = np.empty((0), float)
        x_axis_err_mm[i][l] = np.empty((0), float)
        y_axis_err[i][l] = np.empty((0), float)
        y_axis_err_mm[i][l] = np.empty((0), float)
        z_axis_err[i][l] = np.empty((0), float)
        z_axis_err_mm[i][l] = np.empty((0), float)
    
  # load in struc_coord  
  if S.rts == False:
      struc_coord_clicker_1 = functions.load_obj_pickle(S.root, 'coords_' + 'Oli') 
      struc_coord_clicker_2 = functions.load_obj_pickle(S.root, 'coords_' + 'Aaron') 
  elif S.rts == True:
      struc_coord_clicker_1 = functions.load_obj_pickle(S.root, 'coords_' + 'Oli_test_set') 
      struc_coord_clicker_2 = functions.load_obj_pickle(S.root, 'coords_' + 'Aaron_test_set') 
  struc_coord_mean = functions.mean_from_clickers(struc_coord_clicker_1, struc_coord_clicker_2)
  
  struc_coord = {}
  struc_coord['clicker_1'] = struc_coord_clicker_1 
  struc_coord['clicker_2'] = struc_coord_clicker_2
  struc_coord['mean'] = struc_coord_mean
  
  # initiate max val as 0 for all patients - sliding window stuff
  # patients needs to be = ['0003.npy', '0004.npy', etc.]
  patients = data_loaders.test_set_ids
  val_max_list = {}
  coord_list = {}
  pat_index = {}
  for p in patients:
      val_max_list[p] = {}
      coord_list[p] = {}
      pat_index[p] = {}
      for l in S.landmarks:
          val_max_list[p][l] = 0
          coord_list[p][l] = {'x':0, 'y':0, 'z':0}
          pat_index[p][l] = 0
          
  for slide_index in range(S.sliding_points):
      
      for batch in data_loaders.dataloaders['test']:
        image = batch['image'].to(S.device)
        patient = batch['patient']
        pred = model(image)
        
        #batch_number = 0
        
        for l in S.landmarks: # cycle over all landmarks
          
          for i in range(image.size()[0]): # batch size
              
              dimension = 3
              height_guess = ((gamma) * (2*np.pi)**(-dimension/2) * sigmas[l].item() ** (-dimension)) 
              
              if S.pred_max == True:
                  pred_coords_max, val_max = functions.pred_max(pred, l, S.landmarks)[0], functions.pred_max(pred, l, S.landmarks)[1] # change to gauss fit
              else:
                  pred_coords_max = functions.gauss_max(pred,l,height_guess,sigmas[l].item(), S.in_x, S.in_y, S.in_z, S.landmarks)  
            
              # if max value is greatest for this patient then save the predicted coord for this landmark
              if val_max[i] > val_max_list[patient[i]][l]:
                  val_max_list[patient[i]][l] = val_max[i] # update max val
                  coord_list[patient[i]][l]['x'], coord_list[patient[i]][l]['y'], coord_list[patient[i]][l]['z'] = pred_coords_max[i][0], pred_coords_max[i][1], pred_coords_max[i][2]                  
                  pat_index[patient[i]][l] = slide_index
                  if val_max[i] > 0.7:
                      print_2D_heatmap(image[i][0], l, pred[i][l-1], coord_list[patient[i]][l]['z'], eval_path, patient[i])
         
      S.slide_index += 1
  S.slide_index = 0
    
  # final locations dict
  final_loc = {}
  for p in patients:
      final_loc[p] ={}
      for l in S.landmarks:
          final_loc[p][l]= {'x':0, 'y':0, 'z':0}
  
  for p in patients:
     
      for l in S.landmarks: # cycle over all landmarks
              
          #for i in range(image.size()[0]): # batch size
              pred_max_x, pred_max_y, pred_max_z =  coord_list[p][l]['x'], coord_list[p][l]['y'], coord_list[p][l]['z']
                
              # convert pred to location in orig img
              pred_max_x, pred_max_y, pred_max_z = functions.aug_to_orig(pred_max_x, pred_max_y, pred_max_z, S.downsample_user, p, pat_index[p][l])
              
              # final location add
              final_loc[p][l]['x'], final_loc[p][l]['y'], final_loc[p][l]['z'] = pred_max_x, pred_max_y, pred_max_z
        
              for k in keys: # clicker_1, clicker_2, and mean
                              
                    struc_loc = struc_coord[k][p]
                
                    if struc_loc[l]['present'] == True:
              
                      structure_max_x, structure_max_y, structure_max_z = struc_loc[l]['x'],struc_loc[l]['y'], struc_loc[l]['z']      
                      
                      # print out images for first one in batch
                      #if batch_number == 0 and i == 0: # for first batch 
                       # print('\n')
                       # print('Structure LOC for landmark %1.0f and clicker %s:' % (l,k))
                       # print(structure_max_x, structure_max_y, structure_max_z)
                       # print('Predicted LOC for landmark %1.0f and clicker %s:' % (l,k))
                       # print(pred_max_x, pred_max_y, pred_max_z)
                       # print('\n')
                                  
                      # point to point takes in original structure location!!
                      img_landmark_point_to_point = functions.point_to_point_mm(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z, p)
                      p2p_landmarks[k][l] = np.append(p2p_landmarks[k][l],img_landmark_point_to_point.cpu())
                      # if img_point_to_point > 20mm is an outlier
                      x_p2p, x_p2p_mm, y_p2p, y_p2p_mm, z_p2p, z_p2p_mm = functions.axis_p2p_err(structure_max_x, structure_max_y, structure_max_z, pred_max_x, pred_max_y, pred_max_z, p)
                      x_axis_err[k][l] = np.append(x_axis_err[k][l], x_p2p.cpu())
                      x_axis_err_mm[k][l] = np.append(x_axis_err_mm[k][l], x_p2p_mm.cpu())
                      y_axis_err[k][l] = np.append(y_axis_err[k][l], y_p2p.cpu())
                      y_axis_err_mm[k][l] = np.append(y_axis_err_mm[k][l], y_p2p_mm.cpu())
                      z_axis_err[k][l] = np.append(z_axis_err[k][l], z_p2p.cpu())
                      z_axis_err_mm[k][l] = np.append(z_axis_err_mm[k][l], z_p2p_mm.cpu())
                      if img_landmark_point_to_point > 20:
                          outliers_landmarks[k][l] = np.append(outliers_landmarks[k][l],1)
                  
              # print 2D slice
              print('2D slice for landmark %1.0f' % l)
              print_2D_slice_line(l, pred_max_x, pred_max_y, pred_max_z, struc_coord, eval_path, p)
        
#batch_number += 1 # not sure where to put
   
  for k in keys:
      
      print('\n')
      print('Results summary for clicker %s' % k)    
      print('---------------')
      
      latex_line = []
      csv_line = []
      if S.rts == True:
        name_of_file = os.path.join(eval_path, "results_rts_line_new_%s.txt" % k)
      elif S.rts == False:
        name_of_file = os.path.join(eval_path, "results_line_new_%s.txt" % k)
      txt_file = open(name_of_file, "a")
      
      for l in S.landmarks:
        print('\n')
        print('Landmark %1.0f' % l)
        mean = np.mean(p2p_landmarks[k][l])
        std_mean = np.std(p2p_landmarks[k][l],ddof =1)*(len(p2p_landmarks[k][l]))**-0.5
        median = np.median(p2p_landmarks[k][l])
        upper_perc = np.percentile(p2p_landmarks[k][l], 75)
        lower_perc = np.percentile(p2p_landmarks[k][l], 25)
        error_min = np.amin(p2p_landmarks[k][l])
        error_max = np.amax(p2p_landmarks[k][l])
        outliers_perc = outliers_landmarks[k][l].sum()/len(p2p_landmarks[k][l]) * 100
        mean_x_err = np.mean(x_axis_err[k][l])
        mean_x_err_mm = np.mean(x_axis_err_mm[k][l])
        mean_y_err = np.mean(y_axis_err[k][l])
        mean_y_err_mm = np.mean(y_axis_err_mm[k][l])
        mean_z_err = np.mean(z_axis_err[k][l])
        mean_z_err_mm = np.mean(z_axis_err_mm[k][l])
        
        print('    mean point to point error is ' + str(mean) + '+/-' + str(std_mean))
        print('    median point to point error is ' + str(median))
        print('    mean point to point error is ' + str(mean) + '+/-' + str(std_mean))
        print('    median point to point error is ' + str(median))
        print('    75th percentile is: ' + str(upper_perc))
        print('    25th percentile is  ' + str(lower_perc))
        print('    minimum point to point error is: ' + str(error_min))
        print('    maximum point to point error is: ' + str(error_max))
        print('    mean error in x axis is: ' + str(mean_x_err) + ' (' + str(mean_x_err_mm) + ' mm)')
        print('    mean error in y axis is: ' + str(mean_y_err) + ' (' + str(mean_y_err_mm) + ' mm)')
        print('    mean error in z axis is: ' + str(mean_z_err) + ' (' + str(mean_z_err_mm) + ' mm)')
        print('    percentage of images which were outliers is ' + str(outliers_perc) + '%')
        print('    sigma is ' + str(sigmas[l]))
        print('    trained for ' + str(epochs_completed) + ' epochs')
        print('    pred max used = %s' % S.pred_max)
        print('\n')
        
        
        
        L = ['\n','Landmark %1.0f' % l, '\n', 
             '  mean point to point error is ' + str(mean) + '+/-' + str(std_mean), '\n',
             '  median point to point error is ' + str(median), '\n', 
             '  75th percentile is: ' + str(upper_perc), '\n',
             '  25th percentile is  ' + str(lower_perc), '\n',
             '  minimum point to point error is: ' + str(error_min), '\n',
             '  maximum point to point error is: ' + str(error_max), '\n',
             '  percentage of images which were outliers is ' + str(outliers_perc) + '%', '\n',
             '  mean error in x axis is: ' + str(mean_x_err) + '(' + str(mean_x_err_mm) + 'mm)', '\n',
             '  mean error in y axis is: ' + str(mean_y_err) + '(' + str(mean_y_err_mm) + 'mm)', '\n',
             '  mean error in z axis is: ' + str(mean_z_err) + '(' + str(mean_z_err_mm) + 'mm)', '\n',
             '  sigma is ' + str(sigmas[l]), '\n', 
             '  pred max used = ' + str(S.pred_max), '\n',
             '  trained for ' + str(epochs_completed) + ' epochs\n']
        txt_file.writelines(L)
           
        # write in latex table format
        latex_line_temp = [' & ' + str(round(mean,1)) + '$\pm$' + str(round(std_mean,1))]  
        latex_line = latex_line + latex_line_temp     
        # write in excel format for easy to calc folds 
        csv_line_temp = [str(round(mean,1)) + ',' + str(round(std_mean,1)) + ',']  
        csv_line = csv_line + csv_line_temp   
        
        # add to csv file
        csv_name = os.path.join(S.save_data_path, 'results_summary.csv')
        with open(csv_name, 'a', newline = '') as file:
            writer = csv.writer(file)
            sigma_string = str(sigmas[l])
            writer.writerow(['%s' % S.run_folder, '%s' % epochs_completed_string, 'Landmark %1.0f' % l, 
                 str(mean), str(std_mean),str(median),str(outliers_perc) + '%', sigma_string.replace("\n", " "), time.strftime("%Y%m%d-%H%M%S"), 'pred max used = %s' % S.pred_max])
    
    
      # write in latex/csv form
      txt_file.writelines(latex_line)
      txt_file.writelines(['\n'])
      txt_file.writelines(csv_line)
      txt_file.close()
      
  print('final locations')
  print(final_loc)
  
  functions.save_obj_pickle(final_loc, eval_path, 'final_coords')
  

def print_2D_slice_line(landmark, pred_x, pred_y, pred_z, structure_coord, eval_path, patient):
    
    # image
    #  D x H x W
    img_path = os.path.join(S.root, "CTs", patient) 
    img = np.load(img_path)
    
    plt.figure(figsize=(7, 7))
        
    pred_z = int(pred_z) # convert to nearest int
    img = img[pred_z, :, :]
    
    struc_x_1, struc_y_1, struc_z_1 = structure_coord['clicker_1'][patient][landmark]['x'], structure_coord['clicker_1'][patient][landmark]['y'], structure_coord['clicker_1'][patient][landmark]['z']
    struc_x_2, struc_y_2, struc_z_2 = structure_coord['clicker_2'][patient][landmark]['x'], structure_coord['clicker_2'][patient][landmark]['y'], structure_coord['clicker_2'][patient][landmark]['z']

    # ---- plot as point ------
    plt.imshow(img,cmap = 'Greys_r', alpha = 0.9)
    plt.plot(struc_x_1, struc_y_1, color = 'red', marker = 'x', label = 'Oli z: %1.0f' % struc_z_1)
    plt.plot(struc_x_2, struc_y_2, color = 'blue', marker = 'x', label = 'Aaron z: %1.0f' % struc_z_2)
    plt.plot(pred_x.cpu().numpy(), pred_y.cpu().numpy(),color='green', marker='o', label = 'pred z: %1.0f' % pred_z)
    # add z annotation
    #plt.annotate("%1.0f" % pred_z,(pred_x.cpu().numpy(), pred_y.cpu().numpy()), color = 'green')
    #plt.annotate("%1.0f" % int(struc_z_1),(struc_x_1, struc_y_1), color = 'red')
    #plt.annotate("%1.0f" % int(struc_z_2),(struc_x_2, struc_y_2), color = 'blue')
    plt.legend()
    # ------------------------------------
    
    img_name = os.path.join(eval_path, "2d_slice_%s.png" % patient.replace('.npy', '_%1.0f') % landmark)
    S.img_counter_3 += 1
    plt.savefig(img_name)
    

    
def print_2D_heatmap(img, landmark, heatmap, pred_z, eval_path, patient):
    
    # image
    #  H x W x d
    
    plt.figure(figsize=(7, 7))
        
    pred_z = int(pred_z) # convert to nearest int
    img = img[:, :, pred_z]
    
    heatmap = heatmap.detach().cpu()[:,:,pred_z]
    heatmap = np.ma.masked_where(heatmap < 0.7, heatmap)
    
    # ---- plot as point ------
    plt.imshow(img.cpu(),cmap = 'Greys_r', alpha = 0.9)
    plt.imshow(heatmap, cmap = 'viridis', alpha = 0.7)
    # add z annotation
    #plt.annotate("%1.0f" % pred_z,(pred_x.cpu().numpy(), pred_y.cpu().numpy()), color = 'green')
    #plt.annotate("%1.0f" % int(struc_z_1),(struc_x_1, struc_y_1), color = 'red')
    #plt.annotate("%1.0f" % int(struc_z_2),(struc_x_2, struc_y_2), color = 'blue')
    #plt.legend()
    # ------------------------------------
    
    img_name = os.path.join(eval_path, "2d_slice_heatmap_%s.png" % patient.replace('.npy', '_%1.0f') % landmark)
    S.img_counter_3 += 1
    plt.savefig(img_name)
    
def print_2D_slice(landmark, pred_x, pred_y, pred_z, struc_x, struc_y, struc_z, eval_path, patient):
    
    # image
    #  D x H x W
    img_path = os.path.join(S.root, "CTs", patient) 
    img = np.load(img_path)
        
    plt.figure(figsize=(7, 7))
        
    pred_z = int(pred_z) # convert to nearest int
    img = img[pred_z, :, :]
    
    # ---- plot as point ------
    plt.imshow(img,cmap = 'Greys_r', alpha = 0.9)
    plt.plot(struc_x, struc_y, color = 'red', marker = 'x', label = 'target')
    plt.plot(pred_x.cpu().numpy(), pred_y.cpu().numpy(),color='green', marker='o', label = 'pred')
    # add z annotation
    plt.annotate("%1.0f" % pred_z,(pred_x.cpu().numpy(), pred_y.cpu().numpy()), color = 'green')
    plt.annotate("%1.0f" % int(struc_z),(struc_x, struc_y), color = 'red')
    plt.legend()
    # ------------------------------------
    
    img_name = os.path.join(eval_path, "2d_slice_%s.png" % patient.replace('.npy', '_%1.0f') % landmark)
    S.img_counter_3 += 1
    plt.savefig(img_name)
    


"""
  
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
def plot_3d_pred_img_struc(image, structure, pred, threshold_img, eval_path, patient, landmark):
    
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
    ax.add_collection3d(mesh_pred)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])
    ax.invert_xaxis()
    # rotate the axes and update
    ax.mouse_init(rotate_btn=1, zoom_btn=3)
    img_name = os.path.join(eval_path, patient.replace('.npy', '_%1.0f.png') % landmark)
    S.img_counter_1 += 1
    plt.savefig(img_name)
    
def plot_3d_pred_img_no_pred(image, structure, threshold_img, eval_path, patient, landmark):
    
    verts_img, faces_img = measure.marching_cubes_classic(image, threshold_img)
    verts_structure, faces_structure = measure.marching_cubes_classic(structure)#, threshold_structure)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh_img = Poly3DCollection(verts_img[faces_img], alpha=0.1)
    mesh_structure = Poly3DCollection(verts_structure[faces_structure], alpha=0.6)
    #face_color_img = [0.5, 0.5, 1]
    face_color_img = ['tab:gray']
    mesh_img.set_facecolor(face_color_img)
    
    face_color_structure = ['r', 'b', 'g']
    mesh_structure.set_facecolor(face_color_structure)
    ax.add_collection3d(mesh_img)
    ax.add_collection3d(mesh_structure)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])
    ax.invert_xaxis()
    # rotate the axes and update
    ax.mouse_init(rotate_btn=1, zoom_btn=3)
    img_name = os.path.join(eval_path, patient.replace('.npy', '_%1.0f.png') % landmark)
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
  
    #ax.add_collection3d(mesh_structure)
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
    
    
def print_3D_heatmap(image, structure, pred, landmark, eval_path, patient):
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
  threshold_structure = structure_max # unused
  #threshold_pred = threshold_pred_print # unused
  plot_3d_pred_img_struc(image, structure_1, pred, threshold_img, eval_path, patient, landmark)
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
def print_3D_gauss_heatmap(image, structure_com_x, structure_com_y, structure_com_z, pred, landmark, sigma, eval_path, patient):
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
  plot_3d_pred_img_struc(image, structure_gauss, pred, threshold_img, eval_path, patient, landmark)
"""