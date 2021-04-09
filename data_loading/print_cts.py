# print CTs


import os
import matplotlib.pyplot as plt
import sys


# change working directory
coding_path = r'/home/oliver_umney/GitHub/General-location-finding'
#coding_path = r'/home/rankinaaron98/General-location-finding'
#coding_path = r'C:\Users\olive\OneDrive\Documents\GitHub\General-location-finding'
#os.chdir(coding_path)
#print(os.getcwd())
sys.path.append(coding_path)

import settings as S
S.init()
S.init_new()
from data_loading import data_loaders
from evaluate import evaluate_functions as eval_func
from useful_functs import functions


def print_2D_slice_check(image, structure, landmark, struc_x, struc_y, struc_z, print_path_ct, patient):
    
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
    
        structure = structure.squeeze(0)
        structure_l = eval_func.extract_landmark_for_structure(structure, landmark).cpu().numpy() # edit
        structure = structure.cpu().numpy()
    
        fig = plt.figure(figsize=(7, 7))
    
    #print('image and predicted heatmap')
    
   # print(struc_z)
        struc_z = int(round(struc_z.item()))
        image = image[:, :, struc_z]
        structure_l = structure_l[:, :, struc_z]

    # ---- if want to plot as point ------
        plt.imshow(image,cmap = 'Greys_r', alpha = 0.9)
        plt.plot(struc_x.cpu().numpy(), struc_y.cpu().numpy(), color = 'red', marker = 'x', label = 'target')
  
    # add z annotation
        plt.annotate("%1.0f" % struc_z,(struc_x.cpu().numpy(), struc_y.cpu().numpy()), color = 'red')
        plt.legend()
    # ------------------------------------
    
        img_name = os.path.join(print_path_ct, "2d_slice_%s.png" % patient.replace('.npy', '_%1.0f') % landmark)
        plt.savefig(img_name)

# print CTs

data_loaders.dataset.__train__()
data_loaders.init_reserved_test_set()
file_name_print = "print_cts"
path_print_ct = os.path.join(S.run_path, file_name_print)
try: 
    os.mkdir(path_print_ct)
except OSError as error:
    print(error)
print(len(data_loaders.dataset))

for batch in data_loaders.dataloaders['test']:
    # print dataloader 
    inputs = batch['image'].to(S.device)
    labels = batch['structure'].to(S.device)
    idx = batch['idx']  
    patient = batch['patient']
    print('Patient')
    print(patient)

    for landmark in S.landmarks:
        #patient  = dataset.__getitem__(i)['patient'].squeeze(0)
        structure_loc = functions.landmark_loc(labels, landmark)[0]
        structure_max_x, structure_max_y, structure_max_z = structure_loc[0][0],structure_loc[0][1], structure_loc[0][2] 
        print('landmark, x, y, z')
        print(landmark, structure_max_x, structure_max_y, structure_max_z)
        #structure_extrac = eval_func.extract_landmark_for_structure_np(structure, landmark)
        #eval_func.plot_3d_pred_img_no_pred(inputs[0].squeeze(0).cpu().numpy(), empty_struc, S.threshold_img_print, path_print_ct, patient[0], landmark)
        #print(dataset.__getitem__(i)['patient'])
        print_2D_slice_check(inputs[0], labels[0], landmark, structure_max_x, structure_max_y, structure_max_z, path_print_ct, patient[0])
        
     
'''
# print single image
for i in range(len(dataset)):
    print(dataset.__getitem__(i)['patient'])
    if dataset.__getitem__(i)['patient'] == S.ct_print:
        print('--- printing ---')
        for landmark in S.landmarks:
            structure = dataset.__getitem__(i)['structure'].squeeze(0)
            locations = np.nonzero(np.round(structure) == landmark)
            try:
                x, y, z = locations[0][1], locations[0][0], locations[0][2]
            except IndexError as error:
                    x = 0
                    y = 0
                    z = 0
            empty_struc = np.zeros((128,128,80))
            empty_struc[y][x][z] = landmark
            #structure_extrac = eval_func.extract_landmark_for_structure_np(structure, landmark)
            eval_func.plot_3d_pred_img_no_pred(dataset.__getitem__(i)['image'].squeeze(0).cpu().numpy(), empty_struc, S.threshold_img_print, path_print_ct, dataset.__getitem__(i)['patient'], landmark)
        print(dataset.__getitem__(i)['patient'])
'''