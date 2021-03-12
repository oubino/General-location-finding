# dataloaders

import transformations as T
import dataset_class as D
from torchvision import transforms #, datasets, models
import torch
from torch.utils.data import DataLoader
import math
import os
import numpy
import evaluate_functions as eval_func
import numpy as np


import settings as S
import functions 

os.chdir(S.root) # change to data path and change back at end
#print(os.getcwd())

if S.downsample_user == True:
    trans_plain = transforms.Compose([T.Resize(S.in_z,S.in_x,S.in_y),T.Upsidedown_scipy(), T.Extract_landmark_location(), T.Fix_base_value(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Check_left_right(), T.ToTensor()])
    trans_augment = transforms.Compose([T.Resize(S.in_z,S.in_x,S.in_y),T.Upsidedown_scipy(), T.Extract_landmark_location(), T.Fix_base_value(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Flips_scipy(), T.Horizontal_flip(), T.Check_left_right(), T.ToTensor()])
#trans_plain = transforms.Compose([Normalise(normal_min, normal_max, normal_window), Depth(100,256,256), Upsidedown_scipy(), ToTensor()])
#trans_augment = transforms.Compose([Normalise(normal_min, normal_max, normal_window),Depth(100,256,256), Upsidedown_scipy(), Flips_scipy(), ToTensor()])#, Pre_Transpose(), Check_com_present('bob'), Post_Transpose()])#,  Upsidedown(),Flips(), CentreCrop(), Affine(), Post_Transpose()])#HorizontalFlip()Noise(),, Affine() CentreCrop()CentreCrop(), Flips(), Affine(),
# torch image: C X H X W x D -> this is output and what we deal with from now on
# Upsidedown(), Flips(), CentreCrop(), Affine()
elif S.downsample_user == False:
    trans_plain = transforms.Compose([T.Upsidedown_scipy(),T.Extract_landmark_location(), T.Fix_base_value(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.CentreCrop(S.in_z,S.in_x,S.in_y), T.ToTensor()])
    trans_augment = transforms.Compose([T.Upsidedown_scipy(), T.Extract_landmark_location(), T.Fix_base_value(), T.Normalise(S.normal_min, S.normal_max, S.normal_window),T.CentreCrop(S.in_z,S.in_x,S.in_y), T.Flips_scipy(), T.Horizontal_flip(), T.ToTensor()])
    #trans_plain = transforms.Compose([Normalise(normal_min, normal_max, normal_window), Depth(100,256,256), Upsidedown_scipy(), ToTensor()])
    #trans_augment = transforms.Compose([Normalise(normal_min, normal_max, normal_window),Depth(100,256,256), Upsidedown_scipy(), Flips_scipy(), ToTensor()])#, Pre_Transpose(), Check_com_present('bob'), Post_Transpose()])#,  Upsidedown(),Flips(), CentreCrop(), Affine(), Post_Transpose()])#HorizontalFlip()Noise(),, Affine() CentreCrop()CentreCrop(), Flips(), Affine(),
    # torch image: C X H X W x D -> this is output and what we deal with from now on
    # Upsidedown(), Flips(), CentreCrop(), Affine()


dataset = D.CTDataset(S.root, transform_train = trans_augment, transform_test = trans_plain, test = False )

# split data in train/val/test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(0))
# manual seed ensures that same split everytime to ensure testing on correct dataset!
# i.e. if random split, train, save, random split, test -> may end up testing on same as training!


image_datasets = {
    'train': train_set, 'val': val_set, 'test':test_set
}

# Load data in
dataloaders = {
    'train': DataLoader(train_set, batch_size=S.batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=S.batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set,batch_size = S.batch_size, shuffle = False, num_workers=0)
}

# check dataloaders working
#print(train_set.__getitem__(0)['image'].size()) # i.e. 1 x 224 x 224 as torch tensor (C x H x W)
#print(train_set.__getitem__(0)['structure'].size()) 


print('Example training image')
print('----------------------')
for i in range(1):
    print('image size: ')
    print(train_set.__getitem__(i)['structure'].size()) # i.e. 1 x 224 x 224 as torch tensor (C x H x W)

    
    print('landmark locations for image %1.0f in dataset' % i)
    for l in S.landmarks:
        print(functions.landmark_loc(train_set.__getitem__(i)['structure'].unsqueeze(0),l))


        
# print all images as CT scans to view them
if S.print_CT_check == True:
    
    # training set
    file_name_train = "train_cts"
    train_path_ct = os.path.join(S.run_path, file_name_train)
    try: 
        os.mkdir(train_path_ct)
    except OSError as error:
        print(error)
    
    for i in range(len(train_set)):
        for landmark in S.landmarks:
            structure = train_set.__getitem__(i)['structure'].squeeze(0)
            locations = np.nonzero(np.round(structure) == landmark)
            x, y, z = locations[0][1], locations[0][0], locations[0][2]
            empty_struc = np.zeros((128,128,80))
            empty_struc[y][x][z] = landmark
            #structure_extrac = eval_func.extract_landmark_for_structure_np(structure, landmark)
            eval_func.plot_3d_pred_img_no_pred(train_set.__getitem__(i)['image'].squeeze(0).cpu().numpy(), empty_struc, S.threshold_img_print, train_path_ct, train_set.__getitem__(i)['patient'], landmark)
        print(train_set.__getitem__(i)['patient'])
    
    
    
    # val set
    file_name_val = "val_cts"
    val_path_ct = os.path.join(S.run_path, file_name_val)
    try: 
        os.mkdir(val_path_ct)
    except OSError as error:
        print(error)
    
    for i in range(len(val_set)):
        for landmark in S.landmarks:
            structure = val_set.__getitem__(i)['structure'].squeeze(0)
            locations = np.nonzero(np.round(structure) == landmark)
            x, y, z = locations[0][1], locations[0][0], locations[0][2]
            empty_struc = np.zeros((128,128,80))
            empty_struc[y][x][z] = landmark
            #structure_extrac = eval_func.extract_landmark_for_structure_np(structure, landmark)
            eval_func.plot_3d_pred_img_no_pred(val_set.__getitem__(i)['image'].squeeze(0).cpu().numpy(), empty_struc, S.threshold_img_print, val_path_ct, val_set.__getitem__(i)['patient'], landmark)
        print(val_set.__getitem__(i)['patient'])
    
    
        
    # test set
    file_name_test = "test_cts"
    test_path_ct = os.path.join(S.run_path, file_name_test)
    try: 
        os.mkdir(test_path_ct)
    except OSError as error:
        print(error)
    
    for i in range(len(test_set)):
        for landmark in S.landmarks:
            structure = test_set.__getitem__(i)['structure'].squeeze(0)
            locations = np.nonzero(np.round(structure) == landmark)
            if (locations.size == 0):
                x = 0
                y = 0
                z = 0
            else:
                print('locations size')
                print(locations.size)
                print('locations shape')
                print(locations.shape)
                x, y, z = locations[0][1], locations[0][0], locations[0][2]
            empty_struc = np.zeros((128,128,80))
            empty_struc[y][x][z] = landmark
            #structure_extrac = eval_func.extract_landmark_for_structure_np(structure, landmark)
            eval_func.plot_3d_pred_img_no_pred(test_set.__getitem__(i)['image'].squeeze(0).cpu().numpy(), empty_struc, S.threshold_img_print, test_path_ct, test_set.__getitem__(i)['patient'], landmark)
        print(test_set.__getitem__(i)['patient'])
    
    
    #img = dataset.__getitem__(10)['image']
    #idx = dataset.__getitem__(10)['idx']
    #print(idx)
    
    batch_accumulation = math.ceil(train_set.__len__()/S.batch_size) # rounds it up
    
    os.chdir(S.coding_path) # change to data path and change back at end
    #print(os.getcwd())
    
    