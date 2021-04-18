# dataloaders

from torchvision import transforms #, datasets, models
import torch
from torch.utils.data import DataLoader
import math
import os
import numpy as np
from sklearn.model_selection import KFold
from useful_functs import functions
from evaluate import evaluate_functions as eval_func
from data_loading import dataset_class as D
import settings as S
from data_loading import transformations as T
import matplotlib.pyplot as plt

#os.chdir(S.root) # change to data path and change back at end
#print(os.getcwd())

if S.downsample_user == True:
    trans_plain = transforms.Compose([T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Resize(S.in_z,S.in_x,S.in_y),T.Upsidedown_scipy(), T.Check_left_right(), T.ToTensor()])
    trans_augment = transforms.Compose([T.Resize(S.in_z,S.in_x,S.in_y),T.Upsidedown_scipy(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Shift(), T.Flips_scipy(), T.Horizontal_flip(), T.Check_left_right(), T.ToTensor()])
elif S.downsample_user == False:
    trans_plain = transforms.Compose([T.CentreCrop(S.in_z,S.in_x,S.in_y), T.Upsidedown_scipy(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Check_left_right(), T.ToTensor()])
    trans_augment = transforms.Compose([T.CentreCrop(S.in_z,S.in_x,S.in_y),T.Upsidedown_scipy(), T.Normalise(S.normal_min, S.normal_max, S.normal_window), T.Shift(), T.Flips_scipy(), T.Horizontal_flip(), T.Check_left_right(), T.ToTensor()])

#S.root_struc,
dataset = D.CTDataset(S.root, transform_train = trans_augment, transform_test = trans_plain, test = False)

def init(fold, train_ids, test_ids):
    # initialise dataloader 
    # split train_ids into val and train
    index = int(len(train_ids)/10) # val ids are first 10 percent
    
    val_ids = train_ids[:index]
    train_ids = train_ids[index:]
    
    global print_ids
    print_ids = test_ids
    # so can print out test ids at end
    
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    global batch_acc_batches  
    batch_acc_batches = math.ceil(len(train_ids)/S.batch_size) # rounds it up

    global dataloaders

    dataloaders = {
    'train': DataLoader(dataset, batch_size=S.batch_size, sampler= train_subsampler),
    'test': DataLoader(dataset, batch_size=S.batch_size_test, sampler= test_subsampler),
    'val': DataLoader(dataset, batch_size=S.batch_size, sampler= val_subsampler)  
    }


def init_load_k_fold(fold):
    # initialise dataloader 
    kfold = KFold(n_splits = S.folds_trained_with, shuffle = False)
    kfold_list = list(kfold.split(dataset))
    train_ids = kfold_list[fold][0]
    test_ids = kfold_list[fold][1]
              
    # split train_ids into val and train
    index = int(len(train_ids)/10) # val ids are first 10 percent
    
    val_ids = train_ids[:index]
    train_ids = train_ids[index:]
    
    global print_ids
    print_ids = test_ids
    
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    global batch_acc_batches
    batch_acc_batches = math.ceil(len(train_ids)/S.batch_size) # rounds it up

    global dataloaders

    dataloaders = {
    'train': DataLoader(dataset, batch_size=S.batch_size, sampler= train_subsampler),
    'test': DataLoader(dataset, batch_size=S.batch_size_test, sampler= test_subsampler),
    'val': DataLoader(dataset, batch_size=S.batch_size, sampler= val_subsampler)  
    }
    
def init_reserved_test_set():
    # data is entirely test set
    test_set = dataset
   
    
    global image_datasets
    image_datasets = {
        'test':test_set
    }
    
    global dataloaders
    # Load data in
    dataloaders = {
        'test': DataLoader(test_set,batch_size = S.batch_size_test, shuffle = False, num_workers=0)
    }
    
"""
def init_no_k_fold():
    # split data in train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(0))
    # manual seed ensures that same split everytime to ensure testing on correct dataset!
    # i.e. if random split, train, save, random split, test -> may end up testing on same as training!
    
    global batch_acc_batches   
    batch_acc_batches = math.ceil(train_size/S.batch_size) # rounds it up
    
    global image_datasets
    image_datasets = {
        'train': train_set, 'val': val_set, 'test':test_set
    }
    
    global dataloaders
    # Load data in
    dataloaders = {
        'train': DataLoader(train_set, batch_size=S.batch_size, shuffle = True, num_workers=0),
        'val': DataLoader(val_set, batch_size=S.batch_size, shuffle =True, num_workers=0),
        'test': DataLoader(test_set,batch_size = S.batch_size_test, shuffle = False, num_workers=0)
    }

def init_load_no_k_fold():
    # split data in train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(0))
    # manual seed ensures that same split everytime to ensure testing on correct dataset!
    # i.e. if random split, train, save, random split, test -> may end up testing on same as training!
    
    global batch_acc_batches  
    batch_acc_batches = math.ceil(train_size/S.batch_size) # rounds it up
    
    global image_datasets
    image_datasets = {
        'train': train_set, 'val': val_set, 'test':test_set
    }
    
    global dataloaders
    # Load data in
    dataloaders = {
        'train': DataLoader(train_set, batch_size=S.batch_size, shuffle = True, num_workers=0),
        'val': DataLoader(val_set, batch_size=S.batch_size, shuffle = True, num_workers=0),
        'test': DataLoader(test_set,batch_size = S.batch_size_test, shuffle = False, num_workers=0)
    }
    
"""   
print('Coding directory: ')
print(os.getcwd())
