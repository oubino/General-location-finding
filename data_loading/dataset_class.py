# custom dataset class for the images and masks

import torch
from torch.utils.data import Dataset#, DataLoader
import os
import numpy as np

import settings
from useful_functs import functions

#os.chdir(settings.root) # change to data path and change back at end

# CTDataset
class CTDataset(Dataset):
    """3D CT Scan dataset."""

    def __init__(self, root, transform_train =None, transform_test = None, test = False, train = False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "CTs")))) # ensure they're aligned & index them
        #self.structures = list(sorted(os.listdir(os.path.join(root, "Structures"))))
        # self.structure_centres = list(sorted(os.listdir(os.path.join(root, "Structure Centres"))))
        self.transform_train = transform_train
        self.transform_test = transform_test
        #self.transform_test_no_ds = transform_test_no_ds
        self.test = False
        self.train = False
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx): # convert tensor to list to index items
            idx = idx.tolist() 
            
        img_path = os.path.join(self.root, "CTs", self.imgs[idx]) # image path is combination of root and index 
        img = np.load(img_path) # image read in as numpy array
        
        sample = {'image': img} # both are nd.arrays, stored in sample dataset
        sample['idx'] = idx # should print out which image is problematic
        sample['patient'] = self.imgs[idx]
        
        # load in structure coords
        if settings.train_line == True:
            struc_coord_1 = functions.load_obj_pickle(settings.root, 'coords_Oli')
            struc_coord_2 = functions.load_obj_pickle(settings.root, 'coords_Aaron')  
            struc_coord_1 = struc_coord_1[sample['patient']]
            struc_coord_2 = struc_coord_2[sample['patient']]  
            struc_coord = functions.line_learn_loc(struc_coord_1, struc_coord_2)
        elif settings.train_line == False:
            struc_coord = functions.load_obj_pickle(settings.root, 'coords_' + settings.clicker)  
            struc_coord = struc_coord[sample['patient']]
        sample['coords'] = struc_coord           
        
        if (self.transform_train) and (self.test == False):
            sample = self.transform_train(sample)
        if (self.transform_test) and (self.test == True):
            sample = self.transform_test(sample)
        
        return sample
    
    def __len__(self):
        return len(self.imgs) # get size of dataset

    def __test__(self):
      self.test = True
      self.train = False
    
    def __train__(self):
      self.train = True
      self.test = False

#  -------- think this is redundant ---------
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)

# --------------------------------------------

#os.chdir(settings.coding_path) # change to data path and change back at end
