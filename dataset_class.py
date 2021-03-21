# custom dataset class for the images and masks

import torch
from torch.utils.data import Dataset#, DataLoader
import os
import numpy as np
import settings

os.chdir(settings.root) # change to data path and change back at end

# CTDataset
class CTDataset(Dataset):
    """3D CT Scan dataset."""

    def __init__(self, root, transform_train =None, transform_test = None, test = False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "CTs")))) # ensure they're aligned & index them
        self.structures = list(sorted(os.listdir(os.path.join(root, "Structures"))))
        # self.structure_centres = list(sorted(os.listdir(os.path.join(root, "Structure Centres"))))
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.test = False
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx): # convert tensor to list to index items
            idx = idx.tolist() 
        img_path = os.path.join(self.root, "CTs", self.imgs[idx]) # image path is combination of root and index 
        structure_path = os.path.join(self.root, "Structures", self.structures[idx])
        # structure_centre_path = os.path.join(self.root, "Structure Centres", self.structure_centres[idx])

        img = np.load(img_path) # image read in as numpy array
        structure = np.load(structure_path) # mask - think 0 = background
        # structure_centre = np.load(structure_centre_path)
        
        #print(img.shape)

        sample = {'image': img, 'structure': structure} # both are nd.arrays, stored in sample dataset
        sample['idx'] = idx # should print out which image is problematic
        sample['patient'] = self.imgs[idx]

        sample['coords'] = {}
        for k in settings.landmarks_total:
            sample['coords'][k] = [0,0,0] # x,y,z
        
        
        if (self.transform_train) and (self.test == False):
            sample = self.transform_train(sample) # if transforms present, act on sample
        if (self.transform_test) and (self.test == True):
            sample = self.transform_test(sample) # if transforms present, act on sample
        
        #sample['idx'] = idx
        
        if (structure.max() != structure.min()): # exclude images where no mask
          #print('mask_max:%5.2f, mask_min:%5.2f' % (mask.max(),mask.min()))
          #print('x where its equal to 1')
          #print((np.where(mask == mask.max())[0]))
          return sample 
        else:
          print('no structure?')
    
    def __len__(self):
        return len(self.imgs) # get size of dataset

    def __test__(self):
      self.test = True
    
    def __train__(self):
      self.train = False

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

os.chdir(settings.coding_path) # change to data path and change back at end
