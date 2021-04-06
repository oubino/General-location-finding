# custom dataset class for the images and masks

import torch
from torch.utils.data import Dataset#, DataLoader
import os
import numpy as np

import settings

#os.chdir(settings.root) # change to data path and change back at end

# CTDataset
class CTDataset(Dataset):
    """3D CT Scan dataset."""

    def __init__(self, root, transform_train_resize =None, transform_test_resize = None, transform_test_no_ds = None, transform_train_crop = None, transform_test_crop = None):
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
        self.transform_train_resize = transform_train_resize
        self.transform_test_resize = transform_test_resize
        self.transform_test_no_ds = transform_test_no_ds
        self.transform_train_crop = transform_train_crop
        self.transform_test_crop = transform_test_crop
        self.test_resize = False
        self.train_resize = False
        self.train_crop = False
        self.test_crop = False
        
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
        
        
        if (self.transform_train_resize) and (self.train_resize == True):
            sample = self.transform_train_resize(sample) # if transforms present, act on sample
            #sample['structure'] = self.transform_train_resize(sample['structure'])
        elif (self.transform_test_resize) and (self.test_resize == True):
            sample_orig = self.transform_test_no_ds(sample)
            sample = self.transform_test_resize(sample)
            sample['structure_original'],sample['img_original'] =  sample_orig['structure'], sample_orig['image']
            #sample['structure'] = self.transform_test-resize(sample['structure'])
        elif (self.transform_train_crop) and (self.train_crop == True):
            sample = self.transform_train_crop(sample) # if transforms present, act on sample
            #sample['structure'] = self.transform_train_crop(sample['structure'])
        elif (self.transform_test_crop) and (self.test_crop == True):
            sample_orig = self.transform_test_no_ds(sample)
            sample = self.transform_test_crop(sample)
            sample['structure_original'],sample['img_original'] =  sample_orig['structure'], sample_orig['image']
            #sample['structure'] = self.transform_test_crop(sample['structure'])
            
        
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

    def __test_resize__(self):
      self.test_resize = True
      self.train_resize = False
      self.train_crop = False
      self.test_crop = False
 
    def __train_resize__(self):
      self.test_resize = False
      self.train_resize = True
      self.train_crop = False
      self.test_crop = False
      
    def __test_crop__(self):
      self.test_resize = False
      self.train_resize = False
      self.train_crop = False
      self.test_crop = True
    
    def __train_crop__(self):
      self.test_resize = False
      self.train_resize = False
      self.train_crop = True
      self.test_crop = False
     
    def __test_no_ds__(self):
      self.test_resize = False
      self.train_resize = False
      self.train_crop = False
      self.test_crop = False

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
