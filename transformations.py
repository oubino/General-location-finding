import random
import skimage
import scipy.ndimage
from torchvision import transforms #, datasets, models
from skimage import transform
#from kornia import augmentation as aug
import torch
import numpy as np
import functions
import settings as S

class Rescale(object): 
    # need to change rescale so longer side is matched to int and then pad
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # int keeps aspect ratio the same
        self.output_size = output_size

    def __call__(self, sample):
        # image, structure, structure_centre = sample['image'], sample['structure'], sample['structure_centre'] 
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        #print(mask.max())
        #print(mask.min())
        d, h, w = image.shape[:3] # define image height, width, depth as first 3 values

        
        if isinstance(self.output_size, int): # maintain aspect ratio so no loss of info
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        
        # h and w are swapped for mask because for images,
        # x and y axes are axis 1 and 0 respectively
        img = transform.resize(image, (d, new_h, new_w),preserve_range = True, anti_aliasing = False) # anti-aliasing was false
        structure = transform.resize(structure, (d, new_h,new_w), preserve_range = True,  anti_aliasing = False) # anti-aliasing was false
        #mask_centre =  transform.resize(mask_centre, (new_h,new_w), preserve_range = True, anti_aliasing = True)
        #print(mask.max())
        #print(mask.min())

        return {'image': img, 'structure': structure, 'idx': idx}

class Resize(object):

  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      image, structure, idx = sample['image'], sample['structure'], sample['idx']
      depth = self.depth # 256 - new width(128)
      width = self.width
      height = self.height
    
      d_pre, h_pre, w_pre = image.shape[:3]
        
      image = skimage.transform.resize(image, (depth, width, height), order = 0, preserve_range=True, anti_aliasing=False )
      structure = skimage.transform.resize(structure, (depth, width, height), order = 0, preserve_range=True, anti_aliasing=False )
        
      d_post, h_post, w_post = image.shape[:3] 
    
      S.downsample_ratio_h = np.append(S.downsample_ratio_h, h_pre/h_post)
      S.downsample_ratio_w = np.append(S.downsample_ratio_w, w_pre/w_post)
      S.downsample_ratio_d = np.append(S.downsample_ratio_d, d_pre/d_post)
      S.downsample_idx_list = np.append(S.downsample_idx_list, idx)     
        
      return {'image':image, 'structure': structure, 'idx': idx} # note note !
  
class Fix_base_value(object):  
  """ Some images start from -1024, others from 0 make sure all start from 0 """
                            
  def __call__(self, sample):
      image, structure, idx = sample['image'], sample['structure'], sample['idx']
      if np.round(np.amin(image)) < -1000:
          image = image + np.abs(np.round(np.amin(image)))
      return {'image':image, 'structure': structure, 'idx': idx} # note note !

class Normalise(object):  
  """ Normalise CT scan in the desired examination window
      takes in image as numpy """
  
  def __init__(self, level_min, level_max, window):
      self.level_min = level_min
      self.level_max = level_max
      self.window = window
                          
  def __call__(self, sample):
      image, structure, idx = sample['image'], sample['structure'], sample['idx']
      # need to normalise around different values
      if np.round(np.amin(image)) < 0:
          print("MIN VALUE LESS THAN 0")
          print('idx and min value')
          print(idx, np.amin(image))
          S.error_counter += 1
      level = random.randint(self.level_min, self.level_max)
      minval = max(level - self.window/2, 0) # ensures don't go to negative values
      maxval = level + self.window/2
      img_norm = np.clip(image, minval, maxval)
      img_norm -= minval
      img_norm /= self.window
      return {'image':img_norm, 'structure': structure, 'idx': idx} # note note !

class CentreCrop(object):    
  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      image, structure, idx = sample['image'], sample['structure'], sample['idx']

      d, h, w = image.shape[:3] # define image height, width, depth as first 3 values

      crop_w = (w - self.width) / 2  # 256 - new width(128)
      w_min = w / 4
      crop_h = (h - self.height) / 2
      crop_d = (d - self.depth) / 2
      d_min = crop_d / 4

      image = skimage.util.crop(image, ((0,0),(crop_w, crop_w), (crop_h, crop_h)))
      structure = skimage.util.crop(structure, ((0,0),(crop_w, crop_w),(crop_h,crop_h)))
      if self.depth < d:
        # crop
        crop_d = (d - self.depth) / 2
        
        image = skimage.util.crop(image,((crop_d,crop_d),(0,0),(0,0)))
        structure = skimage.util.crop(structure,((crop_d,crop_d),(0,0),(0,0)))

        new_d = image.shape[0]
        if new_d < self.depth:
          image = skimage.util.pad(image,((0,1),(0,0),(0,0)))
          structure = skimage.util.pad(structure,((0,1),(0,0),(0,0)))
        if new_d > self.depth:
          image = skimage.util.crop(image,((0,1),(0,0),(0,0)))
          structure = skimage.util.crop(structure,((0,1),(0,0),(0,0)))
      
      if self.depth > d:
        # pad
        pad_value = self.depth - d
        image = skimage.util.pad(image,((0,pad_value),(0,0),(0,0)))
        structure = skimage.util.pad(structure,((0,pad_value),(0,0),(0,0)))  

      d, h, w = image.shape[:3] # define image height, width, depth as first 3 values

      return {'image':image, 'structure': structure, 'idx':idx} # note note !


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        # swap color axis because
        # numpy image: D x H x W 
        # torch image: C X H X W x D
        image = image.transpose(1,2,0)
        structure = structure.transpose(1,2,0)
        image = torch.from_numpy(image).float() # dont know why images/mask casted to float here but need to do it again later
        structure = torch.from_numpy(structure).float()
        structure = structure.unsqueeze(0) # force mask to have extra dimension i.e. (1xHxWxD)
        image = image.unsqueeze(0)
        return {'image': image,'structure': structure, 'idx': idx}


class Flips_scipy(object):
    def __call__(self,sample):
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        random_number = random.random()
        angle = random.randint(-10, 10)
        if random_number <= 0.33:
            image = scipy.ndimage.rotate(image, angle, axes = [1,0],reshape = False, order = 0)
            structure = scipy.ndimage.rotate(structure, angle, axes = [1,0], reshape = False, order = 0)
        if (random_number > 0.33) and (random_number <= 0.66):
            image = scipy.ndimage.rotate(image, angle, axes = [1,2], reshape = False, order = 0)
            structure = scipy.ndimage.rotate(structure, angle, axes = [1,2], reshape = False, order = 0)
        else:
            image = scipy.ndimage.rotate(image, angle, axes = [2,0], reshape = False, order = 0)
            structure = scipy.ndimage.rotate(structure, angle, axes = [2,0], reshape = False, order = 0)
        return {'image': image, 'structure': structure, 'idx': idx}
    
class Flip_left_right_structures(object):
    def __call__(self,sample):
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        for i in range(len(S.left_structures)):
            # all locations of left structure, take on value of right structure
            # and vice versa
            # structure is DxHxW
            indices_left = np.round(structure) == S.left_structures[i]
            indices_right = np.round(structure) == S.right_structures[i]
            # trial method if maximum right structure coord > maximum left structure coord then flip
            if np.amin(np.nonzero(indices_right)[2]) > np.amax(np.nonzero(indices_left)[2]):
                structure[indices_left] = S.right_structures[i] 
                structure[indices_right] = S.left_structures[i] 
                #print('flipped landmarks')
            return {'image': image, 'structure': structure, 'idx': idx}

    
class Upsidedown_scipy(object):
    def __call__(self,sample):
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        # if upside down need to flip
        # if left cochlea landmark = 5 above 1/2
        # data is y, x, z
        counter = 0
        landmark_loc = np.where(structure == S.top_structures[counter])
        while landmark_loc[0].size == 0:
            counter += 1
            try:
                landmark_loc = np.where(structure == S.top_structures[counter])
            except:
                print('ERROR NONE OF THE TOP STRUCTURES FOUND IN IMAGE- returning unflipped image')
                return {'image': image, 'structure': structure, 'idx': idx}
        
        z_landmark = landmark_loc[0][0]
        z_size = structure.shape[0] 
        if z_landmark < z_size/2:
            angle = 180
            image = scipy.ndimage.rotate(image, angle, axes = [2,0], reshape = False, order =0)
            #rotate(input, angle, axes=1, 0, reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
            structure = scipy.ndimage.rotate(structure, angle, axes = [2,0], reshape = False, order =0)
            #print('flipped upside down')
            return {'image': image, 'structure': structure, 'idx': idx}
        else:
            return {'image': image, 'structure': structure, 'idx': idx}
        
class Horizontal_flip(object):
    def __call__(self,sample):
        image, structure, idx = sample['image'], sample['structure'], sample['idx']
        random_number = random.random()
        if random_number <= 0.5:
            #print('image shape', image.shape)
            image = np.flip(image, axis = 2).copy()
            structure = np.flip(structure, axis = 2).copy()
            #print('horizontal flipped')
            # flip any left right structures 
            #structure = flip_left_right_structures(structure)
            #print('horizontal flipped')
        return {'image': image, 'structure': structure, 'idx': idx}
       


class Unsqueeze(object):
  def __call__(self,sample):
    image, structure, idx = sample['image'], sample['structure'], sample['idx']
    image = image.unsqueeze(0)
    structure = structure.unsqueeze(0)
    return {'image': image, 'structure': structure, 'idx': idx}


class Noise1(object):  # helps prevent overfitting
 # Random noise images
  def __call__(self,sample):
    image, structure= sample['image'], sample['structure']
    if random.random() <= 1:
      image = skimage.util.random_noise(image, mean = 0, var = 1.0000000001, clip = False)
      # would this work given that mask is unchanged??
    return {'image': image, 'structure': structure}
