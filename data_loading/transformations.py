import random
import skimage
import scipy.ndimage
from torchvision import transforms #, datasets, models
from skimage import transform
import torch
import numpy as np
import math


from useful_functs import functions
import settings as S
from data_loading import numpy_loc


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
        image, structure, idx, patient, coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords']
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
        img = transform.resize(image, (d, new_h, new_w),preserve_range = True, anti_aliasing = True) # anti-aliasing was false
        structure = transform.resize(structure, (d, new_h,new_w), preserve_range = True,  anti_aliasing = True) # anti-aliasing was false
        #mask_centre =  transform.resize(mask_centre, (new_h,new_w), preserve_range = True, anti_aliasing = True)
        #print(mask.max())
        #print(mask.min())

        return {'image': img, 'structure': structure, 'idx': idx, 'patient': patient, 'coords': coords}

class Resize(object):

  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      #print('resized!')
      image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
      depth = self.depth # 256 - new width(128)
      width = self.width
      height = self.height
    
      d_pre, h_pre, w_pre = image.shape[:3]
        
      image = skimage.transform.resize(image, (depth, width, height), order = 1, preserve_range=True, anti_aliasing=True)
      structure = skimage.transform.resize(structure, (depth, width, height), order = 0, preserve_range = True, anti_aliasing=False )
        
      d_post, h_post, w_post = image.shape[:3] 
    
      #S.downsample_ratio_h = np.append(S.downsample_ratio_h, h_pre/h_post)
      #S.downsample_ratio_w = np.append(S.downsample_ratio_w, w_pre/w_post)
      #S.downsample_ratio_d = np.append(S.downsample_ratio_d, d_pre/d_post)
      #.downsample_idx_list.append(patient) 
      S.downsample_ratio_list[patient] = {}
      S.downsample_ratio_list[patient]['h'] = h_pre/h_post
      S.downsample_ratio_list[patient]['w'] = w_pre/w_post
      S.downsample_ratio_list[patient]['d'] = d_pre/d_post
      
      '''
      structure_new = np.zeros(image.shape)
      for l in S.landmarks:
          #index = S.landmarks.index(l)
          old_index = np.where(np.round(structure) == l)
          if sum(old_index) != 0:
              new_z = int(old_index[0] * d_post/d_pre)
              new_y = int(old_index[1] * h_post/h_pre)
              new_x = int(old_index[2] * w_post/w_pre)
              structure_new[new_z][new_y][new_x] = l
        '''
      return {'image':image,'structure': structure , 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords': crop_coords} # note note !'structure': structure_new
  
class Fix_base_value(object):  
  """ Some images start from -1024, others from 0 make sure all start from 0 """
                            
  def __call__(self, sample):
      image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'] , sample['crop_coords']
      if np.round(np.amin(image)) < -1000:
          image = image + np.abs(np.round(np.amin(image)))
      return {'image':image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords': coords, 'crop_coords': crop_coords} # note note !
  
class Extract_landmark_location(object):
    """ Convert structure to tensor of zeros with one value at the desired landmark location """
    def __init__(self, test):
        self.test = test
        # test is only used if training on line!
        # if train then returns point on line, if test returns COM of line (halfway between Aaron and Oli)
        # see numpy_loc.py/landmark_loc_np

    def __call__(self,sample):
        image, structure, idx, patient, coordinates, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        structure_mod = np.zeros(structure.shape)
        
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z
            coords = numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, patient, self.test)[0]
            if sum(coords) != 0 :
                x, y, z = coords[0], coords[1], coords[2]
                structure_mod[z][y][x] = l
                coordinates[l] = [x,y,z]
        return {'image':image, 'structure': structure_mod, 'idx': idx, 'patient':patient, 'coords': coordinates, 'crop_coords':crop_coords} # note note !
    
class Check_landmark_still_there(object):
    """ Check landmark still present during transformations """
    def __init__(self, location, test):
        self.location = location
        self.test = test
    def __call__(self, sample):
        image, structure, idx, patient, coordinates, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z
            # the false at the end doesn't really matter here
            coords = numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, patient, self.test)[0]
            #if sum(coords) != 0:
            #    print('coordinates from rotation for')
            #    print(l)
            #    print(coords[0], coords[1], coords[2])
            print('coordinates old')
            print(l, coords)
            if sum(coords) == 0:
                print('landarks not present post %s' % self.location)
        print('coordinates new')
        print(coordinates)
        return {'image':image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coordinates, 'crop_coords':crop_coords} # note note !
        
        
class Normalise(object):  
  """ Normalise CT scan in the desired examination window
      takes in image as numpy """
  
  def __init__(self, level_min, level_max, window):
      self.level_min = level_min
      self.level_max = level_max
      self.window = window
                          
  def __call__(self, sample):
      image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
      # need to normalise around different values
      if np.round(np.amin(image)) < 0:
          print("MIN VALUE LESS THAN 0")
          print('ident and min value')
          print(patient, np.amin(image))
          S.error_counter += 1
      level = random.randint(self.level_min, self.level_max)
      minval = max(level - self.window/2, 0) # ensures don't go to negative values
      maxval = level + self.window/2
      img_norm = np.clip(image, minval, maxval)
      img_norm -= minval
      img_norm /= self.window
      return {'image':img_norm, 'structure': structure, 'idx': idx, 'patient':patient, 'coords': coords, 'crop_coords':crop_coords} # note note !

class CentreCrop(object):    
  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      #print('cropped')
      image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']

      d, h, w = image.shape[:3] # define image height, width, depth as first 3 values
      
      #print('depth height width')
      #print(d,h,w)
      
      counter = 0
      for l in S.landmarks:    
              
          # need to amend coords of structure
          x_crop, y_crop, z_crop = crop_coords[l][0], crop_coords[l][1], crop_coords[l][2]
          x,y, z = coords[l][0], coords[l][1], coords[l][2]
          #print('pre coords')
          #print(x,y,z)
          #print('crop coords')
          #print(x_crop, y_crop, z_crop)
          
          # crop .. (30,100) removes first 30 pixels from LHS and last 100 pixels from RHS   
          x_left = max(int(np.round(x_crop)) - self.width/2, 0)
          x_left = min(x_left, w - self.width) # e.g. x left max is 150, min is 0
          x_right = w - x_left - self.width
          
          y_left = max(int(np.round(y_crop)) - self.height/2, 0)
          y_left = min(y_left, h- self.height) # e.g. x left max is 150, min is 0
          y_right = h - y_left - self.height
          
          z_left = max(int(np.round(z_crop)) - self.depth/2, 0)
          z_left = min(z_left, d - self.depth) # e.g. x left max is 150, min is 0
          z_right = d - z_left - self.depth
          
          if x_left + x_right != w - self.width:
              print('issue crop x')
              print(x_left + x_right, w - self.width)
              if x_left + x_right == w - self.width - 1:
                  x_left = x_left + 1
              elif x_left + x_right == w - self.width + 1:
                  x_left = x_left - 1
         
          if y_left + y_right != h - self.height:
              print('issue crop y')
              print(y_left + y_right, h - self.height)
              if y_left + y_right == h - self.height - 1:
                  y_left = y_left + 1
              elif y_left + y_right == h - self.height + 1:
                  y_left = y_left - 1
              
          if z_left + z_right != d - self.depth:
              print('issue crop z')
              print(z_left + z_right, d - self.depth)
              if z_left + z_right == d - self.depth - 1:
                  z_left = z_left + 1
              elif z_left + z_right == d - self.depth + 1:
                  z_left = z_left - 1
          
          # print 
         # print('x left right y left right z left right')
          #print(x_left, x_right, y_left, y_right, z_left, z_right)
          
          # post coords
          x = x - x_left
          y = y - y_left
          z = z - z_left
          coords[l][0] = x
          coords[l][1] = y
          coords[l][2] = z
          
          #print('post coords')
          #print(x,y,z)
              
          image_crop = skimage.util.crop(image, ((z_left,z_right),(x_left, x_right), (y_left, y_right)))
          structure_crop = skimage.util.crop(structure, ((z_left,z_right),(x_left, x_right), (y_left, y_right)))
          
          if structure_crop.shape[0] != 30 or image_crop.shape[0] != 30:
              print('30 error')
              print('image shape orig')
              print(image.shape, structure.shape)
              print('image shape, struc shape')
              print(image_crop.shape, structure_crop.shape)
              print('zleft, z right')
              print(z_left, z_right)
              print('z_left+z_right vs d - self.depth')
              print(z_left + z_right,d - self.depth )
          
          if counter == 0:
              structure_output = np.expand_dims(structure_crop, 0)
              image_output = np.expand_dims(image_crop, 0)
              #print('image output shape')
              #print(image_output.shape)
          elif counter != 0:
              structure_output = np.concatenate((structure_output,np.expand_dims(structure_crop, 0)), axis = 0)
              #print('image output shape')
              #print(image_output.shape)
              image_output = np.concatenate((image_output,np.expand_dims(image_crop, 0)), axis = 0)
              #print('image crop shape no expanded')
              #print(image_crop.shape)
              #print('image outputput shape')
              #print(image_output.shape)
          counter += 1
          
          # create image and structure array which has first dimension = number of landmarks
          # shape should be 10xdxhxw
      #print('image shape, struc shape')
      #print(image_output.shape, structure_output.shape)
          
          
      """
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

        """
     # d, h, w = image.shape[:3] # define image height, width, depth as first 3 values

      return {'image':image_output, 'structure': structure_output, 'idx':idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords} # note note !

    
    


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, structure, idx, patient, coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords']
        structure = np.zeros(structure.shape)
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z                
            x, y, z = int(round(coords[l][0])), int(round(coords[l][1])), int(round(coords[l][2]))
            # if z is 80 round to 79
            if z >= S.in_z:
                print('Z BIGGER THAN Z MAX')
                print(z)
                z = S.in_z - 1
            if y >= S.in_y:
                print('Y BIGGER THAN Y MAX')
                print(y)
                y = S.in_y - 1
            if x >= S.in_x:
                print('X BIGGER THAN X MAX')
                print(x)
                x = S.in_x - 1
            structure[z][y][x] = l
        # swap color axis because
        # numpy image: D x H x W 
        # torch image: C X H X W x D
        image = image.transpose(1,2,0)
        structure = structure.transpose(1,2,0)
        image = torch.from_numpy(image).float() # dont know why images/mask casted to float here but need to do it again later
        structure = torch.from_numpy(structure).float()
        structure = structure.unsqueeze(0) # force mask to have extra dimension i.e. (1xHxWxD)
        image = image.unsqueeze(0)
        return {'image': image,'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords}
    
class ToTensor_crop(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        structure = np.zeros(structure.shape)
        counter = 0
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z                
            x, y, z = int(round(coords[l][0])), int(round(coords[l][1])), int(round(coords[l][2]))
            # if z is 80 round to 79
            if z >= S.crop_size_z or z < 0:
                print('Z COORD STRUCTURE OUTSIDE CROP')
                print('need to think about off central heatmap')
                print(z)
                z = 0
            if y >= S.crop_size_y or y < 0:
                print('Y COORD STRUCTURE OUTSIDE CROP')
                print('need to think about off central heatmap')
                print(y)
                y = 0
            if x >= S.crop_size_x or x < 0:
                print('X COORD STRUCTURE OUTSIDE CROP')
                print('need to think about off central heatmap')
                print(x)
                x = 0 
            structure[counter][z][y][x] = l
            counter += 1
        # swap color axis because
        # numpy image: landmarks x D x H x W 
        # torch image: landmarks x C X H X W x D
        image = image.transpose(0,2,3,1) # L x H x W x D
        structure = structure.transpose(0,2,3,1)
        image = torch.from_numpy(image).float() # dont know why images/mask casted to float here but need to do it again later
        structure = torch.from_numpy(structure).float()
        structure = structure.unsqueeze(1) # # L x C x H x W x D (c = channels or batch {notation})
        image = image.unsqueeze(1)
        
        return {'image': image,'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords}



class Flips_scipy(object):
    def __call__(self,sample):
        image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        random_number = random.random()
        angle = random.randint(-10, 10)
        if random_number <= 0.33:
            out_of_bounds = False
            for l in S.landmarks_total:
                x_rotat, y_rotat, z_rotat = functions.rotate(coords[l][0], coords[l][1], coords[l][2], S.in_x, S.in_y, S.in_z, angle, [1,0])
             
                if x_rotat < 0 or x_rotat >= S.in_x or y_rotat < 0 or y_rotat >= S.in_y or z_rotat < 0 or z_rotat >= S.in_z:
                    out_of_bounds = True                   
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [1,0],reshape = False, order = 0)
                coords[l][0], coords[l][1], coords[l][2] = x_rotat, y_rotat, z_rotat
                
                # rotate crop locations
                crop_coords[l][0], crop_coords[l][1], crop_coords[l][2] = functions.rotate(crop_coords[l][0], crop_coords[l][1], crop_coords[l][2], S.in_x, S.in_y, S.in_z, angle, [1,0])                                           
            else:
                print('ROTATION OUT OF BOUNDS')
                

        elif (random_number > 0.33) and (random_number <= 0.66):
            out_of_bounds = False
            for l in S.landmarks_total:
                x_rotat, y_rotat, z_rotat = functions.rotate(coords[l][0], coords[l][1], coords[l][2], S.in_x, S.in_y, S.in_z, angle, [1,2])
             
                if x_rotat < 0 or x_rotat >= S.in_x or y_rotat < 0 or y_rotat >= S.in_y or z_rotat < 0 or z_rotat >= S.in_z:
                    out_of_bounds = True                   
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [1,2],reshape = False, order = 0)
                coords[l][0], coords[l][1], coords[l][2] = x_rotat, y_rotat, z_rotat
                
                # rotate crop locations
                crop_coords[l][0], crop_coords[l][1], crop_coords[l][2] = functions.rotate(crop_coords[l][0], crop_coords[l][1], crop_coords[l][2], S.in_x, S.in_y, S.in_z, angle, [1,2])               
            else:
                print('ROTATION OUT OF BOUNDS')

        else:
            #structure = scipy.ndimage.rotate(structure, angle, axes = [1,0], reshape = False, order = 0)
            out_of_bounds = False
            for l in S.landmarks_total:
                x_rotat, y_rotat, z_rotat = functions.rotate(coords[l][0], coords[l][1], coords[l][2], S.in_x, S.in_y, S.in_z, angle, [2,0])            
                if x_rotat < 0 or x_rotat >= S.in_x or y_rotat < 0 or y_rotat >= S.in_y or z_rotat < 0 or z_rotat >= S.in_z:
                    out_of_bounds = True                   
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [2,0],reshape = False, order = 0)
                coords[l][0], coords[l][1], coords[l][2] = x_rotat, y_rotat, z_rotat            
                # rotate crop locations
                crop_coords[l][0], crop_coords[l][1], crop_coords[l][2] = functions.rotate(crop_coords[l][0], crop_coords[l][1], crop_coords[l][2], S.in_x, S.in_y, S.in_z, angle, [2,0])                       
            else:
                print('ROTATION OUT OF BOUNDS')

        return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}
    
def Flip_left_right_structures(structure):
    
    for i in range(len(S.left_structures)):
        # all locations of left structure, take on value of right structure
        # and vice versa
        # structure is DxHxW
        indices_left = np.round(structure) == S.left_structures[i]
        indices_right = np.round(structure) == S.right_structures[i]
        # trial method if maximum right structure coord > maximum left structure coord then flip
        if (np.nonzero(indices_left)[2].size != 0) and (np.nonzero(indices_right)[2].size != 0):
            structure[indices_left] = S.right_structures[i] 
            structure[indices_right] = S.left_structures[i] 
        elif (np.nonzero(indices_left)[2].size == 0) and (np.nonzero(indices_right)[2].size != 0):
            structure[indices_right] = S.left_structures[i] 
        elif (np.nonzero(indices_left)[2].size != 0) and (np.nonzero(indices_right)[2].size == 0):
            structure[indices_left] = S.right_structures[i] 
    return structure

class Check_left_right(object):
    def __call__(self,sample):
        image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        for i in range(len(S.left_structures)):
            left_structure = S.left_structures[i]
            right_structure = S.right_structures[i]
            left_location = coords[left_structure]
            right_location = coords[right_structure]
            if right_location[0] > left_location[0]: # if right x is greater than left x
                print('ERROR LEFT AND RIGHT WRONG WAy RouND')
                S.error_counter += 1
            """
            indices_left = np.round(structure) == S.left_structures[i]
            indices_right = np.round(structure) == S.right_structures[i]
            if (np.nonzero(indices_left)[2].size != 0) and (np.nonzero(indices_right)[2].size != 0):
                min_right = np.amin(np.nonzero(indices_right)[2])
                max_left = np.amax(np.nonzero(indices_left)[2])
                if min_right > max_left:
                    print('ERROR LEFT AND RIGHT WRONG WAy RouND')
                    S.error_counter += 1
                    """
        return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}
    
    
class Horizontal_flip(object):
    def __call__(self,sample):
        image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        random_number = random.random()
        if random_number <= 0.5:
            image = np.flip(image, axis = 2).copy()
            #structure = np.flip(structure, axis = 2).copy() # get rid
            #structure = Flip_left_right_structures(structure) # get rid
            # flip coordinate in x axis only
            for l in S.landmarks_total:
                x,y,z = coords[l][0], coords[l][1], coords[l][2] 
                x_new = S.in_x - 1 - x # 0 to 127
                coords[l][0] = x_new
                x_crop,y_crop,z_crop = crop_coords[l][0], crop_coords[l][1], crop_coords[l][2] 
                x_new_crop = S.in_x - 1 - x_crop
                crop_coords[l][0] = x_new_crop 
            # flip left right structures left becomes right and vice versa
            for k in range(len(S.left_structures)):
                left_structure = S.left_structures[k]
                right_structure = S.right_structures[k]
                left_location = coords[left_structure]
                right_location = coords[right_structure]
                coords[left_structure] = right_location
                coords[right_structure] = left_location

        return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}
       
    
class Upsidedown_scipy(object):
    def __call__(self,sample):
        image, structure, idx, patient, coords, crop_coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords'], sample['crop_coords']
        # if upside down need to flip
        # if left cochlea landmark = 5 above 1/2
        # data is y, x, z
        # if top structure is below bottom structure then flips by 180
        counter_top = 0
        landmark_loc_top = np.where(structure == S.top_structures[counter_top])
        while landmark_loc_top[0].size == 0:
            counter_top += 1
            try:
                landmark_loc_top = np.where(structure == S.top_structures[counter_top])
            except:
                print('ERROR NONE OF THE TOP STRUCTURES FOUND IN IMAGE- returning unflipped image')
                return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}

        counter_bot = 0
        landmark_loc_bot = np.where(structure == S.bot_structures[counter_bot])
        while landmark_loc_bot[0].size == 0:
            counter_bot += 1
            try:
                landmark_loc_bot = np.where(structure == S.bot_structures[counter_bot])
            except:
                print('ERROR NONE OF THE BOTTOM STRUCTURES FOUND IN IMAGE- returning unflipped image')
                return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords,'crop_coords':crop_coords}

        z_landmark_top = landmark_loc_top[0][0]
        z_landmark_bot = landmark_loc_bot[0][0]
        #z_size = structure.shape[0] 
        if z_landmark_top < z_landmark_bot:
            angle = 180
            image = scipy.ndimage.rotate(image, angle, axes = [2,0], reshape = False, order =0)
            structure = scipy.ndimage.rotate(structure, angle, axes = [2,0], reshape = False, order =0)
            structure = Flip_left_right_structures(structure) # need to flip left/right structures
            return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}
        else:
            return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords, 'crop_coords':crop_coords}
        
    
class Upsidedown(object):
  def __call__(self,sample):
    image, structure, idx, coords = sample['image'], sample['structure'], sample['idx'], sample['coords']
    # if upside down need to flip
    # if left cochlea landmark = 5 above 1/2
    # note that cause kornea takes in z, y, x between pre and post tranpose
    # data is z, y, x
    # com structure takes in y, x, z -> x, y, z
    # therefore if take in z, y, x -> y, z, x (*1)
    structure_com = functions.com_structure(structure.unsqueeze(0), 5)
    structure_com_z = structure_com[0][0][1] # see (*1)
    z_size = structure.size()[1] # cause in this section is z, y, x
    #print(structure_com_z, z_size/2)
    if structure_com_z < z_size/2:
      flip = transforms.Compose([aug.RandomVerticalFlip3D(p=1)])
      image = flip(image)
      structure = flip(structure)
    #structure = torch.squeeze(structure,0)
    return {'image': image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coords}

class Unsqueeze(object):
  def __call__(self,sample):
    image, structure, idx, coords = sample['image'], sample['structure'], sample['idx'], sample['coords']
    image = image.unsqueeze(0)
    structure = structure.unsqueeze(0)
    return {'image': image, 'structure': structure, 'idx': idx, 'coords':coords}




# for now not going to use below because not sure if it acts on image and structure equally
class Affine(object):
  def __call__(self, sample):
    image, structure, idx, coords = sample['image'], sample['structure'], sample['idx'], sample['coords']
    if random.random() <= 0.3:
      affine = transforms.Compose([aug.RandomAffine3D(degrees=(0,10,10), translate=(0,0.05,0.05), same_on_batch=True, p=1)])
      image = affine(image)
      print('affined')
      structure = affine(structure)


    return{'image': image, 'structure': structure, 'idx': idx, 'coords':coords}

class Noise(object):  # helps prevent overfitting
  #Random noise images
  def __call__(self,sample):
    image, structure, idx = sample['image'], sample['structure'], sample['idx']
    if random.random() <= 1:
      noise = transforms.Compose(aug.RandomMotionBlur3D(5, 10, direction=0, p=1))
      image = noise(image)
    return {'image': image, 'structure': structure, 'idx': idx}

class Noise1(object):  # helps prevent overfitting
 # Random noise images
  def __call__(self,sample):
    image, structure= sample['image'], sample['structure']
    if random.random() <= 1:
      image = skimage.util.random_noise(image, mean = 0, var = 1.0000000001, clip = False)
      # would this work given that mask is unchanged??
    return {'image': image, 'structure': structure}
class Pre_Transpose(object):
  def __call__(self, sample):
    image, structure = sample['image'], sample['structure']
    image = torch.transpose(image, 1, 3)
    image = torch.transpose(image, 2, 3)
    structure = torch.transpose(structure, 1, 3)
    structure = torch.transpose(structure, 2, 3)

    return {'image': image, 'structure': structure}

class Post_Transpose(object):
  def __call__(self, sample):
    image, structure = sample['image'], sample['structure']
    #torch.squeeze(image, 0)
    #torch.squeeze(structure, 0)

    structure = torch.squeeze(structure,0)
    image = torch.squeeze(image, 0)
    #torch.squeeze(structure,0)
    #torch.squeeze(image, 0)
    image = torch.transpose(image, 3, 2)
    image = torch.transpose(image, 3, 1)
    structure = torch.transpose(structure, 3, 2)
    structure = torch.transpose(structure, 3, 1)

    return {'image': image, 'structure': structure}
'''
class Squeeze(object):
  def __call__(self, sample):
    image, structure = sample['image'], sample['structure']
    image = torch.squeeze(image, 0)
    print('squeeze')
    structure = torch.squeeze(structure, 0)
    return {'image': image, 'structure': structure}
'''
