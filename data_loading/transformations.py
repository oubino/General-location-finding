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

class Resize(object):

  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      image, idx, patient, coords = sample['image'],  sample['idx'], sample['patient'], sample['coords']
      depth = self.depth # 256 - new width(128)
      width = self.width
      height = self.height
    
      d_pre, h_pre, w_pre = image.shape[:3]
          
      image = skimage.transform.resize(image, (depth, height, width), order = 1, preserve_range=True, anti_aliasing=True)
          
      d_post, h_post, w_post = image.shape[:3] 
      
      # downsize coords location
      for l in S.landmarks_total:
          coords[l]['x'] = coords[l]['x'] * w_post/w_pre
          coords[l]['y'] = coords[l]['y'] * h_post/h_pre
          coords[l]['z'] = coords[l]['z'] * d_post/d_pre
    
      S.downsample_ratio_list[patient] = {}
      S.downsample_ratio_list[patient]['h'] = h_pre/h_post
      S.downsample_ratio_list[patient]['w'] = w_pre/w_post
      S.downsample_ratio_list[patient]['d'] = d_pre/d_post
      
      return {'image':image, 'idx': idx, 'patient':patient, 'coords':coords} # note note !'structure': structure_new

class CentreCrop(object):    
  def __init__(self, depth, width, height):
      self.depth = depth
      self.width = width
      self.height = height
                          
  def __call__(self, sample):
      image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
      d, h, w = image.shape[:3] # define image height, width, depth as first 3 values
      
      # location around which to crop
      
      if S.train_line == True:
          crop_coords_1 = functions.load_obj_pickle(S.root, 'crop_coords_Oli')
          crop_coords_2 = functions.load_obj_pickle(S.root, 'crop_coords_Aaron')
          crop_coords = functions.line_learn_crop(crop_coords_1[patient], crop_coords_2[patient])
      elif S.train_line == False:
          crop_coords = functions.load_obj_pickle(S.root, 'crop_coords_' + S.clicker)
          crop_coords = crop_coords[patient]
      
      x_crop = crop_coords['x']
      y_crop = crop_coords['y']
      z_crop = crop_coords['z']
                      
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
      
      # append crops to list
      S.crop_list[patient] = {}
      S.crop_list[patient]['x_left'] = x_left
      S.crop_list[patient]['y_left'] = y_left
      S.crop_list[patient]['z_left'] = z_left
          
      for l in S.landmarks:              
          # need to amend coords of structure
          x,y, z = coords[l]['x'], coords[l]['y'], coords[l]['z']
         
          # post coords
          x = x - x_left
          y = y - y_left
          z = z - z_left
          coords[l]['x'] = x
          coords[l]['y'] = y
          coords[l]['z'] = z
    
          # failsafe
          if x < 0 or x >= S.in_x or y < 0 or y >= S.in_y or z < 0 or z >= S.in_z:
              print('exiting due to crop failsafe')
              exit()
                        
      if z_left < 0:
          image = np.pad(image, ((-z_left,0),(0, 0), (0, 0)))
          image_crop = skimage.util.crop(image, ((0,0),(y_left, y_right), (x_left, x_right)))
      elif z_left >= 0:
          image_crop = skimage.util.crop(image, ((z_left,z_right),(y_left, y_right), (x_left, x_right)))
          
      if image_crop.shape[0] != S.in_z:
        print('crop error')
        print('image shape orig')
        print(image.shape)
        print('image shape')
        print(image_crop.shape)
        print('zleft, z right')
        print(z_left, z_right)
        print('z_left+z_right vs d - self.depth')
        print(z_left + z_right,d - self.depth )
          
      return {'image':image_crop, 'idx':idx, 'patient':patient, 'coords':coords} # note note !
    
class Upsidedown_scipy(object):
    def __call__(self,sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
        # if upside down need to flip
        # if left cochlea landmark = 5 above 1/2
        # data is y, x, z
        # if top structure is below bottom structure then flips by 180
        upside_down = False
        for i in range(len(S.top_structures)):
            top_structure = S.top_structures[i]
            bot_structure = S.bot_structures[i]
            top_location = coords[top_structure]
            bot_location = coords[bot_structure]
            
            if top_location['z'] < bot_location['z']:
                upside_down = True
                print('upside down!')
        
        if upside_down == True:
            angle = 180
            image = scipy.ndimage.rotate(image, angle, axes = [2,0], reshape = False, order = 3)
            # rotate coords up/down 
            for l in S.landmarks_total:
                coords[l]['z'] = image.shape[0] - 1 - coords[l]['z'] # e.g. loc 25 in z image size 80 becomes 54 (as start from 0)
                coords[l]['x'] = image.shape[2] - 1 - coords[l]['x']
            # flip left and right structures
            for i in range(len(S.left_structures)):
                left_structure = S.left_structures[i]
                right_structure = S.right_structures[i]
                left_location = coords[left_structure]
                right_location = coords[right_structure]
                coords[right_structure] = left_location
                coords[left_structure] = right_location
            print('flipped')
            return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords}
        else:
            return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords}
              
        
class Normalise(object):  
  """ Normalise CT scan in the desired examination window
      takes in image as numpy """
  
  def __init__(self, level_min, level_max, window):
      self.level_min = level_min
      self.level_max = level_max
      self.window = window
                          
  def __call__(self, sample):
      image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
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
      
      for l in S.landmarks_total:
          z, y, x = coords[l]['z'], coords[l]['y'], coords[l]['x']
          img_norm[int(z)][int(y)][int(x)] = 100*l
      
      return {'image':img_norm, 'idx': idx, 'patient':patient, 'coords': coords} # note note !
  
class Shift(object):
    def __call__(self, sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
        
        x_shift, y_shift, z_shift = random.randint(-10,10), random.randint(-10,10), random.randint(-10,10)          
        out_of_bounds = False      
        for l in S.landmarks_total:
            if (coords[l]['x'] + x_shift < 0) or (coords[l]['x'] + x_shift >= S.in_x) or (coords[l]['y'] + y_shift < 0) or (coords[l]['y'] + y_shift >= S.in_y) or  (coords[l]['z'] + z_shift < 0) or (coords[l]['z'] + z_shift >= S.in_z):
                out_of_bounds = True
                    
        if out_of_bounds == False:
            for l in S.landmarks_total:
                coords[l]['x'] = coords[l]['x'] + x_shift
                coords[l]['y'] = coords[l]['y'] + y_shift
                coords[l]['z'] = coords[l]['z'] + z_shift
            image = scipy.ndimage.shift(image, (z_shift, y_shift, x_shift))
        else:
            print('shift out of bounds')
                
        return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords}
        

class Flips_scipy(object):
    def __call__(self,sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
        #random_number = random.random()
        random_number = 0.2
        angle = random.randint(-10, 10)
        
        coords_rotat = {}
        for k in S.landmarks_total:
            coords_rotat[k] = {'x': 0, 'y':0, 'z':0}
            
        if random_number <= 0.33:
            out_of_bounds = False
            for l in S.landmarks_total:
                coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z'] = functions.rotate(coords[l]['x'], coords[l]['y'], coords[l]['z'], S.in_x, S.in_y, S.in_z, angle, [1,0])
             
                if coords_rotat[l]['x'] < 0 or coords_rotat[l]['x'] >= S.in_x or coords_rotat[l]['y'] < 0 or coords_rotat[l]['y'] >= S.in_y or coords_rotat[l]['z'] < 0 or coords_rotat[l]['z'] >= S.in_z:
                    out_of_bounds = True                   
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [1,0],reshape = False, order = 3)
                coords[l]['x'], coords[l]['y'], coords[l]['z'] = coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z']
            else:
                print('ROTATION OUT OF BOUNDS')
                

        elif (random_number > 0.33) and (random_number <= 0.66):
            out_of_bounds = False
            for l in S.landmarks_total:
                coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z'] = functions.rotate(coords[l]['x'], coords[l]['y'], coords[l]['z'], S.in_x, S.in_y, S.in_z, angle, [1,2])
             
                if coords_rotat[l]['x'] < 0 or coords_rotat[l]['x'] >= S.in_x or coords_rotat[l]['y'] < 0 or coords_rotat[l]['y'] >= S.in_y or coords_rotat[l]['z'] < 0 or coords_rotat[l]['z'] >= S.in_z:
                    out_of_bounds = True                   
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [1,2],reshape = False, order = 3)
                coords[l]['x'], coords[l]['y'], coords[l]['z'] = coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z']
            else:
                print('ROTATION OUT OF BOUNDS')

        else:
            #structure = scipy.ndimage.rotate(structure, angle, axes = [1,0], reshape = False, order = 0)
            out_of_bounds = False
            for l in S.landmarks_total:
                coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z'] = functions.rotate(coords[l]['x'], coords[l]['y'], coords[l]['z'], S.in_x, S.in_y, S.in_z, angle, [2,0])            
                if coords_rotat[l]['x'] < 0 or coords_rotat[l]['x'] >= S.in_x or coords_rotat[l]['y'] < 0 or coords_rotat[l]['y'] >= S.in_y or coords_rotat[l]['z'] < 0 or coords_rotat[l]['z'] >= S.in_z:
                    out_of_bounds = True    
            # check if still within bounds due to rotation!
            if out_of_bounds == False:
                image = scipy.ndimage.rotate(image, angle, axes = [2,0],reshape = False, order = 3)
                coords[l]['x'], coords[l]['y'], coords[l]['z'] = coords_rotat[l]['x'], coords_rotat[l]['y'], coords_rotat[l]['z']
            else:
                print('ROTATION OUT OF BOUNDS')
        return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords}
      
class Horizontal_flip(object):
    def __call__(self,sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
        random_number = random.random()
        if random_number <= 0.5:
            image = np.flip(image, axis = 2).copy()
            # flip coordinate in x axis only
            for l in S.landmarks_total:
                x = coords[l]['x']
                x_new = image.shape[2] - 1 - x # 0 to 127
                coords[l]['x'] = x_new
            # flip left right structures left becomes right and vice versa
            for k in range(len(S.left_structures)):
                left_structure = S.left_structures[k]
                right_structure = S.right_structures[k]
                left_location = coords[left_structure]
                right_location = coords[right_structure]
                coords[left_structure] = right_location
                coords[right_structure] = left_location

        return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords }

    
class Check_left_right(object):
    def __call__(self,sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']
        for i in range(len(S.left_structures)):
            left_structure = S.left_structures[i]
            right_structure = S.right_structures[i]
            left_location = coords[left_structure]
            right_location = coords[right_structure]
            if right_location['x'] > left_location['x']: # if right x is greater than left x
                print('ERROR LEFT AND RIGHT WRONG WAy RouND')
                S.error_counter += 1
        
        return {'image': image, 'idx': idx, 'patient':patient, 'coords':coords}   


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, idx, patient, coords = sample['image'], sample['idx'], sample['patient'], sample['coords']

        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z                
            x, y, z = int(round(coords[l]['x'])), int(round(coords[l]['y'])), int(round(coords[l]['z']))
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
            #structure[z][y][x] = l
            coords[l]['x'], coords[l]['y'], coords[l]['z'] = x,y,z
            
            #print_2D_slice(image, l, x, y, z, patient)
            print('landmark, x, y, z')
            print(l, x, y , z)
            locations = np.nonzero(image > 100)
            print(locations)
            z, y, x = locations[2], locations[1], locations[0]
            print(x, y, z)
            
        # swap color axis because
        # numpy image: D x H x W 
        # torch image: C X H X W x D
        image = image.transpose(1,2,0)
        image = torch.from_numpy(image).float() # dont know why images/mask casted to float here but need to do it again later
        image = image.unsqueeze(0)
        return {'image': image,'idx': idx, 'patient':patient, 'coords':coords}
    
import os
import matplotlib.pyplot as plt
def print_2D_slice(img, landmark, struc_x, struc_y, struc_z, patient):
    
    # image
    #  D x H x W
        
    plt.figure(figsize=(7, 7))
        
    img = img[struc_z, :, :]
    
    # ---- plot as point ------
    plt.imshow(img,cmap = 'Greys_r', alpha = 0.9)
    plt.plot(struc_x, struc_y, color = 'red', marker = 'x', label = 'target')
    # add z annotation
    plt.annotate("%1.0f" % int(struc_z),(struc_x, struc_y), color = 'red')
    plt.legend()
    # ------------------------------------
    save_file = "print_img"
    save_path = os.path.join(S.run_path, save_file)
    try: 
      os.mkdir(save_path)
    except OSError as error:
      print(error)
    img_name = os.path.join(save_path, "2d_slice_%s.png" % patient.replace('.npy', '_%1.0f') % landmark)
    S.img_counter_3 += 1
    plt.savefig(img_name)


"""    
class ToTensor_no_ds(object):
    Convert ndarrays in sample to Tensors.

    def __call__(self, sample):
        image, structure, idx, patient, coords = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords']
        structure = np.zeros(structure.shape)
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z                
            x, y, z = int(round(coords[l]['x'])), int(round(coords[l]['y'])), int(round(coords[l]['z']))
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
class Rescale(object): 
    # need to change rescale so longer side is matched to int and then pad
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    

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
    
class Check_landmark_still_there(object):
    Check landmark still present during transformations 
    def __init__(self, location, test):
        self.location = location
        self.test = test
    def __call__(self, sample):
        image, structure, idx, patient, coordinates = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords']
        for l in S.landmarks_total:
            # structure is z, y, x
            # need it in y, x, z
            # the false at the end doesn't really matter here
            coords = numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, patient, self.test)[0]
            #if sum(coords) != 0:
            #    print('coordinates from rotation normal for')
            #    print(l)
            #    print(coords[0], coords[1], coords[2])
            print('coordinates old')
            print(l, coords)
            if sum(coords) == 0:
                print('landarks not present post %s' % self.location)
        print('coordinates new')
        print(coordinates)
        return {'image':image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords':coordinates} # note note !   
    
class Extract_landmark_location(object):
     Convert structure to tensor of zeros with one value at the desired landmark location 
    def __init__(self, test):
        self.test = test
        # test is only used if training on line!
        # if train then returns point on line, if test returns COM of line (halfway between Aaron and Oli)
        # see numpy_loc.py/landmark_loc_np

    def __call__(self,sample):
        image, structure, idx, patient, coordinates = sample['image'], sample['structure'], sample['idx'], sample['patient'], sample['coords']
        #structure_mod = np.zeros(structure.shape)
        
        
        coords_struc = {}
        for l in S.landmarks_total:
            coords_struc[l] = {}
            # structure is z, y, x
            # need it in y, x, z
            coords = numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, patient, self.test)[0]
            if sum(coords) != 0 :
                x, y, z = coords[0], coords[1], coords[2]
                #structure_mod[z][y][x] = l
                coords_struc[l]['x'], coords_struc[l]['y'], coords_struc[l]['z'] = x,y,z
        
        
        Compare_methods('extraction', coordinates, structure, patient)
        
        return {'image':image, 'structure': structure, 'idx': idx, 'patient':patient, 'coords': coordinates} # note note !
    
def Compare_methods(transform, coords, structure, patient):
    print('post %s' % transform)
    print('COORDS - no struc method')
    print(coords)
    coordinates = {}
    for k in S.landmarks_total:
      coordinates[k] = [0,0,0] 
    for l in S.landmarks_total:
        # structure is z, y, x
        # need it in y, x, z
        coords_check = numpy_loc.landmark_loc_np(S.landmarks_total_loc[l],structure,l, patient, True)[0]
        if sum(coords_check) != 0 :
            x, y, z = coords_check[0], coords_check[1], coords_check[2]
            coordinates[l] = [x,y,z]
    print('COORDS - struc method')
    print(coordinates)
    
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
    
    """