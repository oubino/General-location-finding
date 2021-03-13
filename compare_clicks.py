import settings as S
import numpy as np
import os
import math
import csv

def com_structure_np(structure, landmark): # assumes 1 channel
  # structure is (D x H x W)
  # output is x,y,z
  landmark_present = []
  landmark = float(landmark) # ensure that comparison is made properly
  locations = np.nonzero(np.round(structure) == landmark)
  if (len(locations[0]) == 0): # if no landmarks detected for structure
    print('no structure found using np for %1.0f' % (landmark))
    landmark_present.append(False)
    x_com = 0
    y_com = 0
    z_com = 0
  else:
    landmark_present.append(True)
    x_com = 0
    y_com = 0
    z_com = 0
    for k in range(len(locations[0])): # number of landmarks
      x_com += locations[2][k]
      y_com += locations[1][k]
      z_com += locations[0][k]
    x_com /= len(locations[0])
    y_com /= len(locations[0])
    z_com /= len(locations[0])
  coords = [int(x_com),int(y_com),int(z_com)]
  return coords, landmark_present 

# paths
aaron_folder = r'/home/rankinaaron98/data/Facial_asymmetry_aaron/Structures'
oli_folder = r'/home/olive/data/Facial_asymmetry_oli/Structures'
save_structure_folder = r'/home/olive/data/Facial_asymmetry_combined/Structures'
save_ct_folder = r'/home/olive/data/Facial_asymmetry_combined/CTs'
load_ct_folder = r'/home/olive/data/Facial_asymmetry_oli/CTs'
csv_root = r'/home/rankinaaron98/data/Facial_asymmetry_aaron/'

# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]

# limit
limit = 12

# loop over .py
files_aaron = list(sorted(os.listdir(aaron_folder)))
files_oli = list(sorted(os.listdir(oli_folder)))

list_1 = [x for x in files_aaron if x in files_oli]


com_list_aaron = {}
com_list_oli = {}

for k in landmarks:
    com_list_aaron['%1.0f' % k] = []
    com_list_oli['%1.0f' % k] = []

# for common structures add mean to array for both aaron and oli
for i in list_1:
    py_array_aaron = np.load(os.path.join(aaron_folder,i))
    py_array_oli = np.load(os.path.join(oli_folder,i))
    for k in landmarks:
        #com_list['%1.0f' % k].append(5)
        #print(com_structure_np(py_array,k)[0])
        com_list_aaron['%1.0f' % k].append(com_structure_np(py_array_aaron,k)[0])
        com_list_oli['%1.0f' % k].append(com_structure_np(py_array_oli,k)[0])
        
dev_list = {}
dev_upper_limit_list = {}
mean_dev = {}
mean_list = {}

for k in landmarks:
    dev_list['%1.0f' % k] = []
    dev_upper_limit_list['%1.0f' % k] = []
    mean_dev['%1.0f' % k] = []
    mean_list['%1.0f' % k] = []
    
def pixel_to_mm(patient):
    data = csv.reader(open(os.path.join(csv_root, 'image_dimensions.csv')),delimiter=',')
    next(data) # skip first line
    list_img = list(data)#, key=operator.itemgetter(0))
    # sortedlist[img_number][0 = name, 1 = x/y, 2 = z]
    #image_idx = int(image_idx)
    pat_ind = patient.replace('.npy','')
    index = 0 
    for i in range(len(list_img)):
        if list_img[i][0] == pat_ind:
            index = i
    pixel_mm_x = float(list_img[index][1]) # 1 pixel = pixel_mm_x * mm
    pixel_mm_y = float(list_img[index][1])
    pixel_mm_z = float(list_img[index][2])
    
    return pixel_mm_x, pixel_mm_y, pixel_mm_z


# calculate deviation of arrays etc.
for j in range(len(list_1)):
    for k in landmarks:
        x_mm, y_mm, z_mm = pixel_to_mm(list_1[j])
        dev_x = abs(com_list_aaron['%1.0f' % k][j][2] - com_list_oli['%1.0f' % k][j][2])*(x_mm)
        dev_y = abs(com_list_aaron['%1.0f' % k][j][1] - com_list_oli['%1.0f' % k][j][1])*(y_mm)
        dev_z = abs(com_list_aaron['%1.0f' % k][j][0] - com_list_oli['%1.0f' % k][j][0])*(z_mm)
        dev = math.sqrt(dev_x**2 + dev_y**2 + dev_z**2)
        dev_list['%1.0f' % k].append(dev)
        if dev > limit:
            dev_upper_limit_list['%1.0f' % k].append(list_1[j])
            print('image: %s' % list_1[j])
            print('landmark: %1.0f' % k)
            print(dev)
            print('------------')

# average deviation per landmark
for k in landmarks:
    mean_dev_temp = 0
    for j in range(len(list_1)):
        mean_dev_temp += dev_list['%1.0f' % k][j]
    mean_dev_temp /= len(list_1)
    mean_dev['%1.0f' % k].append(mean_dev_temp)
    
# calculate mean of aaron and oli from arrays
for j in range(len(list_1)):
    for k in landmarks:
        mean_x = ((com_list_aaron['%1.0f' % k][j][2] + com_list_oli['%1.0f' % k][j][2])/2)
        mean_y = ((com_list_aaron['%1.0f' % k][j][1] + com_list_oli['%1.0f' % k][j][1])/2)
        mean_z = ((com_list_aaron['%1.0f' % k][j][0] + com_list_oli['%1.0f' % k][j][0])/2)
        coords = [mean_x, mean_y, mean_z]
        mean_list['%1.0f' % k].append(coords)
        
# ------ return a structure with just one point at the mean  ----- #

# for each image create an array
for i in list_1:
    py_array_load = np.load(os.path.join(aaron_folder,i))
    ct = np.load(os.path.join(load_ct_folder,i))
    structure = np.empty(py_array_load.shape)
    # add number at location of x y z for each landmark
    index = list_1.index(i)
    for k in landmarks:
        z = int(mean_list['%1.0f' % k][index][0])
        y = int(mean_list['%1.0f' % k][index][1])
        x = int(mean_list['%1.0f' % k][index][2])
        structure[z][y][x] = k # z y x
    # save image
    np.save(os.path.join(save_structure_folder,i), structure)
    # save ct
    np.save(os.path.join(save_ct_folder,i), ct)
    


# deviations per landmark per image
#print('deviations per landmark per image')
#print(dev_list)

# for each landmark, list of the images with deviations greater than ceratin distance
print('for each landmark, list of the images with deviations greater than certain distance')
print(dev_upper_limit_list)
            
# mean deviation per landmark
print('mean deviation per landmark')
print(mean_dev)
            