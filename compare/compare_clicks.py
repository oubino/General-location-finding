import numpy as np
import os
import math
import csv
from skimage.draw import line_nd
import matplotlib.pyplot as plt

def question(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

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
  coords = [int(z_com),int(y_com),int(x_com)]
  return coords, landmark_present 

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

def histogram(data, coord, landmark):
    # plot and save histogram
    data = np.array(data)
    data = np.sort(data)
    plt.figure()
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Deviation/mm')
    plt.ylabel('Frequency')
    plt.title("%s deviation for landmark %1.0f" % (coord, landmark))
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    hist_name = os.path.join(hist_root, "%s_dev_%1.0f" % (coord, landmark))
    # set x lim to centre around 0
    data_abs = abs(data)
    max_val = data_abs.max()
    plt.xlim(-max_val, max_val)  
    plt.savefig(hist_name)
    
def histogram_plain(data, name):
    # plot and save histogram
    data = np.array(data)
    data = np.sort(data)
    plt.figure()
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                            alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Deviation/mm')
    plt.ylabel('Frequency')
    plt.title("%s" % (name))
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    hist_name = os.path.join(hist_root, "%s" % (name))
    # set x lim to centre around 0
    data_abs = abs(data)
    max_val = data_abs.max()
    plt.xlim(-max_val, max_val)  
    plt.savefig(hist_name)

# paths
clicker_1_folder = r'/home/olive/data/Facial_asymmetry_oli/Structures'
#clicker_1_folder = r'/home/rankinaaron98/data/Facial_asymmetry_aaron_reclicks/Structures'
clicker_2_folder = r'/home/olive/data/Facial_asymmetry_oli_reclicks/Structures'

save_structure_folder = r'/home/olive/data/Facial_asymmetry_combined_reclicks_ajf/Structures'
save_ct_folder = r'/home/olive/data/Facial_asymmetry_combined_reclicks_ajf/CTs'

load_ct_folder = r'/home/olive/data/Facial_asymmetry_oli_reclicks/CTs'
csv_root = r'/home/rankinaaron98/data/Facial_asymmetry_aaron_reclicks/'

hist_root = r'/home/olive/data/Compare/Histograms_reclick_oli_old_new/'

# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]

# limit
limit = 4

# loop over .py
files_clicker_1 = list(sorted(os.listdir(clicker_1_folder)))
files_clicker_2 = list(sorted(os.listdir(clicker_2_folder)))

list_1 = [x for x in files_clicker_1 if x in files_clicker_2]


com_list_clicker_1 = {}
com_list_clicker_2 = {}
dev_list = {}
dev_list_x = {}
dev_list_y = {}
dev_list_z = {}
dev_upper_limit_list = {}
mean_dev = {}
mean_dev_x = {}
mean_dev_y = {}
mean_dev_z = {}
mean_dev_std = {}
mean_list = {}

for k in landmarks:
    com_list_clicker_1['%1.0f' % k] = []
    com_list_clicker_2['%1.0f' % k] = []
    dev_list['%1.0f' % k] = []
    dev_list_x['%1.0f' % k] = []
    dev_list_y['%1.0f' % k] = []
    dev_list_z['%1.0f' % k] = []
    dev_upper_limit_list['%1.0f' % k] = []
    mean_dev['%1.0f' % k] = []
    mean_dev_x['%1.0f' % k] = []
    mean_dev_y['%1.0f' % k] = []
    mean_dev_z['%1.0f' % k] = [] 
    mean_dev_std['%1.0f' % k] = []
    mean_list['%1.0f' % k] = []
    

save_images = question('save images(y) / or not (n)')
plot_histograms = question('histograms(y) / or not (n)')
calc_deviations = question('calc deviations(y) / or not (n)')

# for common structures add mean to array for both clicker_1 and clicker_2
for i in list_1:
    py_array_clicker_1 = np.load(os.path.join(clicker_1_folder,i))
    py_array_clicker_2 = np.load(os.path.join(clicker_2_folder,i))
    for k in landmarks:
        #com_list['%1.0f' % k].append(5)
        #print(com_structure_np(py_array,k)[0])
        com_list_clicker_1['%1.0f' % k].append(com_structure_np(py_array_clicker_1,k)[0])
        com_list_clicker_2['%1.0f' % k].append(com_structure_np(py_array_clicker_2,k)[0])
        
# calculate mean of clicker_1 and clicker_2 from arrays
for j in range(len(list_1)):
    for k in landmarks:
        mean_x = ((com_list_clicker_1['%1.0f' % k][j][2] + com_list_clicker_2['%1.0f' % k][j][2])/2)
        mean_y = ((com_list_clicker_1['%1.0f' % k][j][1] + com_list_clicker_2['%1.0f' % k][j][1])/2)
        mean_z = ((com_list_clicker_1['%1.0f' % k][j][0] + com_list_clicker_2['%1.0f' % k][j][0])/2)
        coords = [mean_x, mean_y, mean_z]
        mean_list['%1.0f' % k].append(coords)
    
click_outlier_counter = 0
if calc_deviations == True:
    # calculate deviation of arrays etc.
    for j in range(len(list_1)):
        for k in landmarks:
            x_mm, y_mm, z_mm = pixel_to_mm(list_1[j])
            dev_x = (com_list_clicker_1['%1.0f' % k][j][2] - com_list_clicker_2['%1.0f' % k][j][2])*(x_mm)
            dev_y = (com_list_clicker_1['%1.0f' % k][j][1] - com_list_clicker_2['%1.0f' % k][j][1])*(y_mm)
            dev_z = (com_list_clicker_1['%1.0f' % k][j][0] - com_list_clicker_2['%1.0f' % k][j][0])*(z_mm)
            dev = math.sqrt(abs(dev_x)**2 + abs(dev_y)**2 + abs(dev_z)**2)
            dev_list['%1.0f' % k].append(dev)
            dev_list_x['%1.0f' % k].append(dev_x)
            dev_list_y['%1.0f' % k].append(dev_y)
            dev_list_z['%1.0f' % k].append(dev_z)
            if dev > limit:
                dev_upper_limit_list['%1.0f' % k].append(list_1[j])
                print('image: %s' % list_1[j])
                print('landmark: %1.0f' % k)
                print(dev)
                click_outlier_counter += 1
                print('------------')
                
    
    # average deviation per landmark
    for k in landmarks:
        a = dev_list['%1.0f' % k]
        mean = np.mean(a)
        mean_dev['%1.0f' % k].append(mean)
        mean_dev_std['%1.0f' % k].append(np.std(a)*(len(list_1)**-0.5))
        # for x y and z
        x = dev_list_x['%1.0f' % k]
        y = dev_list_y['%1.0f' % k]
        z = dev_list_z['%1.0f' % k]
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)
        mean_dev_x['%1.0f' % k].append(mean_x)
        mean_dev_y['%1.0f' % k].append(mean_y)
        mean_dev_z['%1.0f' % k].append(mean_z)
        
        if plot_histograms == True:
      
            # plot and save histogram
            histogram(x, 'x', k)
            histogram(y, 'y', k)
            histogram(z, 'z', k)
        
    
        
# ------ return a structure with just one point at the mean  ----- #

if save_images == True:
    
    # make folders
    try:  
        os.mkdir(save_structure_folder)  
        os.mkdir(save_ct_folder)  
    except OSError as error:  
        print(error) 
  
    # for each image create an array
    for i in list_1:
        py_array_load = np.load(os.path.join(clicker_1_folder,i))
        ct = np.load(os.path.join(load_ct_folder,i))
        structure = np.empty(py_array_load.shape)
        # add number at location of x y z for each landmark
        index = list_1.index(i)
        for k in landmarks:
            z = int(mean_list['%1.0f' % k][index][0])
            y = int(mean_list['%1.0f' % k][index][1])
            x = int(mean_list['%1.0f' % k][index][2])
            # create small block
            for a in range(5):
                for b in range(5):
                    for c in range(5):
                        structure[z-2+c][y-2+b][x-2+a] = k # z y x
        # save image
        np.save(os.path.join(save_structure_folder,i), structure)
        # save ct
        np.save(os.path.join(save_ct_folder,i), ct)
        
    
    
    # create line instead of one average point
''' 
    # for each image create an array
    for i in list_1:
        py_array_load = np.load(os.path.join(clicker_1_folder,i))
        ct = np.load(os.path.join(load_ct_folder,i))
        structure = np.empty(py_array_load.shape)
        # add number at location of x y z for each landmark
        index = list_1.index(i)
        for k in landmarks:
            z_clicker_1 = int(com_list_clicker_1['%1.0f' % k][index][0])
            y_clicker_1 = int(com_list_clicker_1['%1.0f' % k][index][1])
            x_clicker_1 = int(com_list_clicker_1['%1.0f' % k][index][2])
            z_clicker_2 = int(com_list_clicker_2['%1.0f' % k][index][0])
            y_clicker_2 = int(com_list_clicker_2['%1.0f' % k][index][1])
            x_clicker_2 = int(com_list_clicker_2['%1.0f' % k][index][2]) 
            line = line_nd((z_clicker_1, y_clicker_1, x_clicker_1), (z_clicker_2, y_clicker_2, x_clicker_2), endpoint=True)
            # make line thicker
            line_mod = line
            for a in range(5):
               for b in range(5):
                   for c in range(5):
                       x = np.concatenate((line_mod[2],line[2] + a - 2))
                       y = np.concatenate((line_mod[1],line[1] + b - 2))
                       z = np.concatenate((line_mod[0],line[0] + c - 2))
                       line_mod = (z,y,x)                     
            # clip line between x,y,z
            line_mod[0] = np.clip(line_mod[0],min(z_clicker_1,z_clicker_2), max(z_clicker_1,z_clicker_2)) # clip z values between min and max of clicker_1/clicker_2
            line_mod[1] = np.clip(line_mod[1],min(y_clicker_1,y_clicker_2), max(y_clicker_1,y_clicker_2))
            line_mod[2] = np.clip(line_mod[2],min(x_clicker_1,x_clicker_2), max(x_clicker_1,x_clicker_2))
            #x = [ (line_mod[2] < py_array_load.shape[2]) & (line_mod[2] > 0 ]
            #y = [(line_mod[1] < py_array_load.shape[1]) & (line_mod[1] > 0 ] 
            #z = [(line_mod[0] < py_array_load.shape[0]) & (line_mod[0] > 0 ]
            #line_mod = (z,y,x)
            #line_mod[0] = np.clip(line_mod[0], )
            structure[line_mod] = k
        # save image
        np.save(os.path.join(save_structure_folder,i), structure)
        # save ct
        np.save(os.path.join(save_ct_folder,i), ct)
'''

# deviations per landmark per image
histogram_plain(dev_list, 'all devs for all clicks')

# for each landmark, list of the images with deviations greater than ceratin distance
print('for each landmark, list of the images with deviations greater than %1.0f' % limit)
print(dev_upper_limit_list)
print('\n')

print('percentage of clicks which are outliers')
print(100*click_outlier_counter/(len(list_1)*len(landmarks)))
            
# mean deviation per landmark
print('mean deviation per landmark')
print(mean_dev)
print('\n')

# std of mean deviation per landmark
print('std of mean deviation per landmark')
print(mean_dev_std)
print('\n')

print('x mean dev')
print(mean_dev_x)
print('\n')

print('y mean dev')
print(mean_dev_y)
print('\n')

print('z mean dev')
print(mean_dev_z)
print('\n')
            
print(' deviations are clicker_1 - clicker_2')