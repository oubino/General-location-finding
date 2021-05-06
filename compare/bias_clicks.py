import numpy as np
import os
import math
import csv
import pickle
import matplotlib.pyplot as plt
from itertools import chain
def question(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

def pixel_to_mm(patient):
    data = csv.reader(open(os.path.join(root, 'image_dimensions.csv')),delimiter=',')
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
    
    return pixel_mm_z, pixel_mm_y, pixel_mm_x

def histogram(data, coord, landmark):
    # plot and save histogram
    data = np.array(data)
    data = np.sort(data)
    plt.figure()
    n, bins, patches = plt.hist(x=data, bins=list(range(-30,31)), color='#0504aa',
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
    plt.xticks(np.arange(-30,31,4))
    plt.savefig(hist_name)


def load_obj(root, name):
    with open(os.path.join(root, name) + '.pkl', 'rb') as f:       
        return pickle.load(f)

# paths
root = r'/home/olive/data/Facial_asymmetry_test_sets'

clicker_1 = 'Oli_test_set'
clicker_2 = 'Aaron_test_set'
clicker_ab = 'Abby_test_set'


hist_root = r'/home/rankinaaron98/data/Compare_aaron/Histograms_reclick__oli_aaron_testsets/'

# load in pickle file
file_clicker_1 = load_obj(root, 'coords_' + clicker_1)
file_clicker_2 = load_obj(root, 'coords_' + clicker_2)
file_clicker_Ab = load_obj(root, 'coords_' + clicker_ab)


patients_clicker_1 = list(file_clicker_1.keys())
patients_clicker_2 = list(file_clicker_2.keys())
patients_clicker_ab = list(file_clicker_Ab.keys())

# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]

# limit
limit = 20

# common patients
pat_list = [x for x in patients_clicker_1 if x in patients_clicker_2]

com_list_clicker_1 = {}
com_list_clicker_2 = {}
com_list_clicker_ab = {}
dev_list_o = {}
dev_list_a = {}
dev_list_x_o = {}
dev_list_x_a = {}
dev_list_y_o = {}
dev_list_y_a = {}
dev_list_z_o = {}
dev_list_z_a = {}
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
    com_list_clicker_ab['%1.0f' % k] = []
    dev_list_o['%1.0f' % k] = []
    dev_list_x_o['%1.0f' % k] = []
    dev_list_y_o['%1.0f' % k] = []
    dev_list_z_o['%1.0f' % k] = []
    dev_list_a['%1.0f' % k] = []
    dev_list_x_a['%1.0f' % k] = []
    dev_list_y_a['%1.0f' % k] = []
    dev_list_z_a['%1.0f' % k] = []
    dev_upper_limit_list['%1.0f' % k] = []
    mean_dev['%1.0f' % k] = []
    mean_dev_x['%1.0f' % k] = []
    mean_dev_y['%1.0f' % k] = []
    mean_dev_z['%1.0f' % k] = [] 
    mean_dev_std['%1.0f' % k] = []
    mean_list['%1.0f' % k] = []
    

calc_deviations = question('calc deviations(y) / or not (n)')

# for common structures add mean to array for both clicker_1 and clicker_2
for p in pat_list:
    for k in landmarks:
        com_list_clicker_1['%1.0f' % k].append([file_clicker_1[p][k]['z'], file_clicker_1[p][k]['y'], file_clicker_1[p][k]['x']])
        com_list_clicker_2['%1.0f' % k].append([file_clicker_2[p][k]['z'], file_clicker_2[p][k]['y'], file_clicker_2[p][k]['x']])
        com_list_clicker_ab['%1.0f' % k].append([file_clicker_Ab[p][k]['z'], file_clicker_Ab[p][k]['y'], file_clicker_Ab[p][k]['x']])
# calculate mean of clicker_1 and clicker_2 from arrays
for j in range(len(pat_list)):
    for k in landmarks:
        mean_x = ((com_list_clicker_1['%1.0f' % k][j][2] + com_list_clicker_2['%1.0f' % k][j][2])/2)
        mean_y = ((com_list_clicker_1['%1.0f' % k][j][1] + com_list_clicker_2['%1.0f' % k][j][1])/2)
        mean_z = ((com_list_clicker_1['%1.0f' % k][j][0] + com_list_clicker_2['%1.0f' % k][j][0])/2)
        coords = [mean_z, mean_y, mean_x]
        mean_list['%1.0f' % k].append(coords)



latex_line_mean = []
latex_line_mean_std = []
latex_line_x_o = []
latex_line_y_o = []
latex_line_z_o = []
latex_line_x_o = []
latex_line_y_o = []
latex_line_z_o = []
csv_line = []
name_of_file = os.path.join(hist_root, clicker_1 +"_" + clicker_2 + "_compare.txt")
txt_file = open(name_of_file, "a")    
click_outlier_counter = 0


if calc_deviations == True:
    # calculate deviation of arrays etc.
    for j in range(len(pat_list)):
        for k in landmarks:
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list[j])
            dev_x_o = (com_list_clicker_1['%1.0f' % k][j][2] - com_list_clicker_ab['%1.0f' % k][j][2])*(x_mm)
            dev_y_o = (com_list_clicker_1['%1.0f' % k][j][1] - com_list_clicker_ab['%1.0f' % k][j][1])*(y_mm)
            dev_z_o = (com_list_clicker_1['%1.0f' % k][j][0] - com_list_clicker_ab['%1.0f' % k][j][0])*(z_mm)
            dev_o = math.sqrt(abs(dev_x_o)**2 + abs(dev_y_o)**2 + abs(dev_z_o)**2)
            dev_list_o['%1.0f' % k].append(dev_o)
            dev_list_x_o['%1.0f' % k].append(dev_x_o)
            dev_list_y_o['%1.0f' % k].append(dev_y_o)
            dev_list_z_o['%1.0f' % k].append(dev_z_o)
            
            dev_x_a = (com_list_clicker_1['%1.0f' % k][j][2] - com_list_clicker_ab['%1.0f' % k][j][2])*(x_mm)
            dev_y_a = (com_list_clicker_1['%1.0f' % k][j][1] - com_list_clicker_ab['%1.0f' % k][j][1])*(y_mm)
            dev_z_a = (com_list_clicker_1['%1.0f' % k][j][0] - com_list_clicker_ab['%1.0f' % k][j][0])*(z_mm)
            dev_a = math.sqrt(abs(dev_x_a)**2 + abs(dev_y_a)**2 + abs(dev_z_a)**2)
            dev_list_a['%1.0f' % k].append(dev_a)
            dev_list_x_a['%1.0f' % k].append(dev_x_a)
            dev_list_y_a['%1.0f' % k].append(dev_y_a)
            dev_list_z_a['%1.0f' % k].append(dev_z_a)
            
      
#print('x dev: ' + str(dev_list_x_a))
#print('-------------------------')
#print('y dev: ' + str(dev_list_y_a))
#print('--------------------------')
#print('z dev: ' + str(dev_list_z_a))

oli_devs_axis = {}
aaron_devs_axis = {}

oli_devs_axis.update(dev_list_x_o)
oli_devs_axis.update(dev_list_y_o)
oli_devs_axis.update(dev_list_z_o)
oli_devs_axis = dict(chain.from_iterable(map(oli_devs_axis, dev_list_x_o, dev_list_y_o, dev_list_z_o)))
print(oli_devs_axis)

