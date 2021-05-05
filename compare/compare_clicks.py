import numpy as np
import os
import math
import csv
import pickle
import matplotlib.pyplot as plt

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
csv_root = r'/home/data/paed_dataset/test'
root = r'/home/oli/data/results/oli/run_folder/eval_100_3'

#clicker_1 = input('Clicker_1, (e.g. Oli_test_set): ') 
#clicker_2 = input('Clicker_2, (e.g. Aaron_test_set): ') 

hist_root = r'/home/rankinaaron98/data/Compare_aaron/Histograms_reclick__oli_aaron_testsets/'

# load in pickle file
file_clicker_1 = load_obj(root, 'final_coords_no_struc')
file_clicker_2 = load_obj(root, 'final_coords_no_struc_2')

patients_clicker_1 = list(file_clicker_1.keys())
patients_clicker_2 = list(file_clicker_2.keys())

# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]

# limit
limit = 20

# common patients
pat_list = [x for x in patients_clicker_1 if x in patients_clicker_2]

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
    

plot_histograms = question('histograms(y) / or not (n)')
calc_deviations = question('calc deviations(y) / or not (n)')

# for common structures add mean to array for both clicker_1 and clicker_2
for p in pat_list:
    for k in landmarks:
        com_list_clicker_1['%1.0f' % k].append([file_clicker_1[p][k]['z'], file_clicker_1[p][k]['y'], file_clicker_1[p][k]['x']])
        com_list_clicker_2['%1.0f' % k].append([file_clicker_2[p][k]['z'], file_clicker_2[p][k]['y'], file_clicker_2[p][k]['x']])
        
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
latex_line_x = []
latex_line_y = []
latex_line_z = []
csv_line = []
name_of_file = os.path.join(root, "paed_struc_compare.txt")
txt_file = open(name_of_file, "a")    
click_outlier_counter = 0


if calc_deviations == True:
    # calculate deviation of arrays etc.
    for j in range(len(pat_list)):
        for k in landmarks:
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list[j])
            dev_x = (com_list_clicker_1['%1.0f' % k][j][2] - com_list_clicker_2['%1.0f' % k][j][2])*(x_mm)
            dev_y = (com_list_clicker_1['%1.0f' % k][j][1] - com_list_clicker_2['%1.0f' % k][j][1])*(y_mm)
            dev_z = (com_list_clicker_1['%1.0f' % k][j][0] - com_list_clicker_2['%1.0f' % k][j][0])*(z_mm)
            dev = math.sqrt(abs(dev_x)**2 + abs(dev_y)**2 + abs(dev_z)**2)
            dev_list['%1.0f' % k].append(dev)
            dev_list_x['%1.0f' % k].append(dev_x)
            dev_list_y['%1.0f' % k].append(dev_y)
            dev_list_z['%1.0f' % k].append(dev_z)
            if dev > limit:
                dev_upper_limit_list['%1.0f' % k].append(pat_list[j])
                print('image: %s' % pat_list[j])
                print('landmark: %1.0f' % k)
                print(dev)
                click_outlier_counter += 1
                print('------------')
                
    
    # average deviation per landmark
    for k in landmarks:
        a = dev_list['%1.0f' % k]
        mean = np.mean(a)
        mean_dev['%1.0f' % k].append(mean)
        mean_dev_std['%1.0f' % k].append(np.std(a)*(len(pat_list)**-0.5))
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
            histogram(a, 'total', k)
            histogram(x, 'x', k)
            histogram(y, 'y', k)
            histogram(z, 'z', k)
        '''
        latex_line_landmark = ['landmark: ' + str(k)]
        latex_line_temp_mean = [' & ' + str(round(mean,1))] 
        latex_line_mean = latex_line_mean + latex_line_temp_mean    
        
        latex_line_temp_mean_std = [' & ' + str(round(mean_std,1))] 
        latex_line_mean_std = latex_line_mean_std + latex_line_temp_mean_std    
        
        latex_line_temp_x = [' & ' + str(round(mean_x,1))] 
        latex_line_x = latex_line_x + latex_line_temp_x    
        
        latex_line_temp_y = [' & ' + str(round(mean_y,1))] 
        latex_line_y = latex_line_y + latex_line_temp_y   
        
        latex_line_temp_z = [' & ' + str(round(mean_z,1))] 
        latex_line_z = latex_line_z + latex_line_temp_z    
        # write in excel format for easy to calc folds 
    
        txt_file.writelines(latex_line_landmark)
        txt_file.writelines(['\n'])
        txt_file.writelines(latex_line_mean)
        txt_file.writelines(['\n'])
        txt_file.writelines(latex_line_mean_std)
        txt_file.writelines(['\n'])
        txt_file.writelines(latex_line_x)
        txt_file.writelines(['\n'])
        txt_file.writelines(latex_line_y)
        txt_file.writelines(['\n'])
        txt_file.writelines(latex_line_z)
        txt_file.writelines(['\n'])
        txt_file.writelines(['\n'])
txt_file.close()        
    
'''
# for each landmark, list of the images with deviations greater than ceratin distance
print('for each landmark, list of the images with deviations greater than %1.0f' % limit)
print(dev_upper_limit_list)
print('\n')

print('percentage of clicks which are outliers')
print(100*click_outlier_counter/(len(pat_list)*len(landmarks)))




       
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