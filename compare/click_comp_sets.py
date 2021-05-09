import numpy as np
import os
import math
import csv
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")


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


def load_obj(root, name):
    with open(os.path.join(root, name) + '.pkl', 'rb') as f:       
        return pickle.load(f)

# paths
root = r'C:\Users\ranki_252uikw\Documents\MPhysS2'

clicker_o_1 = 'Oli'
clicker_o_2 = 'Oli_new'
clicker_a_1 = 'Aaron'
clicker_a_2 = 'Aaron_new'


# load in pickle file
file_clicker_o_1 = load_obj(root, 'coords_' + clicker_o_1)
file_clicker_o_2 = load_obj(root, 'coords_' + clicker_o_2)
file_clicker_a_1 = load_obj(root, 'coords_' + clicker_a_1)
file_clicker_a_2 = load_obj(root, 'coords_' + clicker_a_2)


patients_clicker_o_1 = list(file_clicker_o_1.keys())
patients_clicker_o_2 = list(file_clicker_o_2.keys())
patients_clicker_a_1 = list(file_clicker_a_1.keys())
patients_clicker_a_2 = list(file_clicker_a_2.keys())
# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]


#print(file_clicker_Ab)
# common patients
pat_list_old = [x for x in patients_clicker_o_1 if x in patients_clicker_a_1]
pat_list_new = [x for x in patients_clicker_o_2 if x in patients_clicker_a_2]
com_list_clicker_o_1 = {}
com_list_clicker_o_2 = {}
com_list_clicker_a_1 = {}
com_list_clicker_a_2 = {}
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
    com_list_clicker_o_1['%1.0f' % k] = []
    com_list_clicker_o_2['%1.0f' % k] = []
    com_list_clicker_a_1['%1.0f' % k] = []
    com_list_clicker_a_2['%1.0f' % k] = []
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
    

#calc_deviations = question('calc deviations(y) / or not (n)')

# for common structures add mean to array for both clicker_1 and clicker_2
for p in pat_list_old:
    for k in landmarks:
        com_list_clicker_o_1['%1.0f' % k].append([file_clicker_o_1[p][k]['z'], file_clicker_o_1[p][k]['y'], file_clicker_o_1[p][k]['x']])
        com_list_clicker_a_1['%1.0f' % k].append([file_clicker_a_1[p][k]['z'], file_clicker_a_1[p][k]['y'], file_clicker_a_1[p][k]['x']])
        
for p in pat_list_new:
    for k in landmarks:
        com_list_clicker_o_2['%1.0f' % k].append([file_clicker_o_2[p][k]['z'], file_clicker_o_2[p][k]['y'], file_clicker_o_2[p][k]['x']])
        com_list_clicker_a_2['%1.0f' % k].append([file_clicker_a_2[p][k]['z'], file_clicker_a_2[p][k]['y'], file_clicker_a_2[p][k]['x']])
# calculate mean of clicker_1 and clicker_2 from arrays

#for j in range(len(pat_list)):
 #   for k in landmarks:
  #      mean_x = ((com_list_clicker_1['%1.0f' % k][j][2] + com_list_clicker_2['%1.0f' % k][j][2])/2)
   #     mean_y = ((com_list_clicker_1['%1.0f' % k][j][1] + com_list_clicker_2['%1.0f' % k][j][1])/2)
    #    mean_z = ((com_list_clicker_1['%1.0f' % k][j][0] + com_list_clicker_2['%1.0f' % k][j][0])/2)
     #   coords = [mean_z, mean_y, mean_x]
      #  mean_list['%1.0f' % k].append(coords)


pat_list_old = [item.replace('.npy', '') for item in pat_list_old]
pat_list_new = [item.replace('.npy', '') for item in pat_list_new]

#pat_list = np.asarray(pat_list, dtype=np.float64, order='C')

#print(pat_list)

lm = ['AMl', 'AMr', 'HMl', 'HMr', 'FZl', 'FZr', 'FNl', 'FNr', 'SOl', 'SOr']
d_x = []
d_y = []
d_z = []
d_old = []
d_new = []
l_old = [0.7, 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7] 
l_new = [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3]

for l in landmarks:
    for j in range(len(pat_list_old)):
        z_mm, y_mm, x_mm = pixel_to_mm(pat_list_old[j])
        dev_x_old = (com_list_clicker_o_1['%1.0f' % l][j][2] - com_list_clicker_a_1['%1.0f' % l][j][2])*(x_mm)
        dev_y_old = (com_list_clicker_o_1['%1.0f' % l][j][1] - com_list_clicker_a_1['%1.0f' % l][j][1])*(y_mm)
        dev_z_old = (com_list_clicker_o_1['%1.0f' % l][j][0] - com_list_clicker_a_1['%1.0f' % l][j][0])*(z_mm)

        dev_old = math.sqrt(abs(dev_x_old)**2 + abs(dev_y_old)**2 + abs(dev_z_old)**2)
    
        i = int(j)
    
        devs_old = [i, l, dev_old]
        d_old.append(devs_old)
       
      

#print(d_old)
df_d_old = pd.DataFrame(d_old, columns=('Patient', 'Landmark','Initial'))
print(df_d_old)

dev=[]

for j in range(len(pat_list_new)):
    for l in landmarks:
        z_mm, y_mm, x_mm = pixel_to_mm(pat_list_new[j])
        dev_x_new = (com_list_clicker_o_2['%1.0f' % l][j][2] - com_list_clicker_a_2['%1.0f' % l][j][2])*(x_mm)
        dev_y_new = (com_list_clicker_o_2['%1.0f' % l][j][1] - com_list_clicker_a_2['%1.0f' % l][j][1])*(y_mm)
        dev_z_new = (com_list_clicker_o_2['%1.0f' % l][j][0] - com_list_clicker_a_2['%1.0f' % l][j][0])*(z_mm)

        dev_new = math.sqrt(abs(dev_x_new)**2 + abs(dev_y_new)**2 + abs(dev_z_new)**2)
    
        i = int(j)
        
        #devs_new = [i, l, dev_new]
    #d_x.append(x_dev)    
    #d_y.append(y_dev)
    #d_z.append(z_dev)
        #d_new.append(devs_new)
        
        if j < len(pat_list_old):
            dev_x_old = (com_list_clicker_o_1['%1.0f' % l][j][2] - com_list_clicker_a_1['%1.0f' % l][j][2])*(x_mm)
            dev_y_old = (com_list_clicker_o_1['%1.0f' % l][j][1] - com_list_clicker_a_1['%1.0f' % l][j][1])*(y_mm)
            dev_z_old = (com_list_clicker_o_1['%1.0f' % l][j][0] - com_list_clicker_a_1['%1.0f' % l][j][0])*(z_mm)
    
            dev_old = math.sqrt(abs(dev_x_old)**2 + abs(dev_y_old)**2 + abs(dev_z_old)**2)
        else: 
            NaN = np.nan
            dev_old = NaN
        
        d = [i, l, dev_old, dev_new]   
        dev.append(d)
       
     
tips = sns.load_dataset("tips")
print(tips)
#print(d_old)
df_d = pd.DataFrame(dev, columns=('Patient', 'Landmark','Initial', 'Revised'))
print(df_d)
print(np.shape(d_new))
df_d_new = pd.DataFrame(d_new, columns=('Patient', 'Landmark','Revised'))
print(df_d_new)


hand =  ['Initial', 'Revised']
plt.figure(figsize=(5,4))
sns_plot_old = sns.boxplot(x = 'Landmark', y = 'Initial',data=df_d, showfliers=False)
#sns_plot_new = sns.boxplot(x = 'Landmark', y = 'Revised',data=df_d_new, color='b', showfliers=False, label = 'Revised dataset')
#plt.legend(bbox_to_anchor=(1.05, 1), handles = [sns_plot_old, sns_plot_new], loc='upper left', borderaxespad=0.)
plt.xlabel('Landmarks')
plt.ylabel('Deviations (mm)')

plt.title("Inter-observer variation between initial and revised datasets")
#sns_plot_x.fig.subplots_adjust(top=1)
plt.savefig('variation_output.png', bbox_inches='tight', dpi=300)





