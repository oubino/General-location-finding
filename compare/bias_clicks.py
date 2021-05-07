import numpy as np
import os
import math
import csv
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#sns.set_theme(style="darkgrid")


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

clicker_1 = 'Oli_test_set'
clicker_2 = 'Aaron_test_set'
clicker_ab = 'Abby_test_set'



# load in pickle file
file_clicker_1 = load_obj(root, 'coords_' + clicker_1)
file_clicker_2 = load_obj(root, 'coords_' + clicker_2)
file_clicker_Ab = load_obj(root, 'coords_' + clicker_ab)


patients_clicker_1 = list(file_clicker_1.keys())
patients_clicker_2 = list(file_clicker_2.keys())
patients_clicker_ab = list(file_clicker_Ab.keys())

# landmarks
landmarks = [1,2,3,4,5,6,7,8,9,10]


#print(file_clicker_Ab)
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
    

#calc_deviations = question('calc deviations(y) / or not (n)')

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


pat_list = [item.replace('.npy', '') for item in pat_list]
#print(pat_list)


d_x = []
d_y = []
d_z = []
d = []
for p in pat_list:
    for l in landmarks:
        for j in range(len(pat_list)):
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list[j])
            dev_x_o = (com_list_clicker_1['%1.0f' % l][j][2] - com_list_clicker_ab['%1.0f' % l][j][2])*(x_mm)
            dev_y_o = (com_list_clicker_1['%1.0f' % l][j][1] - com_list_clicker_ab['%1.0f' % l][j][1])*(y_mm)
            dev_z_o = (com_list_clicker_1['%1.0f' % l][j][0] - com_list_clicker_ab['%1.0f' % l][j][0])*(z_mm)
            
            dev_x_a = (com_list_clicker_2['%1.0f' % l][j][2] - com_list_clicker_ab['%1.0f' % l][j][2])*(x_mm)
            dev_y_a = (com_list_clicker_2['%1.0f' % l][j][1] - com_list_clicker_ab['%1.0f' % l][j][1])*(y_mm)
            dev_z_a = (com_list_clicker_2['%1.0f' % l][j][0] - com_list_clicker_ab['%1.0f' % l][j][0])*(z_mm)
            
        #x_dev = [p, dev_x_o, dev_x_a, l]
        #y_dev = [p, dev_y_o, dev_y_a, l]
        #z_dev = [p, dev_z_o, dev_z_a, l]
        devs = [p,l,dev_x_a,dev_x_o, dev_y_a, dev_y_o, dev_z_a, dev_z_o]
        #d_x.append(x_dev)    
        #d_y.append(y_dev)
        #d_z.append(z_dev)
        d.append(devs)



df_d = pd.DataFrame(d, columns=('Patient', 'Landmark', 'A_x', 'O_x', 'A_y', 'O_y', 'A_z', 'O_z'))
print(df_d) 
print(df_d.shape)   

#sns_plot = sns.lmplot(data=df_d, x = 'A_x',y = 'O_x', hue ='Patient', scatter = True, fit_reg=False, markers=True)  
sns_plot = sns.relplot(x = 'O_x', y = 'A_x', hue = 'Patient', style = 'Landmark', data=df_d, s=75)
plt.xlabel('Oli Deviations')
plt.ylabel('Aaron Deviations')
plt.title("Deviations from Abby's clicks in x-axis")
plt.savefig('bias_output_x.png', bbox_inches='tight', dpi=300)
print(os.getcwd())



'''
#print(d_x)        
df_x = pd.DataFrame(data=d_x, columns=('Patient', 'Oli', 'Aaron', 'Landmark'))
print(df_x)
print(df_x.shape)
sns_plot_x = sns.relplot(x = 'Oli', y = 'Aaron', hue = 'Patient', style = 'Landmark', data=df_x, s=75)
#sns_plot_x.set_size_inches(18.5, 10.5)
plt.xlabel('Oli Deviations')
plt.ylabel('Aaron Deviations')
plt.title("Deviations from Abby's clicks in x-axis")
plt.savefig('bias_output_x.png', bbox_inches='tight', dpi=300)

'''
'''
df_y = pd.DataFrame(data=d_y, columns=('Patient', 'Landmark', 'Oli', 'Aaron'))
print(df_y)
df_z = pd.DataFrame(data=d_z, columns=('Patient', 'Landmark', 'Oli', 'Aaron'))
print(df_z)


# plot



sns_plot_y = sns.relplot(x = 'Oli', y = 'Aaron', hue = 'Patient', style = 'Landmark', data=df_y)
plt.savefig('bias_output_y.png', dpi=300)

sns_plot_z = sns.relplot(x = 'Oli', y = 'Aaron', hue = 'Patient', style = 'Landmark', data=df_z)
plt.savefig('bias_output_z.png', dpi=300)
'''