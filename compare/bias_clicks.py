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
#root = r'C:\Users\ranki_252uikw\Documents\MPhysS2'
root = r'C:\Users\olive\OneDrive\Documents\MPhys'

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

#pat_list = np.asarray(pat_list, dtype=np.float64, order='C')

#print(pat_list)

lm = ['AMl', 'AMr', 'HMl', 'HMr', 'FZl', 'FZr', 'FNl', 'FNr', 'SOl', 'SOr']
d_x = []
d_y = []
d_z = []
d = []
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
        i = int(j)
        
        devs = [ dev_x_a,dev_x_o, dev_y_a, dev_y_o, dev_z_a, dev_z_o, i, lm[j]]
    #d_x.append(x_dev)    
    #d_y.append(y_dev)
    #d_z.append(z_dev)
        d.append(devs)
       
        
o_x_max = np.amax(devs[1])
#print(o_x_max)
#print(d)
df_d = pd.DataFrame(d, columns=('A_x', 'O_x', 'A_y', 'O_y', 'A_z', 'O_z','Patient', 'Landmark'))
#print(df_d)  

# calculate mean deviation per landmark from mean of A and O
df_d['dev_x'] = df_d.apply(lambda row : (row['A_x']/2 + row['O_x']/2), axis = 1)
df_d['dev_y'] = df_d.apply(lambda row : (row['A_y']/2 + row['O_y']/2), axis = 1)
df_d['dev_z'] = df_d.apply(lambda row : (row['A_z']/2 + row['O_z']/2), axis = 1)
df_d['dev_total'] = df_d.apply(lambda row : (row['dev_x']**2 + row['dev_y']**2 + row['dev_z']**2)**0.5, axis = 1)
mean = df_d.groupby(['Landmark'])['dev_total'].mean()
median = df_d.groupby(['Landmark'])['dev_total'].median()
maxi = df_d.groupby(['Landmark'])['dev_total'].max()
mini = df_d.groupby(['Landmark'])['dev_total'].min()

print(mean)

median = round(median,2)

plt.figure(figsize=(8,6))
box_plot = sns.boxplot(x = 'Landmark', y = 'dev_total', palette = 'Set1', data=df_d, showfliers=False)
plt.ylabel('Deviations (mm)')
box_plot.set(xlabel=None)
for xtick in box_plot.get_xticks():
    box_plot.text(xtick - 0.02,median[xtick] + 0.2,median[xtick], 
                    horizontalalignment='center',size=12,color='black', weight='semibold', rotation = 0)
plt.savefig('abby_vs_our_mean.png', bbox_inches='tight', dpi=600)

# below are dx dy dz plots

"""
plt.figure(figsize=(3,3))
sns_plot_x = sns.lmplot(fit_reg=False,x = 'O_x', y = 'A_x' , hue = 'Landmark', legend='full', palette=('deep'),data=df_d)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim(-11,11)
plt.ylim(-11,11)
plt.xlabel('Oli deviations (mm)')
plt.ylabel('Aaron deviations (mm)')
plt.title("Deviations from Abby's clicks in x-axis")
sns_plot_x.fig.subplots_adjust(top=1)
plt.savefig('bias_output_x.png', bbox_inches='tight', dpi=300)


plt.figure(figsize=(3,3))
sns_plot_y = sns.lmplot(fit_reg=False, x = 'O_y', y = 'A_y' , hue = 'Landmark', legend='full', palette=('deep') ,data=df_d)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim(-11,11)
plt.ylim(-11,11)
plt.xlabel('Oli deviations (mm)')
plt.ylabel('Aaron deviations (mm)')
plt.title("Deviations from Abby's clicks in y-axis")
sns_plot_y.fig.subplots_adjust(top=1)
plt.savefig('bias_output_y.png',bbox_inches='tight', dpi=300)

plt.figure(figsize=(3,3))
sns_plot_z = sns.lmplot(fit_reg=False,x = 'O_z', y = 'A_z' , hue = 'Landmark', legend='full', palette=('deep') ,data=df_d)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim(-11,11)
plt.ylim(-11,11)
plt.xlabel('Oli deviations (mm)')
plt.ylabel('Aaron deviations (mm)')
plt.title("Deviations from Abby's clicks in z-axis")
sns_plot_z.fig.subplots_adjust(top=1)
plt.savefig('bias_output_z.png',bbox_inches='tight', dpi=300)

"""

