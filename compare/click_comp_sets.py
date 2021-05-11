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

absent_keys = [x for x in patients_clicker_o_1 if x not in patients_clicker_o_2]
for k in absent_keys:
    file_clicker_o_1.pop(k)
    file_clicker_a_1.pop(k)
    
absent_keys = [x for x in patients_clicker_o_2 if x not in patients_clicker_o_1]
for k in absent_keys:
    file_clicker_o_2.pop(k)
    file_clicker_a_2.pop(k)
    
patients_clicker_o_1 = list(file_clicker_o_1.keys())
patients_clicker_o_2 = list(file_clicker_o_2.keys())
patients_clicker_a_1 = list(file_clicker_a_1.keys())
patients_clicker_a_2 = list(file_clicker_a_2.keys())


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

inter = False

if inter == True:
    for l in landmarks:
        L = lm[l-1]
        for j in range(len(pat_list_old)):
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list_old[j])
            dev_x_old = (com_list_clicker_o_1['%1.0f' % l][j][2] - com_list_clicker_a_1['%1.0f' % l][j][2])*(x_mm)
            dev_y_old = (com_list_clicker_o_1['%1.0f' % l][j][1] - com_list_clicker_a_1['%1.0f' % l][j][1])*(y_mm)
            dev_z_old = (com_list_clicker_o_1['%1.0f' % l][j][0] - com_list_clicker_a_1['%1.0f' % l][j][0])*(z_mm)
    
            dev_old = math.sqrt(abs(dev_x_old)**2 + abs(dev_y_old)**2 + abs(dev_z_old)**2)
        
            i = int(j)
        
            devs_old = [i, L, dev_old]
            d_old.append(devs_old)
           
    dev=[]
    
    for l in landmarks:
        L = lm[l-1]
        for j in range(len(pat_list_new)):
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list_new[j])
            dev_x_new = (com_list_clicker_o_2['%1.0f' % l][j][2] - com_list_clicker_a_2['%1.0f' % l][j][2])*(x_mm)
            dev_y_new = (com_list_clicker_o_2['%1.0f' % l][j][1] - com_list_clicker_a_2['%1.0f' % l][j][1])*(y_mm)
            dev_z_new = (com_list_clicker_o_2['%1.0f' % l][j][0] - com_list_clicker_a_2['%1.0f' % l][j][0])*(z_mm)
    
            dev_new = math.sqrt(abs(dev_x_new)**2 + abs(dev_y_new)**2 + abs(dev_z_new)**2)
        
            i = int(j)
            
            devs_new = [i, L, dev_new]
            d_new.append(devs_new)

elif inter == False:
    for l in landmarks:
        L = lm[l-1]
        for j in range(len(pat_list_old)):
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list_old[j])
            dev_x_old = (com_list_clicker_o_1['%1.0f' % l][j][2] - com_list_clicker_o_2['%1.0f' % l][j][2])*(x_mm)
            dev_y_old = (com_list_clicker_o_1['%1.0f' % l][j][1] - com_list_clicker_o_2['%1.0f' % l][j][1])*(y_mm)
            dev_z_old = (com_list_clicker_o_1['%1.0f' % l][j][0] - com_list_clicker_o_2['%1.0f' % l][j][0])*(z_mm)
    
            dev_old = math.sqrt(abs(dev_x_old)**2 + abs(dev_y_old)**2 + abs(dev_z_old)**2)
        
            i = int(j)
        
            devs_old = [i, L, dev_old]
            d_old.append(devs_old)
           
    dev=[]
    
    for l in landmarks:
        L = lm[l-1]
        for j in range(len(pat_list_new)):
            z_mm, y_mm, x_mm = pixel_to_mm(pat_list_new[j])
            dev_x_new = (com_list_clicker_a_1['%1.0f' % l][j][2] - com_list_clicker_a_2['%1.0f' % l][j][2])*(x_mm)
            dev_y_new = (com_list_clicker_a_1['%1.0f' % l][j][1] - com_list_clicker_a_2['%1.0f' % l][j][1])*(y_mm)
            dev_z_new = (com_list_clicker_a_1['%1.0f' % l][j][0] - com_list_clicker_a_2['%1.0f' % l][j][0])*(z_mm)
    
            dev_new = math.sqrt(abs(dev_x_new)**2 + abs(dev_y_new)**2 + abs(dev_z_new)**2)
        
            i = int(j)
            
            devs_new = [i, L, dev_new]
            d_new.append(devs_new)
    

df_d_old = pd.DataFrame(d_old, columns=('Patient', 'Landmark','Deviation'))
df_d_new = pd.DataFrame(d_new,  columns=('Patient', 'Landmark','Deviation'))
       
if inter == True:
    df_d_new['Annotations'] = 'Revised'
    df_d_old['Annotations'] = 'Initial'
if inter == False:
    df_d_new['Annotator'] = 'Aaron'
    df_d_old['Annotator'] = 'Oli'
    
df_d = pd.concat([df_d_old,df_d_new])

medians_old = df_d_old.groupby(['Landmark'])['Deviation'].median()
vertical_offset_old = df_d_old['Deviation'].median() * 0.08 # offset from median for display

medians_new = df_d_new.groupby(['Landmark'])['Deviation'].median()
vertical_offset_new = df_d_new['Deviation'].median() * 0.08 # offset from median for display

means_new = df_d_new.groupby(['Landmark'])['Deviation'].mean()
means_old = df_d_old.groupby(['Landmark'])['Deviation'].mean()

#hand =  ['Initial', 'Revised']
plt.figure(figsize=(8,6))
if inter == True:       
    box_plot = sns.boxplot(x = 'Landmark', y = 'Deviation', hue = 'Annotations', palette = 'Set1', data=df_d, showfliers=False)
elif inter == False:
    box_plot = sns.boxplot(x = 'Landmark', y = 'Deviation', hue = 'Annotator', palette = 'Set1', data=df_d, showfliers=False)

#sns_plot_new = sns.boxplot(x = 'Landmark', y = 'Revised',data=df_d_new, showfliers=False)
#plt.legend(bbox_to_anchor=(1.05, 1), handles = [sns_plot_old, sns_plot_new], loc='upper left', borderaxespad=0.)
#plt.xlabel('Landmarks')
plt.ylabel('Deviations (mm)')
bottom, top = plt.ylim()
plt.ylim(bottom, 33)

medians_old = round(medians_old,1)
medians_new = round(medians_new,1)

print(means_old)
print(means_new)

means_old = means_old.reindex(index=lm)
means_new = means_new.reindex(index=lm)
medians_old = medians_old.reindex(index=lm)
medians_new = medians_new.reindex(index=lm)

if inter == True:
    for xtick in box_plot.get_xticks():
        if xtick == 1:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 1,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 2:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 0.5,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 0.5,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 3:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 0.5,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 0.5,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 4:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 6:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 0.5,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 7:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 8:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 0.5,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 0,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 9:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 0.5,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new + 0.2,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        else:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)

elif inter == False:
    for xtick in box_plot.get_xticks():
        
        if xtick == 1:
            box_plot.text(xtick - 0.08,medians_old[xtick] + vertical_offset_old + 0.1,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 90)
            box_plot.text(xtick + 0.08,medians_new[xtick] + vertical_offset_new + 0.1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 90)
        elif xtick == 2:
            box_plot.text(xtick - 0.04,medians_old[xtick] + vertical_offset_old - 0.05,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.01,medians_new[xtick] + vertical_offset_new - 1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 3:
            box_plot.text(xtick - 0.03,medians_old[xtick] + vertical_offset_old - 0.05,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.03,medians_new[xtick] + vertical_offset_new - 1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 4:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 5:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 6:
            box_plot.text(xtick - 0.07,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 90)
            box_plot.text(xtick + 0.08,medians_new[xtick] + vertical_offset_new - 1,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 90)
        elif xtick == 7:
            box_plot.text(xtick - 0.07,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 90)
            box_plot.text(xtick + 0.09,medians_new[xtick] + vertical_offset_new - 1.5,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 90)
        elif xtick == 8:
            box_plot.text(xtick - 0.02,medians_old[xtick] + vertical_offset_old - 1,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.02,medians_new[xtick] + vertical_offset_new - 0.6,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        elif xtick == 9:
            box_plot.text(xtick - 0.03,medians_old[xtick] + vertical_offset_old+0.3,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.03,medians_new[xtick] + vertical_offset_new ,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)
        else:
            box_plot.text(xtick - 0.01,medians_old[xtick] + vertical_offset_old,medians_old[xtick], 
                    horizontalalignment='right',size=12,color='pink', weight='semibold', rotation = 0)
            box_plot.text(xtick + 0.01,medians_new[xtick] + vertical_offset_new,medians_new[xtick], 
                horizontalalignment='left',size=12,color='pink', weight='semibold', rotation = 0)


#plt.title("Inter-observer variation between initial and revised datasets")
#sns_plot_x.fig.subplots_adjust(top=1)
box_plot.set(xlabel=None)
if inter == True:
    plt.savefig('inter_observer.png', bbox_inches='tight', dpi=600)
elif inter == False:
    plt.savefig('intra_observer.png', bbox_inches='tight', dpi=600)





