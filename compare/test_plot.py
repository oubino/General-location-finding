# -*- coding: utf-8 -*-
"""
Created on Fri May  7 01:11:34 2021

@author: ranki_252uikw
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  6 21:53:02 2021

@author: ranki_252uikw
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='darkgrid')


landmarks_total = [1,2,3,4,5,6,7,8,9,10]
landmarks_total_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7:'com',8:'com', 9:'com',10:'com', }
struc_list = ['21.npy','22.npy','23.npy','24.npy','25.npy','26.npy']
struc_list = [item.replace('.npy', '') for item in struc_list]
print(struc_list)

coordinates = {}
clickers = ['aaron', 'oli']


d_x = []
d_y = []
d_z = []
d = []
#print(df_x)

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    #d_x[0][i], d_x[1][i], d_x[2][i] = {}, {}, {}
    
    for l in landmarks_total:
         coordinates[i][l] ={}
         x = float(np.random.random_integers(-10, 10 + 1))
         z = float(np.random.random_integers(-10, 10 + 1))
         y = float(np.random.random_integers(-10, 10 + 1))
         
         
         m = float(np.random.random_integers(-10, 10 + 1))
         p = float(np.random.random_integers(-10, 10 + 1))
         o = float(np.random.random_integers(-10, 10 + 1))
         
         
         x_dev = [i,x,m,l]
         y_dev = [i,y,p,l]
         z_dev = [i,z,o,l]
         devs = [i,l,x,m,y,p,z,o]
         #print(x_dev)
         d_x.append(x_dev)
         d_y.append(y_dev)
         d_z.append(z_dev)
         d.append(devs)
         
       # for n in clickers:
        #    coordinates[i][l][n] = {'x':0, 'y':0, 'z':0}
            #d_y[n] = {}
            #df_y = pd.DataFrame(data=d_y)
d_d = pd.DataFrame(d, columns=('P', 'L', 'A_x', 'O_x', 'A_y', 'O_y', 'A_z', 'O_z'))
print(d_d) 
print(d_d.shape)   
sns_plot = sns.relplot(x = 'A_x',y = 'O_x', hue= 'P', style = 'L', data=d_d, s=75)   
plt.xlabel('Oli Deviations')
plt.ylabel('Aaron Deviations')
plt.title("Deviations from Abby's clicks in x-axis")
plt.savefig('test_output_x.png', bbox_inches='tight', dpi=300) 
#print(d_x)
#print(d_x)
#df_x = pd.DataFrame(d_x, columns=('P','A', 'O', 'L'))
#print(df_x)
#print(df_x.shape)
#df_y = pd.DataFrame(d_y, columns=('P','A', 'O', 'L'))
#print(df_y)
#df_z = pd.DataFrame(d_z, columns=('P','A', 'O', 'L'))
#print(df_z)
'''
df = pd.DataFrame(data=d_x, index=('A', 'O', 'L'))
print(df)

df_xt = df.T
print(df_xt) 
#print(df)
# plot
tips = sns.load_dataset("tips")
#print(tips)
#df_2 = pd.DataFrame(data=d_x)
#print(df_2)


sns_plot = sns.relplot(x = 'A', y = 'O', hue= 'P', style = 'L', data=df_x, s=75)
plt.xlabel('Oli Deviations')
plt.ylabel('Aaron Deviations')
plt.title("Deviations from Abby's clicks in x axis")
plt.savefig('bias_output_x.png', bbox_inches='tight', dpi=300)


#sns_plot_y = sns.relplot(x = 'A', y = 'O', hue= 'P', style = 'L', data=df_y)
#sns_plot.savefig('bias_output_y.png')

#sns_plot_z = sns.relplot(x = 'A', y = 'O', hue= 'P', style = 'L', data=df_z)
#sns_plot.savefig('bias_output_z.png')


  
'''
