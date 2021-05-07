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
struc_list = [21,22,23,24,25,26]
coordinates = {}
clickers = ['aaron', 'oli']


d_x = []
d_y = {'A':0,'O':0, 'L':0}
d_z = {'A':0,'O':0, 'L':0}
#print(df_x)

for i in struc_list:
    coordinates[i] = {} # each patient has dictioanary
    #d_x[0][i], d_x[1][i], d_x[2][i] = {}, {}, {}
    
    for l in landmarks_total:
         coordinates[i][l] ={}
         x = float(np.random.random_integers(-10, 10 +1))
         z = float(np.random.random_integers(-10, 10+1))
         y = float(np.random.random_integers(-10, 10+1))
         
         
         m = float(np.random.random_integers(-10,10+1))
         p = float(np.random.random_integers(-10, 10+1))
         o = float(np.random.random_integers(-10, 10+1))
         
         
         x_dev = [i,x,m,l]
         #print(x_dev)
         d_x.append(x_dev)
         
       # for n in clickers:
        #    coordinates[i][l][n] = {'x':0, 'y':0, 'z':0}
            #d_y[n] = {}
            #df_y = pd.DataFrame(data=d_y)
         
            #print(d_x)
#print(d_x)
df = pd.DataFrame(d_x, columns=('P','A', 'O', 'L'))
print(df)                 
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

'''
sns_plot = sns.relplot(x = 'A', y = 'O', hue= 'P', style = 'L', data=df)
sns_plot.savefig('bias_output.png')


  
'''
 x = float(np.random.random_integers(10))
 z = float(np.random.random_integers(10))
 y = float(np.random.random_integers(10))
 
 
 m = float(np.random.random_integers(10))
 p = float(np.random.random_integers(10))
 o = float(np.random.random_integers(10))
 
# print(coordinates)
# for n in clickers:  
 if n == 'oli':
 coordinates[i][l][n]['x'], coordinates[i][l][n]['y'], coordinates[i][l][n]['z'] = m,n,o
 #d_x['A']['O']['L'] = x, y, l
 #d_y[1][p] = x  
  #   d_y[n][n][l].append(x)
  
 
 else:
 coordinates[i][l][n]['x'], coordinates[i][l][n]['y'], coordinates[i][l][n]['z'] = o,p,m
  '''  