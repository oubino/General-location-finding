# convert cts and structures from .nii to .py and save

import SimpleITK as sitk
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# input paths
nii_path = r'/home/oli/data/paed_structures'
nii_path_ct = r'/home/oli/data/paed_cts'

# output paths
struc_path =  r'/home/oli/data/paed_dataset/Structures'
ct_path = r'/home/oli/data/paed_dataset/CTs'
hist_root = r'/home/oli/data/paed_dataset/Histograms'

# make folders if not present
try:  
    os.mkdir(struc_path)  
    os.mkdir(ct_path)
   # os.mkdir(hist_root)
except OSError as error:  
    print(error) 

csv_dimensions_path = r'/home/oli/data/paed_dataset/image_dimensions.csv'

def histogram(ct, ct_numb):
    # plot and save histogram
    data = ct.flatten()
    data = np.sort(data)
    plt.figure()
    n, bins, patches = plt.hist(x=data, bins=1000, color='#0504aa',
                            alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title("pixel intensity for ct %s" % (ct_numb))
    #plt.text(23, 45, r'$\mu=15, b=3$')
    hist_name = os.path.join(hist_root, "%s" % (ct_numb))
    plt.yscale('log', nonposy = 'clip')
    plt.savefig(hist_name)
    # print key metrics
    print('min/max value of CT')
    print(min(data),max(data))
    print('max frequency/ inensity corresponding to max frequency')
    print(n.max(),bins[np.where(n == n.max())])

with open(csv_dimensions_path, 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Patient','In-plane dimension (mm)','Slice thickness (mm)'])
    # loop opver nii images
    files = list(sorted(os.listdir(nii_path)))
    cts_nii = list(sorted(os.listdir(nii_path_ct)))
    files = [x for x in files if x in cts_nii] # ensure structure and ct nii both present
    print(files)
    print(len(files))
    for i in files:
        image = sitk.ReadImage(os.path.join(nii_path,i))
        ct = sitk.ReadImage(os.path.join(nii_path_ct,i))
        print('Image % s' % i)
        dir = image.GetDirection()
        if dir[-1] == -1:
            image = sitk.Flip(image, [False,False,True])
            ct = sitk.Flip(ct, [False, False, True])
        writer.writerow([i.replace('.nii', ''),image.GetSpacing()[0],image.GetSpacing()[2]])
        py_arr_struc = sitk.GetArrayFromImage(image)
        py_arr_ct = sitk.GetArrayFromImage(ct) 
        np_root = i.replace('.nii','')
        struc_file = os.path.join(struc_path,np_root)
        ct_file = os.path.join(ct_path, np_root)
        # clip all CTs between 0 and 3071
        py_arr_ct = np.clip(py_arr_ct,0,3071)
        # if z is greater than 300 then cut in half
        if py_arr_struc.shape[0] > 300:
            print('crop image % s' % i)
            z = py_arr_struc.shape[0]
            py_arr_ct = py_arr_ct[int(z/2):z,:,:]
            py_arr_struc = py_arr_struc[int(z/2):z,:,:]    
        np.save(struc_file,py_arr_struc)
        np.save(ct_file,py_arr_ct)
        #print('min max values') 
        #print(min(py_arr_ct.flatten()),max(py_arr_ct.flatten())) 
        #histogram(py_arr_ct, np_root)
        print(py_arr_struc.shape)
        print(py_arr_ct.shape)
        print('--------------------')

        
        