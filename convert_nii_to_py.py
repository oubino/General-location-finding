# convert .nii to .py and save

import SimpleITK as sitk
import numpy as np
import os

# paths
#nii_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\CTs'
#numpy_path =  r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\CTs_np'

nii_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs'
numpy_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs_np'
# loop opver nii images
files = list(sorted(os.listdir(nii_path)))
print(files)
for i in files:
    sitk = sitk.ReadImage(i)
    print('Image %1.0f' % i)
    py_arr = sitk.GetArrayFromImage(sitk)
    np_root = i.replace('.nii','.py')
    np.save(np_root,py_arr)
    print('--------------------')

# A path to a T1-weighted brain .nii image:
# t1_fn = './brain_t1_0001.nii'
