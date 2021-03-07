# convert .nii to .py and save

import SimpleITK as sitk
import numpy as np

# paths
nii_path = 
numpy_path =  

# loop opver nii images
files = list(sorted(os.listdir(nii_path)))
for i in files:
    sitk = sitk.ReadImage(i)
    py_arr = sitk.GetArrayFromImage(sitk)
    np_root = i.replace('.nii','.py')
    np.save(np_root,py_arr)
    

# A path to a T1-weighted brain .nii image:
# t1_fn = './brain_t1_0001.nii'
