# convert .nii to .py and save

import SimpleITK as sitk
import numpy as np
import os

# paths
nii_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\landmarks_aaron'
numpy_path =  r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\landmarks_aaron_np'

#nii_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs'
#numpy_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs_np'

# loop opver nii images
files = list(sorted(os.listdir(nii_path)))
print(files)
for i in files:
    image = sitk.ReadImage(os.path.join(nii_path,i))
    print('Image % s' % i)
    py_arr = sitk.GetArrayFromImage(image)
    np_root = i.replace('.nii','')
    numpy_file = os.path.join(numpy_path,np_root)
    np.save(numpy_file,py_arr)
    print('--------------------')
    print(py_arr.shape)

# A path to a T1-weighted brain .nii image:
# t1_fn = './brain_t1_0001.nii'
