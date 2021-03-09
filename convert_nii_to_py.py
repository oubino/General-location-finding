# convert .nii to .py and save

import SimpleITK as sitk
import numpy as np
import os

# paths
nii_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\landmarks_aaron'
numpy_path =  r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\landmarks_aaron_np'
csv_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\img_dimensions.csv'

#nii_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs'
#numpy_path = r'/home/rankinaaron98/data/Facial_asymmetry/CTs_np'

with open(csv_dimensions_path, 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Patient','In-plane dimension (mm)','Slice thickness (mm)'])
    # loop opver nii images
    files = list(sorted(os.listdir(nii_path)))
    print(files)
    for i in files:
        image = sitk.ReadImage(os.path.join(nii_path,i))
        print('Image % s' % i)
        dir = image.GetDirection()
        if dir[-1] == -1:
            image = sitk.Flip(image, [False,False,True])
        writer.writerow([i.replace('.nii', ''),image.GetSpacing()[0],image.GetSpacing()[2]])
        py_arr = sitk.GetArrayFromImage(image)
        np_root = i.replace('.nii','')
        numpy_file = os.path.join(numpy_path,np_root)
        np.save(numpy_file,py_arr)
        print(py_arr.shape)
        print('--------------------')
