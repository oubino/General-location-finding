# file check

import os

# paths
#nii_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\CTs'
#numpy_path =  r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry\CTs_np'

path_1 = r'/home/olive/data/Facial_asymmetry_aaron_reclicks/CTs'
path_2 = r'/home/olive/data/Facial_asymmetry_aaron_reclicks/Structures'

# loop opver nii images
files_1 = list(sorted(os.listdir(path_1)))
files_2 = list(sorted(os.listdir(path_2)))

list_1 = [x for x in files_2 if x not in files_1]
list_2 = [x for x in files_1 if x not in files_2]

print('in files 2 but not files 1')
print(list_1)
print('in files 1 but not in files 2')
print(list_2)
