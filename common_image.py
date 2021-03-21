# save common images for Aaron and Oli

import numpy as np
import os
import yes_or_no


# paths
aaron_structures_load = r'/home/rankinaaron98/data/Facial_asymmetry_aaron/Structures'
aaron_cts_load = r'/home/rankinaaron98/data/Facial_asymmetry_aaron/CTs'
oli_structures_load = r'/home/olive/data/Facial_asymmetry_oli/Structures'
oli_cts_load = r'/home/olive/data/Facial_asymmetry_oli/CTs'

aaron_structures_save = r'/home/rankinaaron98/data/Facial_asymmetry_aaron_common/Structures'
aaron_cts_save = r'/home/rankinaaron98/data/Facial_asymmetry_aaron_common/CTs'
oli_structures_save = r'/home/olive/data/Facial_asymmetry_oli_common/Structures'
oli_cts_save = r'/home/olive/data/Facial_asymmetry_oli_common/CTs'


# loop over structures
files_aaron = list(sorted(os.listdir(aaron_structures_load)))
files_oli = list(sorted(os.listdir(oli_structures_load)))

# common files
list_1 = [x for x in files_aaron if x in files_oli]

aaron_or_oli = yes_or_no.question('aaron(y)/oli(n)')
save_images = yes_or_no.question('save images(y)/dont save images(n)')


if save_images == True:
    # for common structures save CT into folder and structure
    for i in list_1:
        # aaron 
        if aaron_or_oli == True:
            aaron_ct = np.load(os.path.join(aaron_cts_load,i))
            aaron_structure = np.load(os.path.join(aaron_structures_load,i))
            np.save(os.path.join(aaron_cts_save,i), aaron_ct)  
            np.save(os.path.join(aaron_structures_save,i), aaron_structure)   
        # oli
        elif aaron_or_oli == False:            
            oli_ct = np.load(os.path.join(oli_cts_load,i))
            oli_structure = np.load(os.path.join(oli_structures_load,i))
            np.save(os.path.join(oli_cts_save,i), oli_ct)
            np.save(os.path.join(oli_structures_save,i), oli_structure)
    
   
# still need to add in CSV for both
print('number of common images')
print(len(list_1))
