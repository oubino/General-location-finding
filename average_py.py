# take in .py arrays and caclulate average of aaron & oli

# paths
aaron_folder = 
oli_folder = 
save_folder = 

# loop over .py
files_aaron = list(sorted(os.listdir(aaron_folder)))
files_oli = list(sorted(os.listdir(oli_folder)))

def concat(list_big, list_small):  
    for elm in list_big:
        if elm in list_small:
            # concat
            index_big = np.where(list_big = elm)
            index_small = np.where(list_small = elm)
            numpy_concat = # concat
            save_root = save_folder + elm
            np.save(save_root,numpy_concat)
            
