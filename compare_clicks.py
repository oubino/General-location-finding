# work out difference between arrays and 
# return error if greater than certain value

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
            # if deviate by certain value return elm
            
            