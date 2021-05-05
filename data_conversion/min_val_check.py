# check min value
import os
import numpy as np
import matplotlib.pyplot as plt

root = r'/home/oli/data/paed_dataset/test'
cts = os.path.join(root, "CTs")
ct_list = list(sorted(os.listdir(cts)))
hist_root = os.path.join(root, "histograms")

def histogram(data, patient):
    # plot and save histogram
    data = np.array(data)
    data = data.flatten()
    data = np.sort(data)
    patient = patient.replace('.npy','')
    plt.figure()
    n, bins, patches = plt.hist(x=data, bins=1000, color='#0504aa', alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("pixel values for %s" % (patient))
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    hist_name = os.path.join(hist_root, "%s" % (patient))
    # set x lim to centre around 0
    #plt.xticks(np.arange(-30,31,4))
    plt.savefig(hist_name)


# amended cts
"""
amend_cts = os.path.join(root, "CTs_amend")
try: 
    os.mkdir(amend_cts)
except OSError as error:
    print(error)
"""

for i in ct_list:
    img_path = os.path.join(root, "CTs", i) 
    img = np.load(img_path)
    min_val = np.amin(img)
    max_val = np.amax(img)
    #histogram(img,i)
    print(i)
    
    if min_val < 0:
        print(i, min_val)
        #histogram(img,i) #seemed to be just case of needing to add
        #img = img + 1023
        #img_save_path = os.path.join(amend_cts, i) 
        #np.save(img_save_path, img)
        
        
        