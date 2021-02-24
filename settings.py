
import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
import yes_or_no


def init():
    global norm_mean, norm_std, batch_size, landmarks, sigmas, num_class
    global in_x, in_y, in_z, epoch_batch, alpha, reg, gamma, lr_max, lr_min
    global step_size, threshold_img_print, normal_min, normal_max
    global normal_window, use_amp, downsample_ratio_h, downsample_ratio_w
    global downsample_ratio_d, downsample_idx_list
    global root, device
    global batch_accumulation
    global coding_path
    global pred_max
    global time_stamp
    global UNET_model_user  
    global downsample_user
    global run_path 
    global num_epoch_batches
    global run_folder
    global run_folder_load
    global epoch_load 
    global writer
    global img_counter_1, img_counter_2, img_counter_3
    global save_data_path
    global landmarks_loc
    global net_features
    global top_structures
    global left_structures, right_structures
    global wing_loss, wing_omega, wing_epsilon, wing_alpha, wing_theta
    global p2p_reg_term
    global error_counter
        
    
    # data path
    locally_or_server = yes_or_no.question('locally(y) / server(n)')
    if locally_or_server == True:
        # use local paths and ask Aaron/Oli for local paths 
        aaron_or_oli = yes_or_no.question('aaron(y) / oli (n)')
        if aaron_or_oli == True:
            # Aaron paths
            coding_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\General-location-finding'
            root = r'C:\Users\ranki_252uikw\Documents\MPhysS2\HNSCC_deepmind_cropped' # note lack of " "
            save_data_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Results'
        elif aaron_or_oli == False:
            # Oli paths
            coding_path = r'C:\Users\olive\OneDrive\Documents\GitHub\General-location-finding'
            root = r'C:\Users\olive\OneDrive\Documents\CNN\3D_data\HNSCC_deepmind_cropped' # note lack of " "
            save_data_path = r'C:\Users\olive\OneDrive\Documents\CNN\Sem 2\Results'
    elif locally_or_server == False:
        # use server paths for data and code for Aaron/Oli
        aaron_or_oli = yes_or_no.question('aaron(y) / oli (n)')
        if aaron_or_oli == True:
            # Aaron paths
            coding_path = r'/home/rankinaaron98/General-location-finding'
            root = r'/home/rankinaaron98/data/HNSCC_deepmind_cropped'
            save_data_path = r'/home/rankinaaron98/data/results/Aaron'
        elif aaron_or_oli == False:
            # Oli paths
            coding_path = r'/home/olive/GitHub/General-location-finding'
            root = r'/home/olive/data/HNSCC_deepmind_cropped'
            save_data_path = r'/home/olive/data/results/Oli'
            
        
        
   
    
    
    print('Results directory:')
    print(save_data_path)
    
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device working on: ')
    print(device)
        
    norm_mean = 180
    norm_std = 180
    
    batch_size = 1
    
    landmarks = [1,2,3,4] # brainstem # not general
    landmarks_loc = {1:'bot', 2:'bot', 3: 'top', 4:'top'} 
        # define which part of OAR to find
    
    # sigmas = defaultdict(float) ?
    sigmas = {} # sigma per landmark
    for k in landmarks:
      sigmas[k] = nn.Parameter(torch.tensor([20.]).to(device))# device = 'cuda'))#.to(device) # what value to initialise sigma
      sigmas[k].requires_grad = True
      #print(sigmas[k])
    
    num_class = len(landmarks)
    
    # input dimensions
    in_x = 128
    in_y = 128
    in_z = 80
    
    
    # training parameters
    epoch_batch = 10
    num_epoch_batches = 5
    net_features = 16
    
    alpha = 1/25000
    reg = 0.01 # reg = 0.001
    gamma = 100000
    lr_max = 0.1
    lr_min = 0.0001
    step_size = 64
    
    threshold_img_print = 0.5
    
    # normalise parameters
    normal_min = 15
    normal_max = 50
    normal_window = 1000
    
    # mixed precision
    use_amp = True
    
    # if downsampling
    downsample_ratio_h = np.empty((0), float)
    downsample_ratio_w = np.empty((0), float)
    downsample_ratio_d = np.empty((0), float)
    downsample_idx_list = np.empty((0), float)
    
    # use predicted max - if want gauss fit set to false
    pred_max = True
    
    # unique timestamp for model
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # decision on whether to use UNET or SCNET
    UNET_model_user = True
    
    # decision on whether to crop or downsample
    downsample_user = True
    
    # run folder
    run_folder = "run_24_feb_16_ft_reg_term"
    run_path = os.path.join(save_data_path, run_folder) 
    try:  
        os.mkdir(run_path)  
    except OSError as error:  
        print(error) 
    
    # load model path
    run_folder_load = "run_24_feb_16_ft_reg_term"
    epoch_load = str(135)
    
    # create tensorboard writer
    tensor_folder = os.path.join(save_data_path, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s-%s' % (time_stamp,run_folder_load))
    writer = SummaryWriter(tensorboard_loc) # may need to amend
    
    # image saved coutner
    img_counter_1 = 0
    img_counter_2 = 0
    img_counter_3 = 0
    
    # structures near the top which can be used for flipping
    top_structures = [5,6,3]
    
    # L/R structures
    left_structures = [1]
    right_structures = [2]
    
    # adaptive wing loss
    wing_loss = False
    # our max for heatmap is pre_factor 
    # ((gamma) * (2*np.pi)**(-dimension/2) * sigma ** (-dimension))  roughly 1
    # from paper, Wang et al
    # We empirically used α= 2.1 in our model. In our ex-periments, we found ω= 14,epsilon= 1,θ= 0.5
    wing_omega = 14
    wing_epsilon = 1
    wing_alpha = 2.1
    wing_theta = 0.5
    
    # penalise p2p 
    p2p_reg_term = 50
    
    # error counter
    error_counter = 0
    
    
    
    
    
