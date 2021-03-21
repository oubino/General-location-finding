
import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
import yes_or_no


def init():
    global norm_mean, norm_std, batch_size, landmarks, sigmas, num_class
    global in_x, in_y, in_z, alpha, reg, gamma, lr_max, lr_min
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
    global img_counter_1, img_counter_2, img_counter_3
    global save_data_path
    global landmarks_loc
    global top_structures, bot_structures
    global left_structures, right_structures
    global wing_loss, wing_omega, wing_epsilon, wing_alpha, wing_theta
    global p2p_reg_term
    global error_counter
    global print_CT_check
    global landmarks_total, landmarks_total_loc
    global landmarks_load, landmarks_load_loc
    global num_class_load 
    global net_features_load 
    global scnet_feat_load 
    global sigmas_load
    global k_folds
    global aaron_or_oli
    
          
    # data path
    locally_or_server = yes_or_no.question('locally(y) / server(n)')
    if locally_or_server == True:
        # use local paths and ask Aaron/Oli for local paths 
        aaron_or_oli = yes_or_no.question('aaron(y) / oli (n)')
        if aaron_or_oli == True:
            # Aaron paths
            coding_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\General-location-finding'
            root = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry' # note lack of " "
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
            combined_data = yes_or_no.question('combined_data (y) / solo_data (n)')
            if combined_data == True:
                # Aaron paths
                coding_path = r'/home/rankinaaron98/General-location-finding'
                root = r'/home/rankinaaron98/data/Facial_asymmetry_combined'
                save_data_path = r'/home/rankinaaron98/data/results/Aaron'
            elif combined_data == False:
                # Aaron paths
                coding_path = r'/home/rankinaaron98/General-location-finding'
                root = r'/home/rankinaaron98/data/Facial_asymmetry_aaron_common'
                save_data_path = r'/home/rankinaaron98/data/results/Aaron'
        # load model path
    
        elif aaron_or_oli == False:           
            combined_data = yes_or_no.question('combined_data (y) / solo_data (n)')
            if combined_data == True:         
                # Oli paths
                coding_path =  r'/home/olive/GitHub/General-location-finding' # r'/home/oliver_umney/GitHub/General-location-finding'
                root = r'/home/olive/data/Facial_asymmetry_combined' # r'/home/oliver_umney/data/Facial_asymmetry_combined'
                save_data_path =  r'/home/olive/data/results/Oli' # r'/home/oliver_umney/data/results/oliver_umney_web' 
            elif combined_data == False:   
                # Oli paths
                coding_path = r'/home/olive/GitHub/General-location-finding'
                root = r'/home/olive/data/Facial_asymmetry_oli_testset'
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
    
    # needed for transfer learning 
    
    # ---- begin -----
    
    # specify landmarks + region was trained on (iff loading in model)
    landmarks_load = [1,2,3,4,5,6,7,8,9,10] # brainstem # not general
    landmarks_load_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7: 'com',8:'com',9:'com',10:'com', }

    #landmarks_load = [1,3,5,7,9] # brainstem # not general
    #landmarks_load_loc = {1:'com', 3: 'com', 5:'com', 7: 'com', 9:'com', }
    
    num_class_load = len(landmarks_load)
    net_features_load = 32
    scnet_feat_load = 64
    
    sigmas_load = {} # sigma per landmark
    for k in landmarks_load:
      sigmas_load[k] = nn.Parameter(torch.tensor([20.]).to(device))# device = 'cuda'))#.to(device) # what value to initialise sigma
      sigmas_load[k].requires_grad = True
      #print(sigmas[k])
    
    # ----- end -----
    
    # specify landmarks + region want to train for
    #landmarks = [1,2,3,5,7,9] # brainstem # not general
    #landmarks_loc = {1:'com',2:'com', 3: 'com', 5:'com', 7:'com', 9:'com'} 
    
    landmarks = [1,2,3,4,5,6,7,8,9,10]
    landmarks_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7:'com',8:'com', 9:'com',10:'com', } 
    
    # specify all structures which are actually in image
    
    # structures near the top which can be used for flipping
    # "AMl", "AMr","HMl", "HMr", "FZl", "FZr", "FNl", "FNr", "SOl", "SOr"
    landmarks_total = [1,2,3,4,5,6,7,8,9,10]
    landmarks_total_loc = {1:'com', 2:'com', 3: 'com', 4:'com', 5:'com',6:'com', 7: 'com',8:'com', 9:'com',10:'com', } 
    top_structures = [5,6]
    bot_structures = [1,2]
    # L/R structures
    left_structures = [1,3,5,7,9]
    right_structures = [2,4,6,8,10]

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
    
    
    alpha = 1/25000
    reg = 0.01 # reg = 0.001
    gamma = 100000
    lr_max = 0.1
    lr_min = 0.0001
    step_size = 64
    
    threshold_img_print = 0.5
    
    # normalise parameters
    normal_min = 15 + 1024
    normal_max = 400 + 1024
    normal_window = 1800
    
    # mixed precision
    use_amp = True
    
    # if downsampling
    downsample_ratio_h = np.empty((0), float)
    downsample_ratio_w = np.empty((0), float)
    downsample_ratio_d = np.empty((0), float)
    downsample_idx_list = []
    
    # use predicted max - if want gauss fit set to false
    pred_max = True
    
    # unique timestamp for model
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # decision on whether to use UNET or SCNET
    UNET_model_user = True
    
    # decision on whether to crop or downsample
    downsample_user = True
 
    
    # image saved coutner
    img_counter_1 = 0
    img_counter_2 = 0
    img_counter_3 = 0

    
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
    
    # whether to print all CTs as a check
    print_CT_check = True
    
    # k folds
    k_folds = 5
    
    # k fold test
    global k_fold_ids
    k_fold_ids = []
    
def init_new():
    # oli vs aaron settings 
    global epoch_batch, num_epoch_batches
    global net_features, scnet_feat
    global run_path, run_folder, run_folder_load
    global epoch_load, folds_trained_with, fold_load
    global writer
    
    epoch_batch = int(input ("Epoch batch: "))
    num_epoch_batches = int(input ("Num epoch batch: "))
    net_features = 32
    scnet_feat = 64
    
    if aaron_or_oli == True:
        run_folder = "run_19_mar_k_fold_aaron"
        run_folder_load = "run_19_mar_k_fold_aaron"
    elif aaron_or_oli == False:
        run_folder = "run_18_mar_k_fold_common_dataset"
        run_folder_load = "run_18_mar_k_fold_common_dataset"
        
    run_path = os.path.join(save_data_path, run_folder) 
    try:  
        os.mkdir(run_path)  
    except OSError as error:  
            print(error) 
        
    # create tensorboard writer
    tensor_folder = os.path.join(save_data_path, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s-%s' % (time_stamp,run_folder_load))
    writer = SummaryWriter(tensorboard_loc) # may need to amend

def init_load():
    # oli vs aaron settings 
    global epoch_batch, num_epoch_batches
    global net_features, scnet_feat
    global run_path, run_folder, run_folder_load
    global epoch_load, folds_trained_with, fold_load
    global writer
    
    epoch_batch = int(input ("Epoch batch: "))
    num_epoch_batches = int(input ("Num epoch batch: "))
    net_features = 32
    scnet_feat = 64
    
    if aaron_or_oli == True:
        run_folder = "run_19_mar_k_fold_aaron"
        run_folder_load = "run_19_mar_k_fold_aaron"
    elif aaron_or_oli == False:
        run_folder = "run_18_mar_k_fold_common_dataset"
        run_folder_load = "run_18_mar_k_fold_common_dataset"
        
    run_path = os.path.join(save_data_path, run_folder) 
    try:  
        os.mkdir(run_path)  
    except OSError as error:  
            print(error) 

    epoch_load = input ("epoch to load in: ")
    folds_trained_with = 5
    fold_load = input ("Fold to load in: None for no folds, 0 for fold 0, 1 for fold 1 etc.: ")
    
    # create tensorboard writer
    tensor_folder = os.path.join(save_data_path, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s-%s' % (time_stamp,run_folder_load))
    writer = SummaryWriter(tensorboard_loc) # may need to amend
    
    
    
    
    
