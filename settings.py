import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter

from useful_functs import yes_or_no
from useful_functs import slide_coords

def init(rts_q):
    global norm_mean, norm_std, batch_size, landmarks, sigmas, num_class
    global in_x, in_y, in_z, alpha, reg, gamma, lr_max, lr_min
    global step_size, threshold_img_print, normal_min, normal_max
    global normal_window, use_amp
    global downsample_ratio_list #, crop_list
    global root, device
    global batch_acc_steps
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
    #global print_CT_check
    global landmarks_total, landmarks_total_loc
    global k_folds
    global aaron_or_oli
    global ct_print
    global k_fold_ids
    global batch_size_test, batch_acc_steps_test
    global train_line, clicker, rts
    global epoch_deep_saved
    global paed_bool

    # paths
    
    Paed_q = yes_or_no.question('Paediatric data (y) / Adult data (n): ')
    if Paed_q == False:
        paed_bool = False
        locally_or_server = yes_or_no.question('locally(y) / server(n): ')
        if locally_or_server == True:
            # use local paths and ask Aaron/Oli for local paths 
            aaron_or_oli = yes_or_no.question('aaron(y) / oli (n): ')
            if aaron_or_oli == True:
                # Aaron paths
                coding_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\General-location-finding'
                root = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Facial_asymmetry_aaron_testset' # note lack of " "
                save_data_path = r'C:\Users\ranki_252uikw\Documents\MPhysS2\Results'
            elif aaron_or_oli == False:
                # Oli paths
                coding_path = r'C:\Users\olive\OneDrive\Documents\GitHub\General-location-finding'
                root = r'C:\Users\olive\OneDrive\Documents\MPhys\3D_data\Facial_asymmetry_reclicks' # note lack of " "
                save_data_path = r'C:\Users\olive\OneDrive\Documents\MPhys\Sem 2\Results'
        elif locally_or_server == False:
            # use server paths for data and code for Aaron/Oli
            aaron_or_oli = yes_or_no.question('aaron(y) / oli (n)')
            if aaron_or_oli == True:
                # Aaron paths
                coding_path = r'/home/rankinaaron98/General-location-finding'
                root = r'/home/rankinaaron98/data/Facial_asymmetry_reclicks'
                save_data_path = r'/home/rankinaaron98/data/results/Aaron'
            # load model path
            elif aaron_or_oli == False:           
                google_oli = yes_or_no.question('are you accessing through: sdk (n)/google (y)')
                if google_oli == False:  
                    # Oli paths
                    coding_path = r'/home/olive/GitHub/General-location-finding'
                    root = r'/home/olive/data/Facial_asymmetry_reclicks'
                    save_data_path = r'/home/olive/data/results/Oli'
                elif google_oli == True:
                    # google Oli paths
                    coding_path = r'/home/oliver_umney/GitHub/General-location-finding'
                    root = r'/home/oliver_umney/data/Facial_asymmetry_reclicks'
                    save_data_path =  r'/home/oliver_umney/data/results/oliver_umney_web' 
    
    elif Paed_q == True:
        paed_bool = True
        coding_path = r'/home/oli/GitHub/General-location-finding'
        if rts_q == False:
            root = r'/home/oli/data/paed_dataset/train'
        elif rts_q == True:
            root = r'/home/oli/data/paed_dataset/test'
        save_data_path =  r'/home/oli/data/results/oli' 
   
    # results directory
    print('Results directory:')
    print(save_data_path)
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device working on: ')
    print(device)
    
    # root
    print('root')
    print(root)
        
    # batch size
    change_batch_size = yes_or_no.question('Would you like to change batch size from default(3): ')
    if change_batch_size == True:
        batch_size = int(input ("Batch size: "))
    else:
        batch_size = 3  
    batch_size_test = 1 # HAS TO BE 1 ! STRUC ORIGINAL OTHERWISE WILL BE DIFFERENT SHAPES AND BATCH LOADER WILL FAIL
    
    
    
    batch_acc_steps = 2  
  
    
    # specify landmarks + region want to train for - AMEND
    #landmarks = [1,2,3,5,7,9] # brainstem # not general
    #landmarks_loc = {1:'com',2:'com', 3: 'com', 5:'com', 7:'com', 9:'com'} 
    landmarks = [1,2,3,4,5,6,7,8,9,10]
    #landmarks_loc = {1:'line',2:'line', 3: 'line',4:'line', 5:'line',6:'line', 7:'line',8:'line', 9:'line',10:'line', }
    landmarks_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7:'com',8:'com', 9:'com',10:'com', } 
    num_class = len(landmarks)
    # make user double check correct
    print('\n')
    print('Landmarks training for')
    print(landmarks)
    print(landmarks_loc)
    input("Press Enter to continue...")
    
    
    # specify all structures which are actually in image
    # structures near the top which can be used for flipping
    # "AMl", "AMr","HMl", "HMr", "FZl", "FZr", "FNl", "FNr", "SOl", "SOr"
    landmarks_total = [1,2,3,4,5,6,7,8,9,10]
    #landmarks_total_loc = {1:'line', 2:'line', 3: 'line', 4:'line', 5:'line',6:'line', 7: 'line',8:'line', 9:'line',10:'line', } 
    landmarks_total_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7:'com',8:'com', 9:'com',10:'com', } 
    top_structures = [5,6]
    bot_structures = [1,2]
    # L/R structures
    left_structures = [1,3,5,7,9]
    right_structures = [2,4,6,8,10]

    # sigma per landmark
    sigmas = {} 
    for k in landmarks:
      sigmas[k] = nn.Parameter(torch.tensor([20.]).to(device))# device = 'cuda'))#.to(device) # what value to initialise sigma
      sigmas[k].requires_grad = True
      #print(sigmas[k])
    
    # input dimensions
    in_x = 192
    in_y = 192
    in_z = 100
    
    # learning params
    alpha = 1/25000
    reg = 0.01 # reg = 0.001
    gamma = 100000
    lr_max = 0.1
    lr_min = 0.0001
    step_size = 64
    
    # print params
    threshold_img_print = 0.5
    
    # normalise parameters
    normal_min = 15 + 1024
    
    normal_max = 400 + 1024
    normal_window = 1800
    
    # mixed precision
    use_amp = True
    
    # if downsampling
    downsample_ratio_list = {}
    
    # if cropping
    #crop_list = {}
    
    # use predicted max - if want gauss fit set to false
    pred_max = True
    
    # unique timestamp for model
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # decision on whether to use UNET or SCNET
    UNET_model_user = True
    
    # decision on whether to crop or downsample
    downsample_q = input ("Downsample(y)/crop(n): ")
    if downsample_q == 'y':
        downsample_user = True
    elif downsample_q == 'n':
        downsample_user = False
    
    print('Downsample')
    print(downsample_user)
 
    # image saved counter
    img_counter_1 = 0
    img_counter_2 = 0
    img_counter_3 = 0

    # adaptive wing loss
    wing_loss = True
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
    #print_CT_check = True
    #ct_print = "0072.npy"
    
    # k folds
    k_folds = 5
    k_fold_ids = []   # k fold test
    
    # train line true
    #rts_q = input (" Eval on reserved test set? (y/n)? ")
    train_line_q = input ("Train/eval on a line (y/n)?: ")
    if rts_q == False:
        rts = False
        if train_line_q == 'y':
            train_line = True
        elif train_line_q == 'n':
            train_line = False
            aaron_or_oli = yes_or_no.question('Aaron clicks(y) / Oli clicks (n): ')
            if aaron_or_oli == True:
                clicker = 'Aaron'
            elif aaron_or_oli == False:
                clicker = 'Oli'
    elif rts_q == True:
        rts = True
        if train_line_q == 'n':
            train_line = False # necessary for crop in transformations
            clicker_input = input ('Aaron(a), Oli (o), Abby (ab): ')
            if clicker_input == 'a':
                clicker = 'Aaron_test_set'
            elif clicker_input == 'o':
                clicker = 'Oli_test_set'
            elif clicker_input == 'ab':
                clicker = 'Abby_test_set'
        elif train_line_q == 'y':
            train_line = True
    else:
        print('ERROR')

def init_slide_window(patients):
    global sliding_window, sliding_points, crop_coords_slide, slide_index
    global crop_list

    # sliding window

    sliding_points = 75
    slide_index = 0
    crop_coords_slide = {}
    # if cropping
    crop_list = {}

    for p in patients:
        crop_coords_slide[p] = {}
        crop_list[p] = {}
        for i in range(sliding_points):
            crop_coords_slide[p][i] = {}
            crop_list[p][i]= {}
            # work out crop coord locations for slide
        crop_coords_slide[p] = slide_coords.coords(p, in_x, in_y, in_z, sliding_points)

            #crop_coords_slide[p][i]['x'] = (in_x/2)*i + in_x/2
   
     
def init_new():
    # oli vs aaron settings 
    global epoch_batch, num_epoch_batches
    global net_features, scnet_feat
    global run_path, run_folder
    global epoch_load, folds_trained_with, fold_load
    
    epoch_batch = int(input ("Epoch batch: "))
    num_epoch_batches = int(input ("Num epoch batch: "))
    net_features= 32
    scnet_feat = 64
    
    # -- AMEND -- 
    run_folder = input ( "run folder (NOTE IT WILL OVERWRITE TENSORBOARD SO MAKE SURE): ")

        #run_folder_load = "run_22_mar_test_aaron_my_data"
    # make user double check correct
    print('\n')
    print('run folder')
    print(run_folder)
    input("Press Enter to continue...")
        
    run_path = os.path.join(save_data_path, run_folder) 
    try:  
        os.mkdir(run_path)  
    except OSError as error:  
            print(error) 
    
def init_load():
    # oli vs aaron settings 
    global epoch_batch, num_epoch_batches
    global net_features, scnet_feat
    global run_path, run_folder
    global epoch_load, folds_trained_with, fold_load
    global landmarks_load, landmarks_load_loc
    global num_class_load, net_features_load, scnet_feat_load, sigmas_load

    net_features_load = 32
    scnet_feat_load = 64
    
    # -- AMEND -- 
    # specify landmarks + region was trained on
    landmarks_load = [1,2,3,4,5,6,7,8,9,10] # brainstem # not general
    landmarks_load_loc = {1:'com',2:'com', 3: 'com',4:'com', 5:'com',6:'com', 7: 'com',8:'com',9:'com',10:'com', }
    #landmarks_load = [1,3,5,7,9] # brainstem # not general
    #landmarks_load_loc = {1:'com', 3: 'com', 5:'com', 7: 'com', 9:'com', }
    num_class_load = len(landmarks_load)
    # make user double check correct
    print('\n')
    print('Landmarks loaded in')
    print(landmarks_load)
    print(landmarks_load_loc)
    input("Press Enter to continue...")
    
    sigmas_load = {} # sigma per landmark
    for k in landmarks_load:
      sigmas_load[k] = nn.Parameter(torch.tensor([20.]).to(device))# device = 'cuda'))#.to(device) # what value to initialise sigma
      sigmas_load[k].requires_grad = True
      #print(sigmas[k])
        
    # -- AMEND -- 
    run_folder = input ( "run folder (NOTE IT WILL OVERWRITE TENSORBOARD SO MAKE SURE): ")


    # make user double check correct
    print('\n')
    print('run folder')
    print(run_folder)
    input("Press Enter to continue...")
        
    run_path = os.path.join(save_data_path, run_folder) 
    #try:  
    #    os.mkdir(run_path)  
    #except OSError as error:  
    #        print(error) 

    epoch_load = input ("epoch to load in: ")
    folds_trained_with = 5
    fold_load = input ("Fold to load in: 0 for fold 0, 1 for fold 1 etc.: ")
        
def tensorboard_init(fold):
    global writer
    
    # create tensorboard writer
    tensor_folder = os.path.join(save_data_path, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s_fold_%s' % (run_folder,fold))
    writer = SummaryWriter(tensorboard_loc) 
    
    
    
# this is only used for final eval on Aaron, Oli, combined
# implemented to ensure uses same prediction to compare to Aaron, Oli, Combined

""" 
    
def init_load_eval_line():
    # oli vs aaron settings 
    global epoch_batch, num_epoch_batches
    global net_features, scnet_feat
    global run_path, run_folder
    global epoch_load, folds_trained_with, fold_load
    global writer
    global landmarks_load, landmarks_load_loc
    global num_class_load, net_features_load, scnet_feat_load, sigmas_load

    net_features_load = 32
    scnet_feat_load = 64
    
    # -- AMEND -- 
    # specify landmarks + region was trained on
    landmarks_load = [1,2,3,4,5,6,7,8,9,10] # brainstem # not general
    # think if put 'line' or 'com' makes no difference - should only eval using COM
    landmarks_load_loc = {1:'line',2:'line', 3: 'line',4:'line', 5:'line',6:'line', 7: 'line',8:'line',9:'line',10:'line', }
    #landmarks_load = [1,3,5,7,9] # brainstem # not general
    #landmarks_load_loc = {1:'com', 3: 'com', 5:'com', 7: 'com', 9:'com', }
    num_class_load = len(landmarks_load)
    # make user double check correct
    print('\n')
    print('Landmarks loaded in')
    print(landmarks_load)
    print(landmarks_load_loc)
    input("Press Enter to continue...")
    
    sigmas_load = {} # sigma per landmark
    for k in landmarks_load:
      sigmas_load[k] = nn.Parameter(torch.tensor([20.]).to(device))# device = 'cuda'))#.to(device) # what value to initialise sigma
      sigmas_load[k].requires_grad = True
      #print(sigmas[k])
        
    # -- AMEND -- 
    if aaron_or_oli == True:
        run_folder = "run_eval_line"
    elif aaron_or_oli == False:
        run_folder = "run_eval_line"
        # make user double check correct
    print('\n')
    print('run folder')
    print(run_folder)
    input("Press Enter to continue...")
        
    run_path = os.path.join(save_data_path, run_folder) 
    #try:  
    #    os.mkdir(run_path)  
    #except OSError as error:  
    #        print(error) 

    epoch_load = input ("epoch to load in: ")
    folds_trained_with = 5
    
    # create tensorboard writer
    tensor_folder = os.path.join(save_data_path, 'tensorboard')
    tensorboard_loc = os.path.join(tensor_folder, '%s-%s' % (time_stamp,run_folder))
    writer = SummaryWriter(tensorboard_loc) 
    
    
    
    """
    
