# initialised model

import settings as S
import network
import data_loaders
import evaluate_functions
import torch
import paths
import os

def init():
    import torch.optim as optim
    from torch.optim import lr_scheduler
    
    global scaler
    global scheduler
    global optimizer
    global model
    
    if S.unet_model_user == True:
        model = network.UNet3d(1,S.num_class, network.unet_feat)
    else:
        model = network.SCNET(1, S.num_class, S.scnet_feat)
    model = model.to(S.device)
    
    # initialise optimizer/scheduler/scaler
    optimizer = optim.Adam([
                    {'params': model.parameters()}
                    #{'params': S.sigmas[3]},
                    #{'params': sigmas[3]} # not general
                ], lr=1e-3, weight_decay = 0.05) # use adam lr optimiser
    for k in S.landmarks:
        optimizer.add_param_group({'params': S.sigmas[k]})
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=S.use_amp)
    
def train(first_train):
    import train_function
    
    if first_train == True:
        global best_loss
        global epochs_completed
        best_loss = 1e10
        epochs_completed = 0
        global model_trained
    
    #elif first_train == False:
        # may have to let it know what best_loss, epochs_completed and model trained are again?

    data_loaders.train_set.dataset.__train__() 
    model_trained, best_loss, epochs_completed = train_function.train_model(model, scaler, optimizer, scheduler, S.alpha,S.reg,S.gamma,S.sigmas, num_epochs=S.epoch_batch, best_loss = best_loss, epochs_completed = epochs_completed)

def evaluate():
    model_trained.eval()   # Set model to the evaluation mode
    data_loaders.test_set.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
    evaluate_functions.performance_metrics(model_trained,S.sigmas,S.gamma, epochs_completed)
    
def save():
    
    epochs_completed_string = str(epochs_completed)
    file_name = "train_" + epochs_completed_string
    train_path = os.path.join(S.run_path, file_name) # directory labelled with epochs_completed
    
    try: 
        os.mkdir(train_path)
    except OSError as error:
        print(error)
        
    PATH_save = os.path.join(train_path, "model.pt")
    PATH_opt_save = os.path.join(train_path, "opt.pt")
    #PATH_sigma_save = os.path.join(train_path,"sigma.pt")
    PATH_scaler_save = os.path.join(train_path,"scaler.pt")
    PATH_val_loss_save = os.path.join(train_path,"val_loss.pt")
    PATH_epochs_completed_save  = os.path.join(train_path,"epochs_completed.pt")
        
    #PATH_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_64features.pt'
    #PATH_opt_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_opt_64features.pt'
    #PATH_sigma_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_sigma_64features.pt'
    #PATH_scaler_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_scaler_64features.pt'
    #PATH_val_loss_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'
    #PATH_epochs_completed_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'


    torch.save(model.state_dict(), PATH_save)
    torch.save(optimizer.state_dict(), PATH_opt_save)
    for k in S.landmarks:
        PATH_sigma_save = os.path.join(train_path,"sigma_%1.0f.pt" % k)
        torch.save({'sigma': S.sigmas[k]}, PATH_sigma_save) 
    torch.save(scaler.state_dict(), PATH_scaler_save)
    torch.save({'best_val_loss': best_loss}, PATH_val_loss_save)
    torch.save({'epochs_completed': epochs_completed}, PATH_epochs_completed_save)
