# initialised model

import torch
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary

from train import train_function
from useful_functs import functions
import settings as S
from model import network
from data_loading import data_loaders
from evaluate import evaluate_functions

class initialise_model:
    """Initialised model is a class"""
    def __init__(self):
        super().__init__()
        
        # initialise model
        if S.UNET_model_user == True:
            self.model = network.UNet3d(1,S.num_class, S.net_features)
        else:
            self.model = network.SCNET(1, S.num_class, S.scnet_feat)
        self.model = self.model.to(S.device)
        
        # Model summary       
        print('Network structure')
        print('-----------------')
        summary(self.model, input_size=(1, S.in_y, S.in_x, S.in_z), batch_size = S.batch_size)
        
        # initialise optimizer/scheduler/scaler
        self.optimizer = optim.Adam([
                        {'params': self.model.parameters()}
                        #{'params': S.sigmas[3]},
                        #{'params': sigmas[3]} # not general
                    ], lr=1e-3, weight_decay = 0.05) # use adam lr optimiser
        for k in S.landmarks:
            self.optimizer.add_param_group({'params': S.sigmas[k]})
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.1)
        self.scaler = torch.cuda.amp.GradScaler(enabled=S.use_amp)
        
    def train(self, first_train):
        
        if first_train == True:
            self.best_loss = 1e10
            self.epochs_completed = 0
    
        data_loaders.dataset.__train__() 
        self.model, self.best_loss, self.epochs_completed = train_function.train_model(self.model, self.scaler, self.optimizer, self.scheduler, S.alpha,S.reg,S.gamma,S.sigmas, num_epochs=S.epoch_batch, best_loss = self.best_loss, epochs_completed = self.epochs_completed)
    
    def evaluate(self, fold):
        
        self.model.eval()   # Set model to the evaluation mode
        data_loaders.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
        if S.train_line == True:
            evaluate_functions.performance_metrics_line(self.model,S.sigmas,S.gamma, self.epochs_completed, fold)
        elif S.train_line == False:
            evaluate_functions.performance_metrics(self.model,S.sigmas,S.gamma, self.epochs_completed, fold)
        
    def save(self, fold):
        
        epochs_completed_string = str(self.epochs_completed)
        fold_string = functions.string(fold)
        file_name = "train_" + epochs_completed_string + fold_string 
        train_path = os.path.join(S.run_path, file_name) # directory labelled with epochs_completed
        
        try: 
            os.mkdir(train_path)
        except OSError as error:
            print(error)
            
        PATH_save = os.path.join(train_path, "model.pt")
        PATH_opt_save = os.path.join(train_path, "opt.pt")
        PATH_scaler_save = os.path.join(train_path,"scaler.pt")
        PATH_val_loss_save = os.path.join(train_path,"val_loss.pt")
        PATH_epochs_completed_save  = os.path.join(train_path,"epochs_completed.pt")
            
        #PATH_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_64features.pt'
        #PATH_opt_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_opt_64features.pt'
        #PATH_sigma_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_sigma_64features.pt'
        #PATH_scaler_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_scaler_64features.pt'
        #PATH_val_loss_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'
        #PATH_epochs_completed_save = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'
    
    
        torch.save(self.model.state_dict(), PATH_save)
        torch.save(self.optimizer.state_dict(), PATH_opt_save)
        for k in S.landmarks:
            PATH_sigma_save = os.path.join(train_path,"sigma_%1.0f.pt" % k)
            torch.save({'sigma': S.sigmas[k]}, PATH_sigma_save) 
        torch.save(self.scaler.state_dict(), PATH_scaler_save)
        torch.save({'best_val_loss': self.best_loss}, PATH_val_loss_save)
        torch.save({'epochs_completed': self.epochs_completed}, PATH_epochs_completed_save)
    
    def print_params(self):
        for name, param in self.model.named_parameters():
            print(name,param)
