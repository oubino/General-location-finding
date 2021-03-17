# loaded in model

import settings as S
import network
import torch.optim as optim
from torch.optim import lr_scheduler
import torch 
import paths
import data_loaders
import train_function
import evaluate_functions
import os
from torchsummary import summary
from torch import nn

class load_model:
    """Loaded in model is now a class"""

    def __init__(self, load_transfered_model):    
        super().__init__()
        # load in sigmas
        for k in S.landmarks_load:
          PATH_sigma_load = os.path.join(paths.epoch_load, "sigma_%1.0f.pt" % k)
          S.sigmas[k] = torch.load(PATH_sigma_load)['sigma'] # what value to initialise sigma
        
        # load in model/optimizer/scaler
        if S.UNET_model_user == True:
            # if model was saved as a transfered model it has different keys
            if load_transfered_model == False: 
                self.model_load = network.UNet3d(1,S.num_class_load, S.net_features_load)
            elif load_transfered_model == True:
                # load in a transfered model with dummy UNET, which can load params into
                self.model_load = network.Transfer_model(S.num_class_load, S.net_features_load, network.UNet3d(1,S.num_class_load, S.net_features_load))
        else:
            self.model_load = network.SCNET(1, S.num_class_load, S.scnet_feat_load)
        self.model_load = self.model_load.to(S.device)  
        self.optimizer_load = optim.Adam([
                        {'params': self.model_load.parameters()}
                       # {'params': S.sigmas[3]} # not general
                    ], lr=1e-3, weight_decay = 0.05) # use adam lr optimiser
        
        self.scaler_load = torch.cuda.amp.GradScaler()
        
        for k in S.landmarks_load:
            self.optimizer_load.add_param_group({'params': S.sigmas_load[k]}) 
            
        # load in current state from files
        self.model_load.load_state_dict(torch.load(paths.PATH_load))
        self.optimizer_load.load_state_dict(torch.load(paths.PATH_opt_load))
        self.scaler_load.load_state_dict(torch.load(paths.PATH_scaler_load))
        
        # add into optimizer any sigmas in sigmas but not in sigmas_load
        list_sigma = [x for x in S.sigmas if x not in S.sigmas_load]
        for k in list_sigma:
            self.optimizer_load.add_param_group({'params': S.sigmas[k]}) 
         
        self.scheduler = lr_scheduler.StepLR(self.optimizer_load, step_size=20000, gamma=0.1)
        
        # Model summary       
        print('Network structure')
        print('-----------------')
        summary(self.model_load, input_size=(1, S.in_y, S.in_x, S.in_z), batch_size = S.batch_size)
        
        
    def freeze_final_layers(self):
        for name, param in self.model_load.named_parameters():
            if (name != 'out.conv.bias' and name != 'out.conv.weight'):
                param.requires_grad = False
        
    def transfer_learn_unet_final_layer(self, class_number, features):
        # model becomes new model with different last layer
        self.model_load = network.Transfer_model(class_number, features, self.model_load)
        self.model_load = self.model_load.to(S.device)
        print('Transferred model')
        summary(self.model_load, input_size=(1, S.in_y, S.in_x, S.in_z), batch_size = S.batch_size)
        for name, param in self.model_load.named_parameters():
            if (param.requires_grad == True):
                print(name)
        
        # check which params have 
        """
        for name, param in model_transfer.named_parameters():
            if param.requires_grad == True:
                print(name,param)
        """
        
        
    def train(self, first_train, transfer_learn_decision):
        # for first train load in best loss and epochs completed
        if first_train == True:
            self.best_loss = torch.load(paths.PATH_val_loss_load)['best_val_loss']
            self.epochs_completed = torch.load(paths.PATH_epochs_completed_load)['epochs_completed']
      
        print('best loss is ')
        print(self.best_loss)
        data_loaders.train_set.dataset.__train__() 
        self.model_load, self.best_loss, self.epochs_completed = train_function.train_model(self.model_load, self.scaler_load, self.optimizer_load, self.scheduler, S.alpha,S.reg,S.gamma,S.sigmas, num_epochs=S.epoch_batch, best_loss = self.best_loss, epochs_completed = self.epochs_completed)
        
    def evaluate_post_train(self):
        # evaluate model
        self.model_load.eval() # trained
        data_loaders.test_set.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
        evaluate_functions.performance_metrics(self.model_load,S.sigmas,S.gamma, self.epochs_completed) # trained x 2
    
    def evaluate_pre_train(self):
        # if not trained load in best loss and epochs completed 
        self.best_loss = torch.load(paths.PATH_val_loss_load)['best_val_loss']
        self.epochs_completed = torch.load(paths.PATH_epochs_completed_load)['epochs_completed']
        print(self.best_loss)
        self.model_load.eval()
        data_loaders.test_set.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
        evaluate_functions.performance_metrics(self.model_load,S.sigmas,S.gamma, self.epochs_completed)
    
    def save(self):
        
        epochs_completed_string = str(self.epochs_completed) # trained
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
        
        torch.save(self.model_load.state_dict(), PATH_save)
        torch.save(self.optimizer_load.state_dict(), PATH_opt_save)
        for k in S.landmarks:
            PATH_sigma_save = os.path.join(train_path,"sigma_%1.0f.pt" % k)
            torch.save({'sigma': S.sigmas[k]}, PATH_sigma_save) 
        torch.save(self.scaler_load.state_dict(), PATH_scaler_save)
        torch.save({'best_val_loss': self.best_loss}, PATH_val_loss_save) # trained
        torch.save({'epochs_completed': self.epochs_completed}, PATH_epochs_completed_save) # trained
    
    def print_params(self):
        for name, param in self.model_load.named_parameters():
            print(name,param)
        

    