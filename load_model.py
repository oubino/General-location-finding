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

def init():    
    
    # load in sigmas - not specific
    for k in S.landmarks:
#      print(paths.PATH_sigma_load)
      PATH_sigma_load = os.path.join(paths.epoch_load, "sigma_%1.0f.pt" % k)
      S.sigmas[k] = torch.load(PATH_sigma_load)['sigma'] # what value to initialise sigma

    global model_load_in
    global optimizer_load
    global scaler_load
    global scheduler
    
    # load in model/optimizer/scaler
    if S.UNET_model_user == True:
        model_load_in = network.UNet3d(1,S.num_class, network.unet_feat)
    else:
        model_load_in = network.SCNET(1, S.num_class, S.scnet_feat)
        
    model_load_in = model_load_in.to(S.device)
    optimizer_load = optim.Adam([
                    {'params': network.model.parameters()}
                   # {'params': S.sigmas[3]} # not general
                ], lr=1e-3, weight_decay = 0.05) # use adam lr optimiser
    for k in S.landmarks:
        optimizer_load.add_param_group({'params': S.sigmas[k]})
    scaler_load = torch.cuda.amp.GradScaler()
    
    model_load_in.load_state_dict(torch.load(paths.PATH_load))
    optimizer_load.load_state_dict(torch.load(paths.PATH_opt_load))
    scaler_load.load_state_dict(torch.load(paths.PATH_scaler_load))
    scheduler = lr_scheduler.StepLR(optimizer_load, step_size=20000, gamma=0.1)

class Transfer_model(nn.Module):
    def __init__(self, n_classes, s_channels, pre_trained_model):
        super().__init__()
        self.n_classes = n_classes
        self.s_channels = s_channels
        
        self.pre_trained = nn.Sequential(
        *list(pre_trained_model.children())[:-1])
        print(*list(pre_trained_model.children())[:-1])
        self.out = network.OutConv(s_channels, n_classes)

    def forward(self, x):
        x1 = self.pre_trained.conv(x)
        x1 = self.pre_trained[0](x)
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)
        output = self.out(x1)
        return output 
    
def freeze_layers():
    for name, param in model_load_in.named_parameters():
        if (name != 'out.conv.bias' and name != 'out.conv.weight'):
            param.requires_grad = False
    model_froze = Transfer_model(15,32,model_load_in)
    model_froze = model_froze.to(S.device)
    summary(model_froze, input_size=(1, S.in_y, S.in_x, S.in_z))
    
def train(first_train):
    # load in val loss
    #global epochs_completed
    if first_train == True:
       # epochs_completed = torch.load(paths.PATH_epochs_completed_load)['epochs_completed']
        global model_load # commented out but may have to uncomment!
        global best_loss # trained
        #global epochs_completed # trained
        best_loss = torch.load(paths.PATH_val_loss_load)['best_val_loss']
        global epochs_completed
        epochs_completed = torch.load(paths.PATH_epochs_completed_load)['epochs_completed']

        
    # may have to specify these if == False
    print('best loss is ')
    print(best_loss)
    data_loaders.train_set.dataset.__train__() 
    model_load, best_loss, epochs_completed = train_function.train_model(model_load_in, scaler_load, optimizer_load, scheduler, S.alpha,S.reg,S.gamma,S.sigmas, num_epochs=S.epoch_batch, best_loss = best_loss, epochs_completed = epochs_completed)
    # trained x 3

def evaluate_post_train():
#    best_loss = torch.load(paths.PATH_val_loss_load)['best_val_loss']
#    print(best_loss)
    model_load.eval() # trained
    data_loaders.test_set.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
    evaluate_functions.performance_metrics(model_load,S.sigmas,S.gamma, epochs_completed) # trained x 2

def evaluate_pre_train():
    best_loss = torch.load(paths.PATH_val_loss_load)['best_val_loss']
    print(best_loss)
    model_load_in.eval()
    epochs_completed = torch.load(paths.PATH_epochs_completed_load)['epochs_completed']
    data_loaders.test_set.dataset.__test__() # sets whole dataset to test mode means it doesn't augment images
    evaluate_functions.performance_metrics(model_load_in,S.sigmas,S.gamma, epochs_completed)

def save():
    
    epochs_completed_string = str(epochs_completed) # trained
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
    
    torch.save(model_load.state_dict(), PATH_save)
    torch.save(optimizer_load.state_dict(), PATH_opt_save)
    for k in S.landmarks:
        PATH_sigma_save = os.path.join(train_path,"sigma_%1.0f.pt" % k)
        torch.save({'sigma': S.sigmas[k]}, PATH_sigma_save) 
    torch.save(scaler_load.state_dict(), PATH_scaler_save)
    torch.save({'best_val_loss': best_loss}, PATH_val_loss_save) # trained
    torch.save({'epochs_completed': epochs_completed}, PATH_epochs_completed_save) # trained

    