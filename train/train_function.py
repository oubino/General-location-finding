# train model function
import time
import copy
from collections import defaultdict
import torch

import settings as S
from data_loading import data_loaders
from useful_functs import functions
from train import loss_func

def train_model(model,scaler, optimizer, scheduler,alpha,reg,gamma,sigmas,num_epochs,best_loss, epochs_completed):
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                print('Learning rates (overall and for sigmas)')
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                print('')
                print('Testing on 1 image from val set')
                print('')
                model.eval()   # Set model to evaluate mode

            metrics_total = defaultdict(float)
            metrics_landmarks = defaultdict(float)
            for i in S.landmarks:
              metrics_landmarks[i] = defaultdict(float) 
            # i.e. metrics_landmarks[3]['loss'] is loss for landmark denoted by 3

            imgs_in_set = 0 # i.e. counts total number of images in train or val or test set
      
            batch_number = 1
            for batch in data_loaders.dataloaders[phase]:
                    # print dataloader 
                    inputs = batch['image']
                    idx = batch['idx']
                    target_coords = batch['coords']
                   # print(labels.size())
                    inputs = inputs.float().to(S.device)
                    #target_coords = target_coords.to(S.device)
                    patients = batch['patient']
                    
                    # target_coords is a dictioanry so is [landmarks]['x'][batch_id]
                    print('target coords')
                    print(target_coords)

                    # zero the parameter gradients
                    #print('zero the grad')
                    #optimizer.zero_grad() 
                    # amp mod
                    #print(optimizer.parameter())
                    #sigma.zero_grad

                    # forward
                    # track history only if in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        with torch.cuda.amp.autocast(enabled = S.use_amp):
                            outputs = model((inputs))
                            # 1. convert masks to heatmaps inside loss function (allows sigma optimisation)
                            loss = loss_func.calc_loss_gauss(model, inputs, outputs, target_coords, idx, metrics_landmarks,alpha,reg,gamma,imgs_in_set,sigmas)
                        
                        # print image for comparison
                        if imgs_in_set == 0:
                          # plot image
                          print(' ---- first image of set ---- (end)')
                        # 2. vs convert to heatmap here means no sigma optimisation


                        # backward + optimize only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            # only step once per epoch
                            #if (batch_number % batch_accumulation == 0) or (batch_number % 10 == 0):
                            #print('reached', batch_number, batch_accumulation)
                            scaler.step(optimizer)
                            scaler.update() 
                            scheduler.step()
                            optimizer.zero_grad()
                                


                    # statistics
                    imgs_in_set += inputs.size(0)
                    batch_number += 1
                
            print('Images in set')    
            print(imgs_in_set)

            print('')
            print('Summary on %s dataset' % phase)
            print('')
            functions.print_metrics(metrics_landmarks, imgs_in_set, phase)
            # print metrics divides the values by number of images
            # i.e. when values added to metrics it creates a total sum over all images in the set
            # within print metrics it divides by the total number of images so all values are means
            
            #print('The following have zero requires grad:')
            #all_have_grad = True
            #for name, param in model.named_parameters():
            #  if param.requires_grad == False:
            #      print (name)
            #      all_have_grad = False
            #if (all_have_grad == True):
            #  print('All parameters have require grad = true')      
            print('Sigmas are')
            for l in S.landmarks:
              print(sigmas[l])
              
              
            # here metrics_landmarks[l]['loss'] is divided by imgs_in_set so loss is defined as average per image!!
            
            epoch_loss = 0
            for l in S.landmarks:
                epoch_loss += metrics_landmarks[l]['loss'] # total loss i.e. each batch loss summed
                
                # add loss per landmark to tensorboard
                S.writer.add_scalar('%s loss for landmark %1.0f' % (phase,l), metrics_landmarks[l]['loss']/imgs_in_set, epochs_completed + epoch + 1)
                if phase == 'train':
                    S.writer.add_scalar('sigma for landmark %1.0f' % l, sigmas[l][0].item(),epochs_completed + epoch + 1)
                #print('writing to tensorboard')
            
            epoch_loss /= imgs_in_set
            
            

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("\n")
                print("------ deep copy best model ------ ")
                best_loss = epoch_loss
                print('  ' + 'best val loss: {:4f}'.format(best_loss))
                print('\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                # save model/optimizer etc. based on current time
                

        time_elapsed = time.time() - since
        finish_time = time.ctime(time_elapsed * (num_epochs - epoch) + time.time())
        print('\n')
        print('Epoch time: ' + '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Estimated finish time (till end of epoch batch): ', finish_time)
        print('\n')
        # save number of epochs completed
        epochs_completed_total = epochs_completed + epoch + 1
        


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, epochs_completed_total

