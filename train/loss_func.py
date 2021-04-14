# l2 loss function
import torch

from useful_functs import functions
import settings as S

def calc_loss_gauss(model, img, pred, target_coords, idx, metrics_landmarks, alpha, reg, gamma, epoch_samples, sigma): 

    # pred is Batch x Classes x Height x Width x Depth
    # target_coords is Batch x Dictionary (Dict[landmark][key]; Keys: x,y,z,locat,present)

    # loss function for difference between gaussian heat maps
    # pred is gaussian heatmap & need to convert target to gaussian heatmap

    # target is 3D CT scan of 1s, 2s, 3s, 4s, 5s, 6s
    # need to calculate max coord for each landmark 

    # i.e. metrics_landmarks[3]['loss'] is loss for landmark denoted by 3
    # metrics is defined per epoch

    # need to calculate total loss for each landmark but also store loss per landmark

    # cumulative over all landmarks
    loss_for_all_landmarks = 0
    total_batch_loss = 0
    total_sum_loss = 0
    total_reg_loss = 0
    total_alpha_loss = 0
    total_point_to_point = 0
    total_p2p_loss = 0
    
    for l in S.landmarks:
      
      # location of prediction for all images 
      pred_coords_max = functions.pred_max(pred, l, S.landmarks) # don't change to gauss fit as gauss fit takes too long
     
      # per landmark
      total_x_posn_landmark = 0
      total_y_posn_landmark = 0
      total_z_posn_landmark = 0
      total_x_targ_posn_landmark = 0
      total_y_targ_posn_landmark = 0
      total_z_targ_posn_landmark = 0
      total_point_to_point_landmark = 0
    
      # for every image in batch
      for i in range(S.batch_size): 

        img_number = epoch_samples + i # epoch_samples is 0, 32, 64 e.g. if batch is size 32
      
        x_size = S.in_x #target.size()[3]
        y_size = S.in_y #target.size()[2] 
        z_size = S.in_z #target.size()[4]
    

        # struc/pred location per image
        structure_com_x, structure_com_y, structure_com_z = target_coords[l]['x'][i],target_coords[l]['y'][i], target_coords[l]['z'][i]
        structure_com_x, structure_com_y, structure_com_z = structure_com_x.to(S.device), structure_com_y.to(S.device), structure_com_z.to(S.device)
        pred_max_x, pred_max_y, pred_max_z =  pred_coords_max[i][0], pred_coords_max[i][1], pred_coords_max[i][2] 

        # average x/y posn
        total_x_posn_landmark += pred_max_x
        total_y_posn_landmark += pred_max_y
        total_z_posn_landmark += pred_max_z
        total_x_targ_posn_landmark += structure_com_x
        total_y_targ_posn_landmark += structure_com_y
        total_z_targ_posn_landmark += structure_com_z

        # point to point per landmark per image
        img_landmark_point_to_point = functions.point_to_point(structure_com_x, structure_com_y, structure_com_z, pred_max_x, pred_max_y, pred_max_z)
        #img_landmark_point_to_point = point_to_point_mm(structure_com_x, structure_com_y, structure_com_z, pred_max_x, pred_max_y, pred_max_z, idx[i].item())

        # create target gauss map
        if target_coords[l]['present'][i] == 1:
          targ_gaus = functions.gaussian_map(structure_com_x,structure_com_y, structure_com_z,S.sigmas[l],gamma,x_size,y_size,z_size, output = True) 
        else:
          # target is full of zeros
          print('zero target')
          targ_gaus = torch.zeros(S.in_y, S.in_x, S.in_z).to(S.device)

        # pred heatmap is based on image in batch & landmark
        index = S.landmarks.index(l)
        pred_heatmap = pred[i][index]
        # l - 1 because l is 1,2,3,4,5,6
        
        # img_loss and sum_loss per landmark
        if S.wing_loss == False:
            img_loss = ((((pred_heatmap - targ_gaus)**2)).sum()) # multiply by targ gaus for box normalisation
            sum_loss = ((((pred_heatmap - targ_gaus)**2)).sum())
        else:
            A_1 = (1/ (1 +torch.pow(S.wing_theta/S.wing_epsilon,S.wing_alpha - targ_gaus)))
            A_2 = (torch.pow((S.wing_theta/S.wing_epsilon),(S.wing_alpha-targ_gaus-1)))
            A = S.wing_omega*A_1*(S.wing_alpha-targ_gaus)*A_2*(1/S.wing_epsilon)
            C = (S.wing_theta*A-S.wing_omega*torch.log(1+torch.pow((S.wing_theta/S.wing_epsilon),S.wing_alpha-targ_gaus)))
            a = torch.abs(targ_gaus-pred_heatmap)
            log_arg = 1 + torch.pow(torch.abs((targ_gaus-pred_heatmap)/S.wing_epsilon), S.wing_alpha-targ_gaus)
            img_loss = torch.where(torch.lt(a,S.wing_theta), S.wing_omega*torch.log(log_arg),A*a - C).sum()
            sum_loss = torch.where(torch.lt(a,S.wing_theta), S.wing_omega*torch.log(log_arg),A*a - C).sum()

        # just to see
        #if sum_loss < 10:
        #    print_3D_heatmap(img.squeeze(0), target.squeeze(0), pred.squeeze(0), l)
            
        
        # regularization term
        squ_weights = torch.tensor(0,dtype=torch.float64).to(S.device)
        for model_param_name, model_param_value in model.named_parameters():
          if model_param_name.endswith('weight'):
            squ_weights += (model_param_value.norm())**2
            
        
        # other loss terms beyond MSE
        regularization = (reg * squ_weights)     
        reg_loss = regularization
        alpha_loss = alpha * (S.sigmas[l].norm())**2    
        p2p_loss = S.p2p_reg_term * img_landmark_point_to_point
        
        img_loss += alpha * (S.sigmas[l].norm())**2 + regularization + p2p_loss

        # add to total loss
        total_batch_loss += img_loss
        total_sum_loss += sum_loss
        total_reg_loss += reg_loss
        total_alpha_loss += alpha_loss
        total_point_to_point += img_landmark_point_to_point
        total_p2p_loss += p2p_loss

        # need to add data to metrics per landmark
        metrics_landmarks[l]['loss'] += img_loss.data.cpu().numpy() # loss per image per landmark
        metrics_landmarks[l]['sum loss'] += sum_loss.data.cpu().numpy() # sum loss per image per landmark
        metrics_landmarks[l]['reg loss'] += reg_loss.data.cpu().numpy() # reg loss per image per landmark
        metrics_landmarks[l]['alpha loss'] += alpha_loss.data.cpu().numpy() # alpha loss per image per landmark
        metrics_landmarks[l]['p2p loss'] += p2p_loss.data.cpu().numpy() # p2p loss per image per landmark
        metrics_landmarks[l]['mean x pred'] += pred_max_x.data.cpu().numpy() # x posn per image per landmark
        metrics_landmarks[l]['mean y pred'] += pred_max_y.data.cpu().numpy() # y posn per image per landmark
        metrics_landmarks[l]['mean z pred'] += pred_max_z.data.cpu().numpy() # z posn per image per landmark
        metrics_landmarks[l]['mean x targ'] += structure_com_x.data.cpu().numpy() # x targ per image per landmark
        metrics_landmarks[l]['mean y targ'] += structure_com_y.data.cpu().numpy() # y targ per image per landmark
        metrics_landmarks[l]['mean z targ'] += structure_com_z.cpu().numpy() # z targ per image per landmark
        metrics_landmarks[l]['mean point to point'] += img_landmark_point_to_point.data.cpu().numpy() # p2p per image per landmark

      # print for every epoch_samples = 0 -> i.e first image in epoch
      if epoch_samples == 0:
        if l == S.landmarks[0]:
          print('------- First image of set --------')
        print('Landmark %1.0f' % l)
        print('--------------')
        print('targ: (%5.2f,%5.2f,%5.2f)' % (structure_com_x,structure_com_y,structure_com_z))
        print('pred: (%5.2f,%5.2f,%5.2f)' % (pred_max_x,pred_max_y,pred_max_z))
        # predicted coordinates
        print('point to point')
        print(img_landmark_point_to_point)
        print('img loss')
        print(img_loss)
        # plot predicted and target heatmap

          
    # return mean batch loss
    mean_batch_loss = (total_batch_loss/S.batch_size)
    
    # mean loss per image
    
    return mean_batch_loss

# metrics was here in chain previously