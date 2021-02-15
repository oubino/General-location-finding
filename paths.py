# paths
import settings as S
import os
 
train_folder_name = "train_" + S.epoch_load
run_folder_name = os.path.join(S.save_data_path, S.run_folder_load)
epoch_load = os.path.join(run_folder_name, train_folder_name)



PATH_load = os.path.join(epoch_load, "model.pt")
PATH_opt_load = os.path.join(epoch_load, "opt.pt")
#PATH_sigma_load = os.path.join(epoch_load, "sigma.pt")
PATH_scaler_load = os.path.join(epoch_load, "scaler.pt")
PATH_val_loss_load = os.path.join(epoch_load, "val_loss.pt")
PATH_epochs_completed_load = os.path.join(epoch_load, "epochs_completed.pt")



# Paths load and save
#PATH_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_64features.pt'
#PATH_opt_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_opt_64features.pt'
#PATH_sigma_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_sigma_64features.pt'
#PATH_scaler_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_scaler_64features.pt'
#PATH_val_loss_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'
#PATH_epochs_completed_load = r'C:\Users\olive\OneDrive\Documents\CNN\Report\3d_model_unet_downsample_val_loss_64features.pt'

