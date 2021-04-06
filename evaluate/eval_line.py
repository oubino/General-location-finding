# script to eval line model on aaron/oli and combined

from model import load_model
from data_loading import data_loaders
from useful_functs import yes_or_no
from useful_functs import functions
import settings

settings.init()
settings.init_load_eval_line()

# eval for all folds
for fold in range(settings.folds_trained_with):
    settings.fold_load = fold # settings needs parameter fold load
    data_loaders.init_load_k_fold(int(fold))    
    load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
    model = load_model.load_model(load_transfered_model)
    print('\n')
    print('Evaluating model for fold %1.0f' % (fold))
    print('----------------')
    
    images = []
    preds = []
    original_imgs = []
    #structures = []
    
    # need to get images and predictions here
    for batch in data_loaders.dataloaders['test']:
        #image = batch['image'].to(S.device)
        #structure = batch['structure'].to(S.device)
        patients = batch['patient']
        
        data_loaders.dataset.__test_resize__()
        image = batch['image']
        pred = model(image, crop = False) 
        for l in settings.landmarks:
            for i in range(image.size()[0]):
                pred_coords_max = functions.pred_max(pred, l, settings.landmarks)
                pred_max_x, pred_max_y, pred_max_z =  pred_coords_max[i][0], pred_coords_max[i][1], pred_coords_max[i][2] 
                settings.landmark_locations_test_set[patients[i]] = {}
                settings.landmark_locations_test_set[patients[i]]['x'] = pred_max_x
                settings.landmark_locations_test_set[patients[i]]['y'] = pred_max_y
                settings.landmark_locations_test_set[patients[i]]['z'] = pred_max_z   
        data_loaders.dataset.__test_crop__()
        image = batch['image']
        pred = model(image, crop = True)    
        # just use structure original and don't use structure. EASIER
        #structure = batch['structure_original']
        original_img = batch['img_original']
        
        images.append(image)
        preds.append(pred)
        original_imgs.append(original_img)
        #structures.append(structure)
        
        print('patient')
        print(patients)
        
    images.to(settings.device)
    preds.to(settings.device)
    original_imgs.to(settings.device)
    #structures.to(settings.device)
    
    # change root to Oli then eval
    settings.root = r'/home/olive/data/Facial_asymmetry_oli_common'
    model.evaluate_line(fold, 'oli', images, preds)
    # change root to Aaron then eval
    settings.root = r'/home/olive/data/Facial_asymmetry_aaron_common'
    model.evaluate_line(fold, 'aaron', images, preds)
    # change root to Combined then eval
    settings.root = r'/home/olive/data/Facial_asymmetry_combined' 
    model.evaluate_line(fold, 'combined', images, preds)
    #print('error counter')
    #print(settings.error_counter)
