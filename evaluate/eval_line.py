# script to eval line model on aaron/oli and combined

from model import load_model
from data_loading import data_loaders
from useful_functs import yes_or_no
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
    
    # need to get images and predictions here
    for batch in data_loaders.dataloaders['test']:
        image = batch['image']
        pred = model(image)
        images.append(image)
        preds.append(pred)
        
    images.to(settings.device)
    preds.to(settings.device)
    
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
