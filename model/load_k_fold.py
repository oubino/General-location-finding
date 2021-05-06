# load k fold

from model import load_model
from data_loading import data_loaders
from useful_functs import yes_or_no
import settings

def init(fold):
    
    # initialise data loader
    #if fold == None:
    #    data_loaders.init_load_no_k_fold()
    #else:
    data_loaders.init_load_k_fold(int(fold))  
    settings.tensorboard_init(fold) # initialise tensorboard writer
    # initialise sliding window crop locations
    slid_wind_ids = data_loaders.test_set_ids + data_loaders.val_set_ids
    settings.init_slide_window(slid_wind_ids)

    load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
    model = load_model.load_model(load_transfered_model)
    train_decision = yes_or_no.question('train loaded in model?')
    if train_decision == True:
        # if train is true then ask number of epochs to train for otherwise don't need to as just eval
        settings.epoch_batch = int(input ("Epoch batch: "))
        settings.num_epoch_batches = 1
        freeze_decision = yes_or_no.question('freeze all but last layer?')
        if freeze_decision == True:
            model.freeze_final_layers()
        transfer_learn_decision = yes_or_no.question('transfer learn to new number of classes')
        if transfer_learn_decision == True:
            class_number = input ("New number of classes ")
            feature_number = input ("Enter number of features pre trained model trained with ")
            class_number = int(class_number)
            feature_number = int(feature_number)
            model.transfer_learn_unet_final_layer(class_number, feature_number)
        print('Training model')
        print('------------')
        model.train(True, transfer_learn_decision, fold)
        print('Saving model to files')
        print('------------')
        model.save(fold)
        for i in range(settings.num_epoch_batches - 1):
            print('Training model')
            print('------------')
            model.train(False, transfer_learn_decision, fold)
            print('Saving model to files')
            print('------------')
            model.save(fold)
            print('\n')
            print('Evaluating model')
            print('----------------')
        model.evaluate_post_train(fold)
    elif train_decision == False:
        print('\n')
        print('Evaluating model')
        print('----------------')
        model.evaluate_pre_train(fold)
    print('error counter')
    print(settings.error_counter)
