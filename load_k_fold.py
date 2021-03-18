# load k fold

import load_model
import data_loaders
import yes_or_no
import settings

def init():
    # initialise data loader
    data_loaders.init_load_k_fold(settings.fold_load)
    
    load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
    model = load_model.load_model(load_transfered_model)
    train_decision = yes_or_no.question('train loaded in model?')
    if train_decision == True:
        freeze_decision = yes_or_no.question('freeze all but last layer?')
        if freeze_decision == True:
            model.freeze_final_layers()
        transfer_learn_decision = yes_or_no.question('transfer learn to new number of classes')
        if transfer_learn_decision == True:
            class_number = input ("New number of classes")
            feature_number = input ("Enter number of features pre trained model trained with")
            class_number = int(class_number)
            feature_number = int(feature_number)
            model.transfer_learn_unet_final_layer(class_number, feature_number)
        print('Training model')
        print('------------')
        model.train(True, transfer_learn_decision)
        print('Saving model to files')
        print('------------')
        model.save()
        for i in range(settings.num_epoch_batches - 1):
            print('Training model')
            print('------------')
            model.train(False, transfer_learn_decision)
            print('Saving model to files')
            print('------------')
            model.save()
            print('\n')
            print('Evaluating model')
            print('----------------')
        model.evaluate_post_train()
    elif train_decision == False:
        print('\n')
        print('Evaluating model')
        print('----------------')
        model.evaluate_pre_train()
    print('error counter')
    print(settings.error_counter)
