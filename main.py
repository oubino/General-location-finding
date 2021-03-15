# global variable settings
import settings
settings.init()

# library imports
#import functions
import time
import yes_or_no


# initialise or load model
initialise = yes_or_no.question('initialise new network?')
#save_model = functions.yes_or_no('would you like to save model at end?')

if initialise == True:
    import initialise_model
    model = initialise_model.initialise_model()
    print('Training model')
    print('--------------')
    model.train(True)
    print('Saving model to files')
    print('------------')
    model.save()
    for i in range(settings.num_epoch_batches - 1):
        start_time = time.time()
        print('Training model')
        print('--------------')
        model.train(False)
        time_elapsed = time.time() - start_time
        end_time = time.ctime(time_elapsed * (settings.num_epoch_batches - i - 2) + time.time())
        print('\n')
        print('Estimated finish time: ', end_time)
        print('\n')
        print('Saving model to files')
        print('------------')
        model.save()
    print('\n')
    print('Evaluating model')
    print('----------------')
    model.evaluate()   
    print('error counter')
    print(settings.error_counter)
        

elif initialise == False:
    import load_model
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
