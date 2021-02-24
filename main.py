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
    initialise_model.init()
    print('Training model')
    print('--------------')
    initialise_model.train(True)
    print('Saving model to files')
    print('------------')
    initialise_model.save()
    for i in range(settings.num_epoch_batches - 1):
        start_time = time.time()
        print('Training model')
        print('--------------')
        initialise_model.train(False)
        time_elapsed = time.time() - start_time
        end_time = time.ctime(time_elapsed * (settings.num_epoch_batches - i - 2) + time.time())
        print('\n')
        print('Estimated finish time: ', end_time)
        print('\n')
        print('Saving model to files')
        print('------------')
        initialise_model.save()
    print('\n')
    print('Evaluating model')
    print('----------------')
    initialise_model.evaluate()   
    print('error counter')
    print(settings.error_counter)
        

elif initialise == False:
    import load_model
    load_model.init()
    train_decision = yes_or_no.question('train loaded in model?')
    if train_decision == True:
        print('Training model')
        print('------------')
        load_model.train(True)
        print('Saving model to files')
        print('------------')
        load_model.save()
        for i in range(settings.num_epoch_batches - 1):
            print('Training model')
            print('------------')
            load_model.train(False)
            print('Saving model to files')
            print('------------')
            load_model.save()
            print('\n')
            print('Evaluating model')
            print('----------------')
        load_model.evaluate_post_train()
    elif train_decision == False:
        print('\n')
        print('Evaluating model')
        print('----------------')
        load_model.evaluate_pre_train()
    print('error counter')
    print(settings.error_counter)
