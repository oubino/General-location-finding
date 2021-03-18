# init_k_fold

import settings
import initialise_model
import time
import data_loaders
from sklearn.model_selection import KFold



def init(init_fold):
    
    kfold = KFold(n_splits = settings.k_folds, shuffle = False)
    list_splits = list(kfold.split(data_loaders.dataset))
    print('list splits')
    print(list_splits)
         
    for fold, (train_ids, test_ids) in enumerate(list_splits[init_fold:], init_fold):
        # i.e. if init fold is 1 then skips first fold when initialising
        start_time_fold = time.time()
        # different dataloader for each fold
        print('fold')
        print(fold)
        data_loaders.init(fold, train_ids, test_ids)     
        model = initialise_model.initialise_model()
        print('Training model')
        print('--------------')
        model.train(True)
        print('Saving model to files')
        print('------------')
        model.save(fold)
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
            model.save(fold)
        print('\n')
        print('Evaluating model')
        print('----------------')
        model.evaluate(fold)   
        print('error counter')
        print(settings.error_counter)
        time_elapsed_fold = time.time() - start_time_fold
        end_time_fold = time.ctime(time_elapsed_fold * (settings.k_folds - fold - 1) + time.time())
        print('\n')
        print('Estimated finish time of all folds: ', end_time_fold)
    

    

    
    
    