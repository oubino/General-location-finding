# init_k_fold

import time
from sklearn.model_selection import KFold

import settings
from model import initialise_model
from data_loading import data_loaders



def init(init_fold):
    
    if init_fold == None:
        # i.e. don't want to do k fold learning
        start_time = time.time()
        data_loaders.init_no_k_fold()   
        model = initialise_model.initialise_model()
        print('Training model')
        print('--------------')
        model.train(True)
        print('Saving model to files')
        print('------------')
        model.save(None) 
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
            model.save(None)
        print('\n')
        print('Evaluating model')
        print('----------------')
        # change batch size to 1 pre eval back to original after
        batch_size_original = settings.batch_size
        settings.batch_size = 1
        model.evaluate(None)   
        settings.batch_size = batch_size_original
        print('error counter')
        print(settings.error_counter)

    else:  
        kfold = KFold(n_splits = settings.k_folds, shuffle = False)
        
        list_splits = list(kfold.split(data_loaders.dataset))
        
        for fold, (train_ids, test_ids) in enumerate(list_splits[int(init_fold):], int(init_fold)):
            print('------')
            print('fold')
            print(fold)
            print('------')
            print('train ids/test ids')
            print(train_ids, test_ids)
             
        # in case want to initialise from not fold 0
        for fold, (train_ids, test_ids) in enumerate(list_splits[int(init_fold):], int(init_fold)):
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
                print('Estimated finish time of this fold: ', end_time)
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
    

    

    
    
    