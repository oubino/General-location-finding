# init_no_k_fold
import settings
import initialise_model
import time

def init():
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