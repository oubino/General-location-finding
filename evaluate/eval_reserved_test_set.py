# script to test on images

from data_loading import data_loaders
from model import load_model
from useful_functs import yes_or_no
import settings

settings.init()
settings.init_load()

# cts and structures & csv path
settings.root = r'/home/rankinaaron98/Facial_asymmetry_aaron_testset'

# evaluate model

# initialise data loader
data_loaders.init_reserved_test_set()

# load in appropriate model
load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
model = load_model.load_model(load_transfered_model)
print('\n')
print('Evaluating model')
print('----------------')
model.evaluate_pre_train()

initialise = yes_or_no.question('initialise new network?')

if initialise == True:
    settings.init_new()
    from model import init_k_fold
    fold_init = input ("Fold to initialise from, if want to do full folds then put 0, if want to train folds 3,4,5 put 2 (folds are 0,1,2,3,4) ")
    # assumption here is that folds are done in same way everytime!!
    for i in range(3):
        fold_init = i
        init_k_fold.init(fold_init)
        print('test ids for each fold')
        print(settings.k_fold_ids)
    

elif initialise == False:
    settings.init_load()
    from model import load_k_fold
    load_k_fold.init(settings.fold_load)
    print('test ids for fold (i.e. double check)')
    print(settings.k_fold_ids)
    

