# global variable settings
import settings
settings.init()

# library imports
#import functions
import time
import yes_or_no
import init_k_fold
import load_no_k_fold
import load_k_fold


# initialise or load model
initialise = yes_or_no.question('initialise new network?')
#save_model = functions.yes_or_no('would you like to save model at end?')

if initialise == True:
    fold_init = input ("Fold to initialise from, if want to do full folds then put 0, if want to train folds 3,4,5 put 2 (folds are 0,1,2,3,4). If want no folds then put None. ")
    # assumption here is that folds are done in same way everytime!!
    init_k_fold.init(fold_init)
    print('test ids for each fold')
    print(settings.k_fold_ids)

elif initialise == False:
    load_k_fold.init(settings.fold_load)
    print('test ids for fold (i.e. double check)')
    print(settings.k_fold_ids)
    

