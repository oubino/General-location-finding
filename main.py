# global variable settings
import settings
settings.init(False) # rts

# library imports
from useful_functs import yes_or_no

# initialise or load model
initialise = yes_or_no.question('initialise new network?')

if initialise == True:
    settings.init_new()
    from model import init_k_fold
    fold_init = input ("Fold to initialise from, if want to do full folds then put 0, if want to train folds 3,4,5 put 2 (folds are 0,1,2,3,4) ")
    # assumption here is that folds are done in same way everytime!!
    init_k_fold.init(fold_init)
    print('test ids for each fold')
    print(settings.k_fold_ids)
    

elif initialise == False:
    settings.init_load()
    from model import load_k_fold
    load_k_fold.init(settings.fold_load)
    print('test ids for fold (i.e. double check)')
    print(settings.k_fold_ids)
    

