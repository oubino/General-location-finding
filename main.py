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
    init_k_fold.init()

elif initialise == False:
    load_k_fold = yes_or_no.question('load in fold')
    if load_k_fold == True:
        load_k_fold.init()
    elif load_k_fold == False:
        load_no_k_fold.init()
    

