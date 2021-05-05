# script to test on images


from useful_functs import yes_or_no
import settings
settings.init(True) # rts is True
settings.root = r'/home/oli/data/paed_dataset/test'
settings.init_load()
from data_loading import data_loaders
from model import load_model   

# initialise data loader
data_loaders.init_reserved_test_set()

# init slid window
settings.init_slide_window(data_loaders.test_set_ids)

# load in appropriate model
load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
model = load_model.load_model(load_transfered_model)
print('\n')
print('Evaluating model')
print('----------------')
model.evaluate_pre_train(fold = settings.fold_load)

