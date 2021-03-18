# script to test on images

import data_loaders
import load_model
import yes_or_no
import settings

settings.init()

# cts and structures & csv path
settings.root = r'/home/olive/data/Test_set'


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

