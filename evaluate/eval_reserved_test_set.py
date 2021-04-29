# script to test on images

from data_loading import data_loaders
from model import load_model
from useful_functs import yes_or_no
import settings

<<<<<<< HEAD:evaluate/eval_reserved_test_set.py
settings.init(True)

=======
settings.init()
settings.init_load()
>>>>>>> parent of bf55f8c (fixing eval_reserve):eval_reserved_test_set.py

# cts and structures & csv path
settings.root = r'/home/rankinaaron98/Facial_asymmetry_test_sets'

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

