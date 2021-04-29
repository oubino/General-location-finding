# script to test on images

from data_loading import data_loaders
from model import load_model
from useful_functs import yes_or_no
import settings

settings.init(True)


# cts and structures & csv path
settings.root = r'/home/rankinaaron98/Facial_asymmetry_aaron_testset'

# evaluate model
downsample_q = input ("Downsample(y)/crop(n) ")
if downsample_q == 'y':
       downsample_user = True
elif downsample_q == 'n':
       downsample_user = False
# initialise data loader
data_loaders.init_reserved_test_set()

# load in appropriate model
load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
model = load_model.load_model(load_transfered_model)
print('\n')
print('Evaluating model')
print('----------------')
model.evaluate_pre_train()

