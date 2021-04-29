# script to test on images


from useful_functs import yes_or_no
import settings
print('here')
settings.init()
print('here 2')
settings.init_load()
print('here 3')
from data_loading import data_loaders
print('here 4')
from model import load_model   
# cts and structures & csv path
settings.root = r'/home/rankinaaron98/Facial_asymmetry_test_sets'
print('here 5')
# evaluate model
'''downsample_q = input ("Downsample(y)/crop(n) ")
if downsample_q == 'y':
     downsample_user = True 
elif downsample_q == 'n':  
     downsample_user = False
''' 

# initialise data loader
data_loaders.init_reserved_test_set()
print('here 6')
# load in appropriate model
load_transfered_model = yes_or_no.question('are you loading in a model which was saved as a transfered model')
model = load_model.load_model(load_transfered_model)
print('\n')
print('Evaluating model')
print('----------------')
model.evaluate_pre_train(fold = settings.fold_load)

