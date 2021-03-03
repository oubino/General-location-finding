# General-location-finding
Code for general location finding in 3D CT scans 
- UNET based off Ronenberger et al
- SCN based off Payer et al
- heatmap regression (sigma trainable parameter) based off Payer et al

All necessary modifications to the parameters etc. can be done in settings.py.

To train/load in/evaluate the model run main.py and choose the corresponding option

Trained off dataset of CT scans/structures of Head and Neck open source - see https://arxiv.org/pdf/1809.04430.pdf
- structures trained for 1:left parotid gland 2: right parotid gland 3: brainstem 4: spinal chord 5: left cochlea 6:right cochlea
