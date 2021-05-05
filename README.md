# General-location-finding
Code for general location finding in 3D CT scans 
- UNET based off Ronenberger et al
- SCN based off Payer et al
- heatmap regression (sigma trainable parameter) based off Payer et al

All necessary modifications to the parameters etc. can be done in settings.py.

To train/load in/evaluate the model run main.py and choose the corresponding option

Trained off dataset of CT scans/structures of Head and Neck open source - see https://arxiv.org/pdf/1809.04430.pdf
- structures trained for 1:left parotid gland 2: right parotid gland 3: brainstem 4: spinal chord 5: left cochlea 6:right cochlea
- modified to train for 1: Angle of mandible (L) 2: Angle of mandible (R) 3: Head of mandible (L) 4: Head of mandible (R) 5: Frontal process of zygomatic bone (L) 6: Frontal process of zygomatic bone (R) 7: Frontal nasal process (L) 8: Frontal nasal process (R) 9: Superior orbital fissure (L) 10: Superior orbital fissure (R)

Data loading:
- Convert all .nii to .py files (i.e. CTs and structures - where it's assumed structures are same size as CTs with values at the locations e.g. for patient 0002.npy, there will be one CT and one Structure -> structure will have 1s at location of AMl, 2s at AMr, etc. and zero elsewhere)

Data conversion:
- 
