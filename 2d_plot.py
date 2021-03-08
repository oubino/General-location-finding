# plot 2D slice

import data_loaders as D
import matplotlib
import matplotlib.pyplot as plt

image = D.train_set.__getitem__(0)['image']
structure = D.train_set.__getitem__(0)['structure']

print(structure.max())

img = image.numpy().squeeze(0)
structure = structure.numpy().squeeze(0)

print(img.shape)
plt.figure()
img = img[:, :, 50]
structure = structure[:, :, 50]

plt.imshow(img,cmap = 'Greys_r', alpha = 0.5)

# define the colors
cmap = matplotlib.colors.ListedColormap(['0','r', 'k','b','g','y','m'])

# create a normalize object the describes the limits of
# each color
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


# plot it
plt.imshow(structure, cmap = cmap, norm = norm, alpha = 0.5 )
plt.show()