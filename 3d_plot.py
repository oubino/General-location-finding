# plot 3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import data_loaders as D
from skimage import measure
import numpy as np
import settings as S
import functions


def plot_3d(image, structure, threshold_img, threshold_structure):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    #p = image.transpose(2,1,0)
    print(D.train_set.__getitem__(0)['image'].size())
    
    verts_img, faces_img = measure.marching_cubes_classic(image, threshold_img)
    verts_structure, faces_structure = measure.marching_cubes_classic(structure, threshold_structure)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh_img = Poly3DCollection(verts_img[faces_img], alpha=0.1)
    mesh_structure = Poly3DCollection(verts_structure[faces_structure], alpha=0.6)
    #face_color_img = [0.5, 0.5, 1]
    face_color_img = ['tab:gray']
    mesh_img.set_facecolor(face_color_img)
    #face_color_structure = ['r', 'b', 'g']
    face_color_structure = ['tab:red']
    mesh_structure.set_facecolor(face_color_structure)
    ax.add_collection3d(mesh_img)
    ax.add_collection3d(mesh_structure)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])

    ax.invert_xaxis()

    # rotate the axes and update
    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    plt.show()

image_3d = D.train_set.__getitem__(0)['image']
print(image_3d.type)
image_3d = image_3d.squeeze(0)
image_3d = image_3d.numpy()

max_value = np.max(image_3d)
print(max_value)

structure_3d = D.train_set.__getitem__(0)['structure']
structure_3d = structure_3d.squeeze(0)
structure_3d = structure_3d.numpy()

for l in S.landmarks:
    print('com of structure')
    print(functions.com_structure(D.train_set.__getitem__(0)['structure'].unsqueeze(0), l))

plot_3d(image_3d,structure_3d,max_value/2, 6)

