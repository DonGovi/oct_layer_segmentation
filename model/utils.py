import os

import cv2
import math
import numpy as np



def trellis(image, q, u):
    '''
    initialize sparse trellis with Q*U in the image.

    Args:
      image: (ndarray) (height, width)
      q: (int) number of normals
      u: (int) number of nodes in each normal

    Return:
      x_coords: (ndarray) coordinates in width (q)
      y_coords: (ndarray) coordinates in height (u)
    '''
    height, width = image.shape
    assert q <= width, "number of normals is out of width"
    assert u <= height, "number of nodes per normal is out of height"

    x = np.arange(q)
    y = np.arange(u)
    xx, yy = np.meshgrid(x, y)    #meshgride index

    gs_x = width/q
    gs_y = height/u
    x_coords = np.reshape(np.rint(xx * gs_x), -1).astype(np.int32)      # (Q*U,)
    y_coords = np.reshape(np.rint(yy * gs_y), -1).astype(np.int32)      # (Q*U,)

    return x_coords, y_coords


def funcEnergy(image, q, u, tau, w=0.5, kernel=(9,9), sigma=100):
    '''
    Build energy matrix with U*Q

    Args:
      image: (ndarray) (height, width)
      q: (int) number of normals
      u: (int) number of nodes in each normal
    
    Return:
      energy: (ndarray) (U, U, Q) numbers in the first column is the phi function,
                               numbers in others is the phi function plus the distance betweeen node (q-1, i) and (q, j)
    '''
    x_coords, y_coords = trellis(image, q, u)
    xx = np.reshape(np.array(x_coords, dtype=np.float32), (u,q))
    yy = np.reshape(np.array(y_coords, dtype=np.float32), (u,q))

    #img = cv2.GaussianBlur(image, kernel, sigma)              # Blur image by Gaussian filter
    grad_y = cv2.Sobel(img, -1, 0, 1)                     # Compute image's gradiant along the Y axis
    grad_nodes = grad_y[y_coords, x_coords]               # (U*Q,)
    grad_nodes = np.reshape(grad_nodes, (u, q))     # (U, Q)
    phi = 1/(1 + tau * np.power(grad_nodes, 2))           # phi function, (U, Q)
    dist = np.zeros((u, u, q), dtype=np.float32)          # distance matrix, distances between each node in the q colums and each node in the q-1 columns
    # element (i, j, k) in dist means the distance between 
    # the i-th point in the k-th normal and the j-th point in the (k-1)-th normal

    energy = np.zeros((u, u, q), dtype=np.float32)
    for i in range(q):
        phi_i = np.expand_dims(phi[..., i], 1)     # (U, 1)
        phi_i = np.repeat(phi_i, u, 0)       # (U*U, )
        phi_i = np.reshape(phi_i, (u, u))    # (U, U)
        # numbers in the same row are the same
        if i == 0:
            dist[..., i] = 0
        else:
            dx = np.power(xx[:, i]-xx[:, i-1], 2)      # (U, )
            dx = np.expand_dims(dx, axis=0)    #(1, U)
            dx = np.repeat(dx, u, axis=0)    #(U, U)
            
            # distance of Y coordinates in i-1 columns
            dy0 = np.expand_dims(yy[:, i-1], axis=0)     # (1, U)
            dy0 = np.repeat(dy0, u, axis=0)      #(U, U)
            # distance of Y coordinates in i columns
            dy1 = np.repeat(yy[:, i], u, 0)    #(U*U, )
            dy1 = np.reshape(dy1, (u,u))    #(U, U)
            dy = np.power(dy1 - dy0, 2)

            dist[..., i] = np.sqrt(dx+dy)
        energy[..., i] = phi_i + w*dist[..., i]

    return energy, x_coords, y_coords




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    file = "F:/OCT/picture/00737898_Chen_Jianguo__60281_Angio Retina_OS_2018-01-04_09-23-01_M_1953-02-10_Main Reportcb.jpg"
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    energy, x_coords, y_coords = funcEnergy(img, 200, 100, 3)
    print(energy[...,0])



