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
      x_coords: (tuple)
      y_coords: (tuple)
    '''
    height, width = image.shape
    assert q <= width, "number of normals is out of width"
    assert u <= height, "number of nodes per normal is out of height"

    x = np.arange(q)
    y = np.arange(u)
    xx, yy = np.meshgrid(x, y)    #meshgride index
    gs_x = width/q
    gs_y = height/u
    x_coords = tuple(np.reshape(np.rint(xx * gs_x), -1))      # length Q*U
    y_coords = tuple(np.reshape(np.rint(yy * gs_y), -1))      # length Q*U

    return x_coords, y_coords


def energy_func(image, q, u, tau):
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
    img = cv2.GaussianBlur(image, (3, 3), 5)              # Blur image by Gaussian filter
    grad_y = cv2.Sobel(img, -1, 0, 1)                     # Compute image's gradiant along the Y axis
    grad_nodes = grad_y[y_coords, x_coords]               # shape (U, Q)
    phi = 1/(1 + tau * np.power(grad_nodes, 2))           # phi function, (U, Q)
    dist = np.zeros((u, u, q), dtype=np.float32)          # distance matrix, distances between each node in the q colums and each node in the q-1 columns
    # element (i, j, k) in dist means the distance between 
    # the i-th point in the k-th normal and the j-th point in the (k-1)-th normal
    xx = np.reshape(np.array(x_coords, dtype=np.float32), (u,q))
    yy = np.reshape(np.array(y_coords, dtype=np.float32), (u,q))
    energy = np.zeros((u, u, q), dtype=np.float32)
    for i in range(q):
        phi_i = np.expand_dims(phi[..., i], 1)     # (U, 1)
        phi_i = np.repeat(phi_i, u, 0)       # (U, U)
        # numbers of a row are the same
        if i == 0:
            dist[..., i] = 1
        else:
            dx = np.power(xx[:, i]-xx[:, i-1], 2)      # (U, )
            dx = np.expand_dims(dx, axis=0)    #(1, U)
            dx = np.repeat(dx, u, axis=0)    #(U, U)
            # distance of Y coordinates in i-1 columns
            dy0 = np.expand_dims(yy[:, i-1], 0)     # (1, U)
            dy0 = np.repeat(dy0, u, 0)      #(U, U)
            # distance of Y coordinates in i columns
            dy1 = np.repeat(yy[:, i], 0)    #(U, )
            dy1 = np.reshape(dy1, (u,u))    #(U, U)
            dy = np.power(dy1 - dy0, 2)

            dist[..., i] = np.sqrt(dx+dy)
        energy[..., i] = phi_i * dist[..., i]

    return energy


