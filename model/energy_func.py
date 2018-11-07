import os

import cv2
import numpy as np


def trellis(image, q, u):
    '''
    initialize sparse trellis with q*u in the image.

    Args:
    image: (ndarray) (height, width)
    q: (int) number of normals
    u: (int) number of nodes for each normal

    Returns:
    x_coords: (tuple)
    y_coords: (tuple)
    '''
    height, width = image.
    assert q <= width, "number of normals is out of width"
    assert u <= height, "number of nodes per normal is out of height"

    x = np.arange(q)
    y = np.arange(u)
    xx, yy = np.meshgrid(x, y)    #meshgride index
    gs_x = width/q
    gs_y = height/w
    x_coords = tuple(np.reshape(np.rint(xx * gs_x), -1))
    y_coords = tuple(np.reshape(np.rint(yy * gs_y), -1))

    return x_coords, y_coords


def energy_func(image, q, u, tau):
    x_coords, y_coords = trellis(q, u)
    grad_y = cv2.Sobel(image, -1, 0, 1)
    grad_node = grad_y[y_coords, x_coords]
    phi = 1/(1 + tau * np.power(grad_node, 2))