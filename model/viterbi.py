import os

import cv2
import math
import numpy as np

from utils import trellis, funcEnergy


def viterbi(energy_net, x_coords, y_coords, num_layer=1):
    '''
    Find the boundary with the lowest energy

    Args:
      energy_net: (ndarray) (u, u, q)
      x_coords: (ndarray) (q,)
      y_coords: (ndarray) (u,)
      num_layer: (int) number of segmentation layers
    '''

    boundary = []
    x_coords = np.reshape(x_coords, (energy_net.shape[0], energy_net.shape[2]))
    y_coords = np.reshape(y_coords, (energy_net.shape[0], energy_net.shape[2]))
    for l in range(num_layer):
        layer = []      # index in image
        y_loc = []      # index in energy net
        for q in range(energy_net.shape[2]):
            # find the best point in normal q of layer l
            if q == 0:
                energy = energy_net[:, 0, q]    # (u,)               
            else:
                energy = energy_net[:, y_loc[q-1], q]

            loc = np.argmin(energy)
            layer.append((x_coords[loc, q], y_coords[loc, q]))
            y_loc.append(loc)
            # set the energy of this point to inf
            energy_net[loc, :, q] = np.array([np.inf]*energy_net.shape[1])
        boundary.append(layer)
    
    return boundary


def segVis(img, boundary):
    '''
    Draw the line in the image

    Args:
      img: (ndarray) (height, width, channel)
      boundary: (nadrray) (layer, number) 
                each element is a tuple (y, x)
    '''
    for l in range(len(boundary)):
        # draw layer boundary iteratively
        for i in range(len(boundary[0])-1):
            start = boundary[l][i]
            end = boundary[l][i+1]

            cv2.line(img, start, end, (0, 0, 255), 2)
    
    plt.imshow(img)


if __name__ == "__main__":
