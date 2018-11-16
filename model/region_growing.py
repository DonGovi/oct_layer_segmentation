import os

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def imgGrad(img, kernel_size=11):
    '''
    Compute image gradient along Y axis

    Parameters:
      img: (ndarray) gray image with (height, width)
      kernel_size: (int) odd int for cv2.medianBlur's kernel
    
    Returns:
      gradY: (ndarray) gradient image with  (height, width)
    '''
    med = cv2.medianBlur(img, kernel_size)
    gradY = cv2.Sobel(med, cv2.CV_64F, 0, 1)
    
    return gradY


def funcPhi(grad, tau=1):
    '''
    Energy function based on image gradient, larger gradient lead to smaller energy

    Parameters:
      grad: (ndarray) gradient image with (height, width)
      tau: (float or int) weight parameter for energy
    
    Returns:
     phi: (ndarray) energy array
    '''
    phi = 1/(1 + tau * np.power(grad, 2))
    
    return phi
    

def getField(coord, area=5):
    offset = np.arange(area) - int(area/2)
    coords = coord + offset
    
    return coords


def bdGrow(grad_img, start, area=5, tau=1):
    height, width = grad_img.shape
    x, y = start
    y_start = getField(y, area)
    x_start = np.array([x]*area)
    start_energy = funcPhi(grad_img[y_start, x_start], tau)
    loc = np.argmin(start_energy)
    y += loc - int(area/2)
    start = (x, y)
    boundary = np.zeros(grad_img.shape[1], dtype=np.float32)
    boundary[x] = y
    
    # Search from the start point to the front
    forward_x = x
    forward_y = y
    while(forward_x > 0):
        forward_x -= 1
        y_coords = getField(forward_y, area)
        x_coords = np.array([forward_x]*area)
        grad = grad_img[y_coords, x_coords]
        energy = funcPhi(grad, tau)
        loc = np.argmin(energy)
        forward_y += loc - int(area/2)
        boundary[forward_x] = forward_y
    
    # Search from the start point to the back
    backward_x = x
    backward_y = y
    while(backward_x+1 < width):
        backward_x += 1     # shift one pixel to right
        y_coords = getField(backward_y, area)
        x_coords = np.array([backward_x]*area)
        grad = grad_img[y_coords, x_coords]
        energy = funcPhi(grad, tau)
        loc = np.argmin(energy)
        backward_y += loc - int(area/2)
        boundary[backward_x] = backward_y
    
    boundary = boundary.astype(np.int32)
    return boundary.tolist()
