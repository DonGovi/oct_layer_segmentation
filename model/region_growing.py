import os

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def imgGrad(img, kernel_size=11):
    med = cv2.medianBlur(img, kernel_size)
    gradY = cv2.Sobel(med, cv2.CV_64F, 0, 1)
    
    return gradY


def funcPhi(grad, tau=1):
    phi = 1/(1 + tau * np.power(grad, 2))
    
    return phi


def bdGrow(grad_img, start, area=5, tau=1, weight=None):
    height, width = grad_img.shape
    x, y = start
    boundary = [start]
    while(x+1 < width):
        x += 1     # shift one pixel to right
        offset = np.arange(area) - int(area/2)
        y_coords = y + offset
        x_coords = np.array([x]*area)
        grad = grad_img[y_coords, x_coords]
        if weight is None:
            energy = funcPhi(grad, tau)
        else:
            dx = 1
            dy = np.power(offset, 2)
            dist = np.sqrt(dx + dy)
            energy = funcPhi(grad, tau) + weight * dist
        loc = np.argmin(energy)
        y += loc - int(area/2)
        seg_point = (x, y)
        boundary.append(seg_point)
    
    return boundary