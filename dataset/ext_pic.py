
import os
import cv2
import numpy as np

path = "F:/OCT/normal_10-7/"
save_path = "F:/OCT/picture/"

f_list = os.listdir(path)

for file in f_list:
    arr = cv2.imread(os.path.join(path, file))
    pic = arr[566:975, 99:955]
    cv2.imwrite(os.path.join(save_path, file), pic)