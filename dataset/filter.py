 
import os
import cv2
import numpy as np

path = "F:/OCT/picture/"

f_list = os.listdir(path)

for file in f_list:
    ext_name = file.split(" ")[2]
    print(ext_name)
    if ext_name[7] == "a":
        os.remove(os.path.join(path, file))