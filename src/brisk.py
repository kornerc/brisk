
import pybrisk
import numpy as np
import cv2
import numpy as np

def get_keypoints(img):
    kp_temp = pybrisk.detect(img)
    kp = []
    for i in range(kp_temp.shape[0]):
        kp.append(cv2.KeyPoint())
        kp[i].pt = (kp_temp[i, 0], kp_temp[i, 1])
        kp[i].size = kp_temp[i, 2]
        kp[i].angle = kp_temp[i, 3]
        kp[i].response = kp_temp[i, 4]
    return kp

