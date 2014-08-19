
import pybrisk
import numpy as np
import cv2
import numpy as np

def get_keypoints(img):
    kp_array = pybrisk.detect(img)
    kp = []
    for i in range(kp_array.shape[0]):
        kp.append(cv2.KeyPoint())
        kp[i].pt = (kp_array[i, 0], kp_array[i, 1])
        kp[i].size = kp_array[i, 2]
        kp[i].angle = kp_array[i, 3]
        kp[i].response = kp_array[i, 4]
    return kp, kp_array

def get_features(img, kp_array):
    features_and_kp = pybrisk.compute(img, kp_array)
    features = features_and_kp[0]
    kp_array_new = features_and_kp[1]
    kp = []
    for i in range(kp_array_new.shape[0]):
        kp.append(cv2.KeyPoint())
        kp[i].pt = (kp_array_new[i, 0], kp_array_new[i, 1])
        kp[i].size = kp_array_new[i, 2]
        kp[i].angle = kp_array_new[i, 3]
        kp[i].response = kp_array_new[i, 4]
    return kp, features
