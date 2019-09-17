# This function is used to calculate standard deviation of the givenn
# 4d image to calculate the ROI.

import os
import nibabel as nib
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import numpy as np

def std_mri(filename):
    data_4d = nib.load(filename)
    data = data_4d.get_data();
    size_4d = data_4d.shape;
    total_t = size_4d[3]



    x_co = size_4d[0]/2;
    y_co = size_4d[1]/2;

    center_image_x = x_co
    center_image_y = y_co
    total_x = size_4d[0]
    total_y = size_4d[1]
    slice_number = int(size_4d[2]/2)


    for i in range(total_t):
        a = data[:, :, slice_number, i]
        a = ndimage.median_filter(a,6)

    mean_data = np.zeros(shape=(total_x, total_y))
    for i in range(total_t):
        mean_data = mean_data + data[:, :, slice_number, i]

    mean_data /= total_t

    std_data = np.zeros(shape=(total_x, total_y))
    std_final = np.zeros(shape=(total_x, total_y))

    for i in range(total_t):
        std_data = data[:, :, slice_number, i] - mean_data
        std_data = np.square(std_data)
        std_final = std_final + std_data

    std_final /= (total_t -1)
    std_final = np.sqrt(std_final)

    std_final = std_final > 25

    std_final[:, 0: center_image_y - 65] = 0
    std_final[:, center_image_y + 65:] = 0
    std_final[0: center_image_x - 65, :] = 0
    std_final[center_image_x + 65:, :] = 0

    x_co = 0
    y_co = 0
    co = 0
    for i in range(total_x):
        for j in range(total_y):
            if std_final[i][j] > 0:
                x_co += i
                y_co += j
                co += 1

    if (co == 0):
        co =1
    x_co /= co
    y_co /= co

    x_co = int(x_co)
    y_co = int(y_co)
    return  x_co,y_co

print ("hello")
