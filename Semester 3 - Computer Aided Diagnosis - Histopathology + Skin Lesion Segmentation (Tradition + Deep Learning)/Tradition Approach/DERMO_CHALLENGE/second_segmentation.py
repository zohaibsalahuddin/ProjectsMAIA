#######################################################################
# GENERATE THE SUPERPIXEL SEGMENTATION SEPARATELY FOR TRAINING, VALIDATION
# AND THE TESTING SET IN SEPARATE FOLDER. YOU WILL HAVE TO RUN THIS EACH TIME.
# IT TAKES ORIGINAL UNCHANGED IMAGE AS AN INPUT.
#######################################################################
import numpy as np
import numpy.ma as ma
import cv2
import os
import glob
import matplotlib.pyplot as plt
import math
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_uint
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from skimage.morphology import disk, dilation, erosion
from skimage.feature import greycomatrix, greycoprops

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage import exposure

filenames = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\nv\\*.jpg")
filenames.sort()
k=0
features_length = 6
features_pos = np.zeros((len(filenames),features_length))

for img in filenames:
    k = k+1
    print(k)
    print(img)
    cl1 = cv2.imread(img)

    cl1 = cv2.cvtColor(cl1, cv2.COLOR_BGR2RGB)
    
    cl1_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(cl1_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(cl1, thresh2, 1, cv2.INPAINT_TELEA)

    img_thresh = dst

    img_thresh_old = img_thresh

    from skimage.color import rgb2gray
    from skimage.exposure import equalize_hist, equalize_adapthist
    from skimage.filters import threshold_otsu
    from skimage.draw import ellipse


    def get_mask(s):
        mask = np.zeros(s, dtype=np.uint8)
        rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 2) - 1, round(s[1] / 2) - 1)
        mask[rr, cc, :] = 1
        return mask

    #Generate the superpixels
    img_thresh = img_thresh_old
    test = get_mask(img_thresh.shape)
    img_thresh = img_thresh * test
    ellipse_inv = np.ones(test.shape) - test

    numSegments = 200
    compact = 20
    segments = slic(img_thresh, n_segments=numSegments, enforce_connectivity=True, multichannel=True, convert2lab=True)

    superpixels_segment = segments

    mean_rgb = np.mean(img_thresh, axis=(0, 1))
    img_minusrgb = np.zeros(img_thresh.shape)
    img_minusrgb[:, :, 0] = img_thresh[:, :, 0] - mean_rgb[0]
    img_minusrgb[:, :, 1] = img_thresh[:, :, 1] - mean_rgb[1]
    img_minusrgb[:, :, 2] = img_thresh[:, :, 2] - mean_rgb[2]

    numRegions = superpixels_segment.max()

    count = 0
    for i in range(numRegions):
        temp_t = (superpixels_segment == i)
        if (np.sum(temp_t * ellipse_inv[:, :, 0]) > 0):
            count = count + 1

    valid_super = superpixels_segment.max() - count
    feature_vector = np.zeros((valid_super, 4))

    cl1m = cl1

    index_track = 0;
    for i in range(numRegions):
        segments = (superpixels_segment == i)
        temp = np.sum(segments)
        if (np.sum(segments * ellipse_inv[:, :, 0]) > 0):
            index_track = index_track + 1
        else:
            feature_vector[i - index_track, 0] = i
            feature_vector[i - index_track, 1] = np.sum((segments * cl1[:, :, 1]) / temp)
            feature_vector[i - index_track, 2] = np.sum((segments * cl1[:, :, 2]) / temp)
            feature_vector[i - index_track,3] = np.sum((segments * img_minusrgb[:,:,1])/ temp )

    from sklearn.cluster import KMeans
    #K-means for merge the superpixels with similar intensities
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_vector[:, 1:])

    segments = superpixels_segment
    mask = np.zeros(img_thresh.shape[:2])
    for i in range(valid_super):
        num_seg = feature_vector[i, 0]
        if kmeans.labels_[i] == 1:
            mask[segments == num_seg] = 1
        else:
            mask[segments == num_seg] = 0

    lesion = np.sum(mask * cl1_gray[:, :] * test[:, :, 1])
    les_c = np.sum((mask * cl1_gray[:, :] * test[:, :, 1]) > 0)
    lesion = lesion / les_c
    background = np.sum((((np.ones(mask.shape) - mask) * cl1_gray[:, :]) * test[:, :, 1]))
    back_c = np.sum((((np.ones(mask.shape) - mask) * cl1_gray[:, :]) * test[:, :, 1]) > 0)
    background = background / back_c

    print(background, lesion)
    if (background < lesion):
        mask = (np.ones(mask.shape) - mask) * test[:, :, 1]

    print("Values: ", lesion, background)

    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask, kernel, iterations=1)
    img_erosion = np.uint8(img_erosion)

    img_blob = np.zeros(img_erosion.shape)

    img_blob = img_erosion
    _, contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seg = len(contours)
    mask = np.zeros([seg, img_blob.shape[0], img_blob.shape[1]], np.uint8)
    for i in range(seg):
        cv2.drawContours(mask[i, :, :], [contours[i]], -1, 1, -1)

    biggest_contour = 0
    test_segment = np.zeros(cl1.shape[:2])

    for i in range(seg):
        biggest_cand_x = np.sum(np.where(mask[biggest_contour, :, :] > 0)[0]) / np.sum(
            len(np.where(mask[biggest_contour, :, :] > 0)[0]))
        biggest_cand_y = np.sum(np.where(mask[biggest_contour, :, :] > 0)[1]) / np.sum(
            len(np.where(mask[biggest_contour, :, :] > 0)[1]))
        center_cand_x = np.sum(np.where(mask[i, :, :] > 0)[0]) / np.sum(len(np.where(mask[i, :, :] > 0)[0]))
        center_cand_y = np.sum(np.where(mask[i, :, :] > 0)[1]) / np.sum(len(np.where(mask[i, :, :] > 0)[1]))
        mean_img_x = cl1.shape[0] / 2
        mean_img_y = cl1.shape[1] / 2
        error_big_x = biggest_cand_x - mean_img_x
        error_big_y = biggest_cand_y - mean_img_y
        if (abs(center_cand_x - mean_img_x) < 150) and (abs(center_cand_y - mean_img_y) < 200):

            error = abs(biggest_cand_x - mean_img_x) + abs(biggest_cand_y - mean_img_y)
            error_cand = abs(center_cand_x - mean_img_x) + abs(center_cand_y - mean_img_y)
            cand = i

            if (error_cand <= error or error_big_x > 150 or error_big_y > 200):
                biggest_contour = i;
   

    for i in range(seg):
        if (np.sum(mask[i, :, :]) < 50):
            leave = 0
        else:
            max_count = np.sum(mask[biggest_contour, :, :])
            max_sum_color = np.sum(cl1_gray * mask[biggest_contour, :, :])
            max_mean = max_sum_color / max_count

            seg_count = np.sum(mask[i, :, :])
            seg_sum_color = np.sum(cl1_gray * mask[i, :, :])
            seg_mean = seg_sum_color / seg_count

            if (abs(seg_mean - max_mean) < 15):
                test_segment[(mask[i, :, :] > 0)] = 255

    if (np.sum(test_segment) == 0):
        s = test_segment.shape
        rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 4) - 1, round(s[1] / 4) - 1)
        test_segment[rr, cc] = 255


    print("writing")
    path = "C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\nv_mask\\" + "nv"+ "_" + str(k) + ".jpg"

    print(path)
    cv2.imwrite(path,test_segment)

