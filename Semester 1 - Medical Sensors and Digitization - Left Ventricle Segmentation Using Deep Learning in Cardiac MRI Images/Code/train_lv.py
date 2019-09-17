
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras import backend as K

import tensorflow as tf

import nibabel as nb 

import data_lv as dlv

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
TRAIN_PATH = 'D:/tensorflow-keras/LV_MRI_Segmentation/data/training/'
TEST_PATH = 'D:/tensorflow-keras/LV_MRI_Segmentation/data/testing/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

#dlv.create_traindata()
#dlv.create_testdata()

# load data from 
img_train = np.load('train_data2.npy')
mask_train = np.load('train_mask2.npy')

img_test = np.load('test_data.npy')
mask_test = np.load('test_mask.npy')

print (img_train.shape)
print(mask_train.shape)

# Build U-Net model
#def get_unet():
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

# sgd = SGD(lr=0.000001, decay=1e-6, momentum=1.9)
model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

#Train Model
# Fit model
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('model-lv_segmentation2.h5', verbose=1, save_best_only=True)

results = model.fit(img_train, mask_train, validation_split=0.1, batch_size=32, epochs=250, 
                    callbacks=[earlystopper, checkpointer])



# Predict on train, val and test
# model = load_model('model-lv_segmentation.h5')

# preds_test = model.predict(img_test, verbose=1)

# # Threshold predictions if it is > 0.5 true, ow false
# preds_test_t = (preds_test > 0.5).astype(np.uint8)

# np.save("predictedtest_mask.npy",preds_test_t)

# def display(Test_gt, Test_predicted):
# 	plt.subplot(1,2,1)
# 	plt.imshow(Test_gt[:,:,0], cmap = plt.cm.gray)
# 	plt.subplot(1,2,2)
# 	plt.imshow(Test_predicted[:,:,0], cmap = plt.cm.gray)
# 	plt.show()

 # def dice_coef(y_true, y_pred):
 #     y_true_f = K.flatten(y_true)
 #     y_pred_f = K.flatten(y_pred)
 #     intersection = K.sum(y_true_f * y_pred_f)
 #     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)


# # Compare the predicated mask with original ground truth of test data

# #Display result
# print(mask_test.shape)
# print(preds_test_t.shape)
# display(mask_test[3], preds_test_t[3])

# loss_test= dice_coef_loss(mask_test[3],preds_test_t[3])
# print(loss_test)

