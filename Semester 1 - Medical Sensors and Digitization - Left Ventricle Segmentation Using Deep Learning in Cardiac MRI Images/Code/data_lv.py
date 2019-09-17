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
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
# from keras.layers.merge import concatenate
from keras.layers import Concatenate
from keras.layers import concatenate

import tensorflow as tf
# for data augmentation
import glob
from keras.preprocessing.image import ImageDataGenerator
import nibabel as nib 

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
TRAIN_PATH = 'D:/tensorflow-keras/LV_MRI_Segmentation/data/training/'
TEST_PATH = 'D:/tensorflow-keras/LV_MRI_Segmentation/data/testing/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# seed = 42
# random.seed = seed
# np.random.seed = seed
dim = 64
def create_traindata():
	# Get train and test IDs
	train_ids = next(os.walk(TRAIN_PATH))[1]	

	# Get and resize train images and masks
	X_train = [] # np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	Y_train = [] #np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

	print('Getting and resizing train images and masks ... ')
	sys.stdout.flush()


	for n in range(0,90):
	    if (n >8):
	        fold = "patient0" + str(n+1)
	    else:
	        fold = "patient00" + str(n+1)
	    f =os.listdir( TRAIN_PATH + fold)
	    #print f
	    data_ni_frame_di = nib.load(TRAIN_PATH + fold + "/" + f[2] )
	    data_ni_frame_di_gt = nib.load(TRAIN_PATH + fold + "/" + f[3])
	    data_ni_frame_sy = nib.load(TRAIN_PATH + fold + "/" + f[4] )
	    data_ni_frame_sy_gt = nib.load(TRAIN_PATH + fold + "/" + f[5])
	    data_4d = nib.load(TRAIN_PATH + fold + "/" + f[1])

	    frame1 = data_ni_frame_di.get_data()
	    frame1_gt = data_ni_frame_di_gt.get_data()
	    frame2 = data_ni_frame_sy.get_data()
	    frame2_gt = data_ni_frame_sy_gt.get_data()

	    data = data_4d.get_data()
	    size_4d = data_4d.shape
	    total_t = size_4d[3]
	    size_im = frame1.shape

	    x_co = int(size_im[0]/2)
	    y_co = int(size_im[1]/2)
	    slice_no = size_im[2]

	    frame1= frame1[x_co-dim:x_co+dim, y_co-dim:y_co+dim]
	    frame1 = (frame1 * (255/np.max(frame1)))
	    frame1_gt = ((frame1_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] >2))  * 3*np.ones((2*dim,2*dim,slice_no),dtype=int)
	    frame2 = frame2[x_co-dim:x_co+dim, y_co-dim:y_co+dim]
	    frame2 = ((frame2 * (255/np.max(frame2))))
	    frame2_gt =  ((frame2_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] >2)) * 3*np.ones((2*dim,2*dim,slice_no),dtype=int)
	        


	    if n==0:
	    	X_train = frame1
	    	X_train = np.concatenate ( (X_train, frame2), axis=2)
	    else:
	    	X_train = np.concatenate ( (X_train, frame1), axis=2)
	    	X_train = np.concatenate ( (X_train, frame2), axis=2)

	    if n==0:
	    	Y_train = frame1_gt
	    	Y_train = np.concatenate ( (Y_train, frame2_gt), axis=2)

	    else:
	    	Y_train = np.concatenate ( (Y_train, frame1_gt), axis=2)
	    	Y_train = np.concatenate ( (Y_train, frame2_gt), axis=2)
	    print ('Processing Patient ' + str(n+1))
	print('Done!')

	#save the data for another use
	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	X_train = np.transpose(X_train, (2, 1, 0))
	Y_train = np.transpose(Y_train, (2, 1, 0))
	X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], IMG_CHANNELS)
	Y_train = Y_train.reshape(-1, Y_train.shape[1], Y_train.shape[2], IMG_CHANNELS)
	np.save('train_data.npy', X_train)
	np.save('train_mask.npy', Y_train)

def create_testdata():
	test_ids = next(os.walk(TEST_PATH))[1] #folder Patient001

	X_test=[]
	Y_test=[]
	print('Getting and resizing test images and masks ... ')
	sys.stdout.flush()
	
	for n in range(90,100):
	    if (n >98):
	        fold = "patient" + str(n+1)
	    else:
	        fold = "patient0" + str(n+1)
	    f =os.listdir( TEST_PATH + fold)
	    #print f
	    data_ni_frame_di = nib.load(TEST_PATH + fold + "/" + f[2] )
	    data_ni_frame_di_gt = nib.load(TEST_PATH + fold + "/" + f[3])
	    data_ni_frame_sy = nib.load(TEST_PATH + fold + "/" + f[4] )
	    data_ni_frame_sy_gt = nib.load(TEST_PATH + fold + "/" + f[5])
	    data_4d = nib.load(TEST_PATH + fold + "/" + f[1])

	    frame1 = data_ni_frame_di.get_data()
	    frame1_gt = data_ni_frame_di_gt.get_data()
	    frame2 = data_ni_frame_sy.get_data()
	    frame2_gt = data_ni_frame_sy_gt.get_data()

	    data = data_4d.get_data()
	    size_4d = data_4d.shape
	    total_t = size_4d[3]
	    size_im = frame1.shape

	    x_co = int (size_im[0]/2)
	    y_co = int (size_im[1]/2)
	    slice_no = size_im[2]

	    frame1= frame1[x_co-dim:x_co+dim, y_co-dim:y_co+dim]
	    frame1 = (frame1 * (255/np.max(frame1)))
	    frame1_gt = ((frame1_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] >1))  * frame1_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] 
	    frame2 = frame2[x_co-dim:x_co+dim, y_co-dim:y_co+dim]
	    frame2 = (frame2 * (255/np.max(frame2)))
	    frame2_gt =  ((frame2_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] >1)) * frame2_gt[x_co - dim:x_co + dim, y_co - dim:y_co + dim] 
	        


	    if n==90:
	    	X_test = frame2
	    	#X_test = np.concatenate ( (X_test, frame2), axis=2)
	    else:
	    	X_test = np.concatenate ( (X_test, frame2), axis=2)
	    	#X_test = np.concatenate ( (X_test, frame2), axis=2)

	    if n==90:
	    	Y_test = frame2_gt
	    	#Y_test = np.concatenate ( (Y_test, frame2_gt), axis=2)

	    else:
	    	Y_test = np.concatenate ( (Y_test, frame2_gt), axis=2)
	    	#Y_test = np.concatenate ( (Y_test, frame2_gt), axis=2)
	    print ('Processing Patient ' + str(n+1))

	print('Done!')

	#save data fon another use
	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)
	X_test = np.transpose(X_test, (2, 1, 0))
	Y_test = np.transpose(Y_test, (2, 1, 0))
	X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], IMG_CHANNELS)
	Y_test = Y_test.reshape(-1, Y_test.shape[1], Y_test.shape[2], IMG_CHANNELS)
	np.save('test_data_s.npy', X_test)
	np.save('test_mask_s.npy', Y_test)


def display(X_train, Y_train):
	plt.subplot(1,2,1)
	plt.imshow(X_train[:,:,0], cmap = plt.cm.gray)
	plt.subplot(1,2,2)
	plt.imshow(Y_train[:,:,0], cmap = plt.cm.gray)



#create_traindata()
create_testdata()

# img_train = np.load('train_data.npy')
# mask_train = np.load('train_mask.npy')
# size_aug = 2
# IMG_CHANNELS = 1


# print ("Before Size")
# print(img_train.shape)
# print ("Before Size")
# print(mask_train.shape)

# data_gen_args = dict(
# 					rotation_range=60,
# 					fill_mode='nearest')
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_datagen.fit(img_train, augment=True, seed=seed)
# mask_datagen.fit(mask_train, augment=True, seed=seed)

# img_train_n = img_train[:,:,:,0]
# mask_train_n = mask_train[:,:,:,0]

# print (img_train_n.shape)

# img_train_mod = np.transpose(img_train_n, (2,1,0))
# mask_train_mod = np.transpose(mask_train_n, (2,1,0))

# print (img_train_mod.shape)
# print (mask_train_mod.shape)


# k=0;
# for X_batch in image_datagen.flow(img_train, batch_size=size_aug, seed=seed):
# 	for i in range(0, size_aug):
# 		temp = X_batch[i].reshape(128,128,1)
# 		img_train_mod = np.concatenate ( (img_train_mod,temp), axis=2)
		
# 	print(img_train_mod.shape)
# 	k = k+1
# 	if (k>3000):
# 		break


	
# k=0
# for Y_mask in mask_datagen.flow(mask_train, batch_size=size_aug,seed=seed):
# 	for i in range(0, size_aug):
# 		temp = (Y_mask[i].reshape(128,128,1));
# 		mask_train_mod = np.concatenate ( (mask_train_mod, temp), axis=2)	

# 	print(mask_train_mod.shape)
# 	k = k+1
# 	if (k>3000):
# 		break

# X_train = np.asarray(img_train_mod)
# Y_train = np.asarray(mask_train_mod)
# X_train = np.transpose(X_train, (2, 1, 0))
# Y_train = np.transpose(Y_train, (2, 1, 0))
# X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], IMG_CHANNELS)
# Y_train = Y_train.reshape(-1, Y_train.shape[1], Y_train.shape[2], IMG_CHANNELS)

# np.save('train_data2.npy', X_train)
# np.save('train_mask2.npy', Y_train)



# img_train = np.load('train_data1.npy')
# mask_train = np.load('train_mask1.npy')

# print (img_train.shape)
# print(mask_train.shape)

# for i in range(1700,1710):
# 	plt.figure(i)
# 	display(img_train[i], mask_train[i])

# for i in range(1,10):
# 	plt.figure(i)
# 	display(img_train[i], mask_train[i])
# plt.show()

# create_traindata()
# create_testdata()

#img_train = np.load('train_data.npy')
#mask_train = np.load('train_mask.npy')



#for i in range(1700,1710):
#	plt.figure(i)
#	display(img_train[i], mask_train[i])

#for i in range(1,10):
#	plt.figure(i)
#	display(img_train[i], mask_train[i])
#plt.show()


#Reading and displaying from .npy
#img = np.load('train_data.npy')
#mask = np.load('train_mask.npy')
#test = np.load('test_data.npy')

