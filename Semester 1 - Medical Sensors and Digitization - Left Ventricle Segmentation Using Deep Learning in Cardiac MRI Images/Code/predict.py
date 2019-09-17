
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, model_from_json
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import concatenate
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.util import img_as_ubyte
# from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from medpy.metric.binary import hd, dc

# for data augmentation
import glob
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
	
import nibabel as nb 
from scipy import ndimage as ndi
from scipy import ndimage
from skimage.filters import roberts, sobel, scharr, prewitt

from skimage import feature
import cv2
# seed = 42
# random.seed = seed
# np.random.seed = seed

def post_process_my(Test_predicted):
	struc_element = disk(4)	
	total = Test_predicted.shape[0]
	post_predicted = np.zeros((total,128,128,1), dtype=np.uint8)
	for j in range(total):
		myimage =  Test_predicted[j] #to uint8
		opened = closing(myimage[:,:,0], struc_element)	
		openedd = np.expand_dims(opened, axis=-1)					
		post_predicted[j,:,:,:] = openedd	
	return post_predicted		

def post_process_lv(Test_predicted):
	struc_element = disk(4)	
	total = Test_predicted.shape[0]
	post_predicted = np.zeros((total,128,128,1), dtype=np.uint8)
	for j in range(total):
		myimage =  Test_predicted[j] #to uint8
		opened = opening(myimage[:,:,0], struc_element)	
		openedd = np.expand_dims(opened, axis=-1)					
		post_predicted[j,:,:,:] = openedd	
	return post_predicted	

def display(Test_gt, Test_predicted, Post_predict,Final):
	plt.subplot(2,2,1)
	plt.imshow(Test_gt[:,:,0], cmap = plt.cm.gray)
	plt.subplot(2,2,2)
	plt.imshow(Test_predicted[:,:,0], cmap = plt.cm.gray)
	plt.subplot(2,2,3)
	plt.imshow(Post_predict[:,:,0], cmap = plt.cm.gray)
	plt.subplot(2,2,4)
	plt.imshow(Final[:,:,0], cmap = plt.cm.gray)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    smooth = 1.0
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


img_train = np.load('train_data.npy')
mask_train = np.load('train_mask.npy')

img_test = np.load('test_data_s.npy')
mask_test = np.load('test_mask_s.npy')

print ("Shape of Image Test")
print(img_test.shape)

model = load_model ('model-lv_segmentation2.h5')
model.save_weights('t_lv.h5')
with open('model_architecture.json','w') as f:
	f.write(model.to_json())


with open('model_architecture.json','r') as f:
	model_type1 = model_from_json(f.read())

with open('model_architecture.json','r') as f:
	model_type2 = model_from_json(f.read())

model_type1.load_weights('t_lv.h5')
model_type2.load_weights('t_my.h5')

model_type1.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model_type2.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

preds_test_lv = model_type1.predict(img_test, verbose=1)
preds_test_my = model_type2.predict(img_test, verbose=1)
# preds_train = model.predict(img_train[:int(img_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(img_train[int(img_train.shape[0]*0.9):], verbose=1)
#preds_test = model.predict(img_test, verbose=1)

# Threshold predictions if it is > 0.5 true, ow false
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)




# Threshold predictions if it is > 0.5 true, ow false
preds_test_t_lv = (preds_test_lv > 0.5).astype(np.uint8)
preds_test_t_my = (preds_test_my > 0.5).astype(np.uint8)

shape_n = preds_test_t_lv.shape
print(shape_n)
print(shape_n[0])
print(shape_n[1])
print(shape_n[2])
		

processed_predicted_lv = post_process_lv(preds_test_t_lv)
processed_predicted_my = post_process_my(preds_test_t_my)

ones_n = np.ones ((shape_n[0],shape_n[2],shape_n[1],1),dtype=int)
print (ones_n.shape)
print (processed_predicted_lv.shape)



lv_inv = (processed_predicted_lv < 1) * ones_n
my_inv = (processed_predicted_my < 1) * ones_n


print (lv_inv.shape)
processed_predicted_my_mod = processed_predicted_my * lv_inv *2
processed_predicted_lv_mod  = processed_predicted_lv * processed_predicted_my *3
final_img = processed_predicted_my_mod +  processed_predicted_lv_mod


for i in range(14,23):
	processed_predicted_my[i,: , :, 0] = cv2.Canny(processed_predicted_my[i,: , :, 0],0,1)
	processed_predicted_lv[i,: , :, 0] = cv2.Canny(processed_predicted_lv[i,: , :, 0],0,1)
	ones_my = np.ones ((128,128,3),dtype=int)
	ones_lv = np.ones ((128,128,3),dtype=int)



	ones_lv[:,:,1] = 0
	ones_lv[:,:,2] = 0
	ones_lv[:,:,0] = processed_predicted_lv[i,: , :, 0]


	ones_lv [:,:,1] = img_test[i,:,:,0]
	ones_lv [:,:,2] =  processed_predicted_my[i,: , :, 0]
	plt.figure(i)
	plt.imshow(ones_lv, cmap = plt.cm.gray)
# plt.hold()
# plt.imshow(ones_lv)
# plt.imshow(ones_my)


plt.show()
# def HausdorffDist(A,B):
#     # Hausdorf Distance: Compute the Hausdorff distance between two point
#     # clouds.
#     # Let A and B be subsets of metric space (Z,dZ),
#     # The Hausdorff distance between A and B, denoted by dH(A,B),
#     # is defined by:
#     # dH(A,B) = max(h(A,B),h(B,A)),
#     # where h(A,B) = max(min(d(a,b))
#     # and d(a,b) is a L2 norm
#     # dist_H = hausdorff(A,B)
#     # A: First point sets (MxN, with M observations in N dimension)
#     # B: Second point sets (MxN, with M observations in N dimension)
#     # ** A and B may have different number of rows, but must have the same
#     # number of columns.
#     #
#     # Edward DongBo Cui; Stanford University; 06/17/2014

#     # Find pairwise distance
#     D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
#     # Find DH
#     dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
#     return(dH)




# sum_dc=0
# count = 0
# print ("starting")
# print (mask_test.shape)
# for i in range(0,114):
# 	# plt.figure(i)
# 	print ("Size" + str(mask_test[i].shape))
# 	mask_test[i,:,:,0]  = (mask_test[i,:,:,0] > 2) * mask_test[i,:,:,0]
# 	# mask_test[i,:,:,0] = mask_test[i,:,:,0] + 1
# 	# processed_predicted_lv[i,:,:,0] = processed_predicted_lv[i,:,:,0] + 1
# 	print (mask_test.shape)
# 	# a = dice_coef(mask_test[i,:,:,0].astype(float), processed_predicted_lv[i,:,:,0].astype(float))
# 	# print (a)
# 	c = dc(mask_test[i,:,:,0].astype(float), processed_predicted_lv_mod[i,:,:,0].astype(float))
# 	print (c)


# 	if (c > 0.01):
# 		sum_dc = sum_dc + c
# 		count = count + 1
# 	# plt.subplot(1,2,1)
# 	# plt.imshow(mask_test[i,:,:,0], cmap = plt.cm.gray)
# 	# plt.subplot(1,2,2)
# 	# plt.imshow(processed_predicted_lv[i,:,:,0], cmap = plt.cm.gray)
# 	# display(mask_test[i],processed_predicted_lv[i], processed_predicted_my[i] ,final_img[i])
# dice_avg = sum_dc/count
# print(dice_avg)
# plt.show()


# avg_dice = sum_dc / count
# print ("The Average Dice Scored Achieved is: ")
# print (avg_dice)
# print ("Count")
# print (count)


# sum_dc=0
# count = 0
# for i in range(0,114):
# 	# plt.figure(i)
# 	print ("Size" + str(mask_test[i].shape))
# 	mask_test[i,:,:,0]  = (mask_test[i,:,:,0] > 1) * (mask_test[i,:,:,0] < 3) * mask_test[i,:,:,0]
# 	# mask_test[i,:,:,0] = mask_test[i,:,:,0] + 1
# 	# processed_predicted_lv[i,:,:,0] = processed_predicted_lv[i,:,:,0] + 1
# 	print (mask_test.shape)
# 	# a = dice_coef(mask_test[i,:,:,0].astype(float), processed_predicted_lv[i,:,:,0].astype(float))
# 	# print (a)
# 	c = dc(mask_test[i,:,:,0].astype(float), processed_predicted_my_mod[i,:,:,0].astype(float))
# 	print (c)


# 	if (c > -0.01):
# 		sum_dc = sum_dc + c
# 		count = count + 1
# 	# plt.subplot(1,2,1)
# 	# plt.imshow(mask_test[i,:,:,0], cmap = plt.cm.gray)
# 	# plt.subplot(1,2,2)
# 	# plt.imshow(processed_predicted_my_mod[i,:,:,0], cmap = plt.cm.gray)
# 	#display(mask_test[i],processed_predicted_lv[i], processed_predicted_my_mod[i] ,final_img[i])



# dice_avg = sum_dc/count
# print(dice_avg)
# print ("Myocardium")
# plt.show()


# avg_dice = sum_dc / count
# print ("The Average Dice Scored Achieved is: ")
# print (avg_dice)



#######################################################################################################################
# Data Augmentation
# def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
#                     mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
#                     flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (128,128),seed = 1):
#     '''
#     can generate image and mask at the same time
#     use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
#     if you want to visualize the results of generator, set save_to_dir = "your path"
#     '''
#     image_datagen = ImageDataGenerator(**aug_dict)
#     mask_datagen = ImageDataGenerator(**aug_dict)
#     image_generator = image_datagen.flow_from_directory(
#         train_path,
  		  # classes = [image_folder],
#         class_mode = None,
#         color_mode = image_color_mode,
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = save_to_dir,
#         save_prefix  = image_save_prefix,
#         seed = seed)
#     mask_generator = mask_datagen.flow_from_directory(
#         train_path,
#         class_mode = None,
#         color_mode = mask_color_mode,
#         target_size = target_size,
#         batch_size = batch_size,
#         save_to_dir = save_to_dir,
#         save_prefix  = mask_save_prefix,
#         seed = seed)
#     train_generator = zip(image_generator, mask_generator)    

# def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
#     image_name_arr = glob.glob(os.path.join(image_path,"%s*.nii.gz"%image_prefix))
#     image_arr = []
#     mask_arr = []
#     for index,item in enumerate(image_name_arr):
#         img = io.imread(item,as_gray = image_as_gray)
#         img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
#         mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
#         mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask

#         image_arr.append(img)
#         mask_arr.append(mask)
#     image_arr = np.array(image_arr)
#     mask_arr = np.array(mask_arr)
#     return image_arr,mask_arr


# # data_gen_args = dict(rotation_range=0.2,
# #                     width_shift_range=0.05,
# #                     height_shift_range=0.05,
# #                     shear_range=0.05,
# #                     zoom_range=0.05,
# #                     horizontal_flip=True,
# #                     fill_mode='nearest')

# # myGenerator = trainGenerator(20,'D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data','image','label',
# # 				data_gen_args,save_to_dir = "D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/aug")


# # image_arr,mask_arr = geneTrainNpy("D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/aug", 
# # 								"D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/aug")
#######################################################################################################
# dlv.create_traindata()
# dlv.create_testdata()

# img_train = np.load('train_data.npy')
# mask_train = np.load('train_mask.npy')
# size_aug = 1
# IMG_CHANNELS = 1


# print ("Before Size")
# print(img_train.shape)
# print ("Before Size")
# print(mask_train.shape)

# data_gen_args = dict(featurewise_center=True,
# 					featurewise_std_normalization=True,
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
# 		img_train_mod = np.concatenate ( (img_train_mod, X_batch[i].reshape(128, 128,1)), axis=2)
		
# 	print(img_train_mod.shape)
# 	k = k+1
# 	print (k)

# 	if (k>1700):
# 		break


	
# k=0
# for Y_mask in mask_datagen.flow(mask_train, batch_size=size_aug,seed=seed):
# 	for i in range(0, size_aug):
# 		mask_train_mod = np.concatenate ( (mask_train_mod, Y_mask[i].reshape(128, 128,1)), axis=2)	

# 	print(mask_train_mod.shape)
# 	k = k+1
# 	if (k>1700):
# 		break

# X_train = np.asarray(img_train_mod)
# Y_train = np.asarray(mask_train_mod)
# X_train = np.transpose(X_train, (2, 1, 0))
# Y_train = np.transpose(Y_train, (2, 1, 0))
# X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], IMG_CHANNELS)
# Y_train = Y_train.reshape(-1, Y_train.shape[1], Y_train.shape[2], IMG_CHANNELS)

# np.save('train_data1.npy', X_train)
# np.save('train_mask1.npy', Y_train)



# img_train = np.load('train_data1.npy')
# mask_train = np.load('train_mask1.npy')

# print (img_train.shape)
# print(mask_train.shape)

# def display(X_train, Y_train):
# 	plt.subplot(1,2,1)
# 	plt.imshow(X_train[:,:,0], cmap = plt.cm.gray)
# 	plt.subplot(1,2,2)
# 	plt.imshow(Y_train[:,:,0], cmap = plt.cm.gray)

# for i in range(1700,1710):
# 	plt.figure(i)
# 	display(img_train[i], mask_train[i])

# for i in range(1,10):
# 	plt.figure(i)
# 	display(img_train[i], mask_train[i])
# plt.show()

	######################################################################################################
# image_generator = image_datagen.flow_from_directory(
#     "D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/image",    
#     class_mode=None,
#     seed=seed)
# print(np.asarray(image_generator).shape())

# mask_generator = mask_datagen.flow_from_directory(
#     "D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/label",    
#     class_mode=None,
#     seed=seed)
# def plotImages( images_arr, n_images=2):
#     fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         if img.ndim != 2:
#             img = img.reshape( (128,128))
#         ax.imshow( img, cmap="Greys_r")
#         ax.set_xticks(())
#         ax.set_yticks(())
#     plt.tight_layout()

# image_datagen.fit(img_train, augment=True)
# mask_datagen.fit(mask_train, augment=True)
# augmented_images, _ = next( image_datagen.flow(img_train, mask_train, batch_size=4*4))
# plotImages( augmented_images)
#GEt 10 samples of the augmented data
# aug_image = [next(image_generator)[0].astype(np.uint8) for i in range(10)]
# aug_mask = [next(mask_generator)[0].astype(np.uint8) for i in range(10)]

# print(aug_mask.shape())
# print(aug_image.shape())
# np.save("D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/image_arr.npy",image_generator)
# np.save("D:/MAIA/MAIA-1st Semister UB/Medical Sensors/Project/data/mask_arr.npy",mask_generator)

#combine them to older images in order



# combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)


