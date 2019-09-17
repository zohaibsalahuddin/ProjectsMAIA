import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

import warnings
warnings.filterwarnings("ignore")


from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
import os
# import sys
import random
# import warnings
from PIL import Image
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


from keras.models import  load_model, model_from_json
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
from medpy.metric.binary import hd, dc

import tensorflow as tf
	
import nibabel as nib 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import warnings
warnings.filterwarnings("ignore")

model_type1 = None
model_type2 = None
slice_no = None
datalv = None
total_slices = None

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


def my_model():
	#print ("In My Model")
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
	return model_type1,model_type2


def test():
	global datalv
	global total_slices
	print ("Works")
	fileName, _ = QFileDialog.getOpenFileName(dlg, "Open File",QDir.currentPath())
	#if fileName:
		#print (fileName)
	img = nib.load(fileName)
	datalv = img.get_data()
	#print(datalv.shape)
	total_slices = datalv.shape[2] - 1
	#print (total_slices)
	slice_no = get_slice()
	frame1 = datalv[:,:,slice_no].T
	frame1 = (frame1 * (255/np.max(frame1)))
	Image.fromarray((np.uint8(frame1))).save('test.png')
	image1 = QImage('test.png')
	dlg.inputImg.setPixmap(QPixmap.fromImage(image1))



def get_slice():
	global slice_no
	dlg.sliceNoSpin.setMaximum(total_slices)
	dlg.sliceNoSpin.setMinimum(0)
	slice_no = (dlg.sliceNoSpin.value())
	# print (str_1)
	# print (type (str_1))
	# slice_no = int(str_1)
	#print (slice_no)
	return slice_no
# def predict():

#  	X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)

def predict ():
	global datalv
	#print(datalv.shape)
	total_slices = datalv.shape[2] - 1
	#print (total_slices)
	slice_no = get_slice()
	#print ("Printing the Slice Number")
	#print(slice_no)
	frame1 = datalv[:,:,slice_no].T
	frame1 = (frame1 * (255/np.max(frame1)))
	Image.fromarray((np.uint8(frame1))).save('test.png')
	image1 = QImage('test.png')
	dlg.inputImg.setPixmap(QPixmap.fromImage(image1))
	orig = datalv[:,:,slice_no]
	orig = orig.T
	train = datalv[:,:,slice_no]
	#print(train.shape)
	x_co = int(train.shape[0]/2)
	y_co = int(train.shape[1]/2)	
	dim = 64
	train = train[x_co-dim:x_co+dim, y_co-dim:y_co+dim]
	#print(train.shape[0])
	#print(train.shape[1])
	train = train.reshape(1, train.shape[0], train.shape[1], 1)
	#print (train.shape)
	preds_test_lv = model_type1.predict(train, verbose=1)
	preds_test_my = model_type2.predict(train, verbose=1)

	preds_test_t_lv = (preds_test_lv > 0.5).astype(np.uint8)
	preds_test_t_my = (preds_test_my > 0.5).astype(np.uint8)

	#print(preds_test_lv.shape)
	#print(preds_test_my.shape)
	shape_n = preds_test_t_lv.shape
	#print(shape_n)
	#print(shape_n[0])
	#print(shape_n[1])
	#print(shape_n[2])
			

	processed_predicted_lv = post_process_lv(preds_test_t_lv)
	processed_predicted_my = post_process_my(preds_test_t_my)

	ones_n = np.ones ((shape_n[0],shape_n[2],shape_n[1],1),dtype=int)
	#print (ones_n.shape)
	#print (processed_predicted_lv.shape)



	lv_inv = (processed_predicted_lv < 1) * ones_n
	my_inv = (processed_predicted_my < 1) * ones_n


	#print (lv_inv.shape)
	processed_predicted_my_mod = processed_predicted_my * lv_inv *2
	processed_predicted_lv_mod  = processed_predicted_lv * processed_predicted_my *3
	final_img = processed_predicted_my_mod +  processed_predicted_lv_mod

	#print (final_img.shape)
	#print (processed_predicted_lv.shape)
	#print (processed_predicted_my.shape)

	final = final_img[0,:,:,0].T
	final = (final * (255/np.max(final)))
	Image.fromarray((np.uint8(final))).save('test.png')
	image1 = QImage('test.png')
	dlg.outputImg.setPixmap(QPixmap.fromImage(image1))

	lv = processed_predicted_my_mod[0,:,:,0].T
	lv = (lv * (255/np.max(lv)))
	Image.fromarray((np.uint8(lv))).save('test.png')
	image1 = QImage('test.png')
	dlg.myo_lvImg.setPixmap(QPixmap.fromImage(image1))

	my = processed_predicted_lv_mod[0,:,:,0].T
	my = (my * (255/np.max(my)))
	Image.fromarray((np.uint8(my))).save('test.png')
	image1 = QImage('test.png')
	dlg.leftventricleImg.setPixmap(QPixmap.fromImage(image1))


	ones_n1 = np.ones ((128,128),dtype=int)
	orig = orig.reshape(orig.shape[0], orig.shape[1])
	x_co = int(orig.shape[0]/2)
	y_co = int(orig.shape[1]/2)	
	dim = 64
	lv_inv_n = (processed_predicted_lv[0,:,:,0].T < 1) * ones_n1
	my_inv_n = (processed_predicted_my[0,:,:,0].T < 1) * ones_n1
	orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim]	 = orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim] * lv_inv_n 
	orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim]	 = orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim] * my_inv_n
	orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim]	 = orig[x_co-dim:x_co+dim, y_co-dim:y_co+dim] + (processed_predicted_lv_mod[0,:,:,0].T *128)+  (processed_predicted_my_mod[0,:,:,0].T *255)

	my = orig
	my = (my * (255/np.max(my)))
	Image.fromarray((np.uint8(my))).save('test.png')
	image1 = QImage('test.png')
	dlg.overlay.setPixmap(QPixmap.fromImage(image1))





app = QtWidgets.QApplication([])
dlg = uic.loadUi("test1.ui")
model_type1,model_type2 = my_model()
dlg.loadBtn.clicked.connect(test)
dlg.predictBtn.clicked.connect(predict)
dlg.show()
app.exec()


