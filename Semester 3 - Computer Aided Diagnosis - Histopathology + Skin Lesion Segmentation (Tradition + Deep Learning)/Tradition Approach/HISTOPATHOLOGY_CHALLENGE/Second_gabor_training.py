#################################################################
# EXTRACTING GABOR FEATURES AND SAVING - Then Combining features saved
# in color features. Run color_features.py first, the features will
# be saved. Then run this file :)
################################################################


from scipy.stats import itemfreq
from skimage.feature import greycomatrix, greycoprops
import mahotas
from sklearn.svm import LinearSVC
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy.ma as ma
from skimage.draw import ellipse
from skimage.feature import local_binary_pattern
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import entropy

METHOD = 'uniform'

def glcm_f(segment_region):
    glcm = greycomatrix(segment_region, [5], [0], 256)

    stats = ["dissimilarity", "correlation", "contrast", "homogeneity", "ASM", "energy"]
    dissimilarity = greycoprops(glcm, stats[0])[0, 0]
    correlation = greycoprops(glcm, stats[1])[0, 0]
    contrast = greycoprops(glcm, stats[2])[0, 0]
    homogeneity = greycoprops(glcm, stats[3])[0, 0]
    ASM = greycoprops(glcm, stats[4])[0, 0]
    energy = greycoprops(glcm, stats[5])[0, 0]
    entropy_f = entropy(segment_region)
    temp_features = [dissimilarity, correlation,contrast,homogeneity,ASM,energy,entropy_f]
    return temp_features

def fd_haralick(image):
    # convert the image to grayscale
    gray1 = image[:,:,0]
    gray2 = image[:,:,1]
    gray3 = image[:,:,2]

    gray1 = gray1.astype(int)
    gray2 = gray2.astype(int)
    gray3 = gray3.astype(int)

    # compute the haralick texture feature vector

    haralick_mean = mahotas.features.haralick(gray1).mean(axis=0)
    haralick_std = mahotas.features.haralick(gray2).mean(axis=0)
    haralick_range = mahotas.features.haralick(gray3).mean(axis=0)
    # return the result
    test = np.concatenate((haralick_mean,haralick_std,haralick_range),axis=0)
    return test


def build_filters(test_lambda):
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 5
    for theta in np.arange(0, np.pi, np.pi / 8):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':test_lambda,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def gabor_features(img_orig):
    filters = build_filters(1*np.pi / 4)
    p2 = process(img_orig, filters)
    filters = build_filters(2*np.pi / 4)
    p3 = process(img_orig, filters)
    filters = build_filters(3*np.pi / 4)
    p4 = process(img_orig, filters)
    filters = build_filters(4*np.pi / 4)
    p5 = process(img_orig, filters)
    filters = build_filters(5*np.pi / 4)
    p6 = process(img_orig, filters)
    filters = build_filters(6*np.pi / 4)
    p7 = process(img_orig, filters)

    f1 = fd_haralick(p2)
    f2 = fd_haralick(p3)
    f3 = fd_haralick(p4)
    f4 = fd_haralick(p5)
    f5 = fd_haralick(p6)
    f6 = fd_haralick(p7)


    test = np.concatenate((f1,f2,f3,f4,f5,f6),axis=0)

    return test


def extract_features(train_class):
    for img_no in range(len(train_class)):
        img_orig_grayscale = cv2.imread(train_class[img_no], cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread(train_class[img_no])
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        # img_orig = staintools.LuminosityStandardizer.standardize(img_orig)

        l, a, b = cv2.split(img_orig)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        a = clahe.apply(a)
        b = clahe.apply(b)
        img_orig = cv2.merge((l, a, b))

        gabor_f = gabor_features(img_orig)

        feature_v = np.hstack((gabor_f))

        feature_v = np.expand_dims(feature_v, 0)
        if img_no == 0:
            feature_vector = feature_v
        else:
            feature_vector = np.concatenate((feature_vector, feature_v), axis=0)
        print(img_no)

        if (img_no % 100 == 0):
            print(img_no)
    return feature_vector


def features_normalize(feature_vectors, class_vectors):
    mean_x = np.mean(feature_vectors, axis=0)
    std_x = np.std(feature_vectors, axis=0)
    where_are_zeros = (std_x == 0)
    std_x[where_are_zeros] = 1
    randomize = np.arange(len(class_vectors))
    np.random.shuffle(randomize)
    feature_vectors_n = feature_vectors[randomize, :]
    feature_vectors_n = (feature_vectors_n - np.mean(feature_vectors_n, axis=0)) / np.std(feature_vectors_n, axis=0)
    class_vectors_n = class_vectors[randomize]
    return feature_vectors_n, class_vectors_n, mean_x, std_x

filenames_positive = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Histopathology\\Training\\train\\b0\\*.png")
filenames_negative = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Histopathology\\Training\\train\\m0\\*.png")


filenames_positive_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Histopathology\\Validation\\val\\b0\\*.png")
filenames_negative_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Histopathology\\Validation\\val\\m0\\*.png")

print(len(filenames_positive))
print(len(filenames_negative))
print(len(filenames_positive_val))
print(len(filenames_negative_val))

filenames_positive.sort()
filenames_negative.sort()

filenames_positive_val.sort()
filenames_negative_val.sort()

CROSS_VALIDATION = 5
TOTAL_TRAINING = 2400
STEP = TOTAL_TRAINING / CROSS_VALIDATION

# Training with original Images
test_les = filenames_negative_val
test_nv = filenames_positive_val
train_les = filenames_negative
train_nv = filenames_positive

# Loading Testing Images


print("Training Positives")
# extracting features for positive class
positive_train_features = extract_features(train_nv)
positive_train_class = np.ones((len(train_nv), 1))

print("Training Negatives")
# extracting features for negative class
negative_train_features = extract_features(train_les)
negative_train_class = np.zeros((len(train_les), 1))

print("Training Positives Tests")
# extracting features for positive test class
positive_test_features = extract_features(test_nv)
positive_test_class = np.ones((len(test_nv), 1))


print("Training Negative Tests")
# extracting features for negative test class
negative_test_features = extract_features(test_les)
negative_test_class = np.zeros((len(test_les), 1))

# concatenating the two features
train_feature = np.concatenate((positive_train_features, negative_train_features), axis=0)
train_class = np.concatenate((positive_train_class, negative_train_class), axis=0)

np.save("train_feat_gaborrgb3.npy",train_feature)

# normalizing and shuffling the train features
train_feature, train_class, mean_x, std_x = features_normalize(train_feature, train_class)

# concatenating the test features
test_feature = np.concatenate((positive_test_features, negative_test_features), axis=0)
test_class = np.concatenate((positive_test_class, negative_test_class), axis=0)
np.save("test_feat_gaborrgb4.npy",test_feature)
# normalizing and shuffling the test features
test_feature = (test_feature - mean_x) / std_x

where_are_NaNs = np.isnan(train_feature)
train_feature[where_are_NaNs] = 0

where_are_NaNs = np.isnan(test_feature)
test_feature[where_are_NaNs] = 0



from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']

clf = LinearSVC(C = 0.1, random_state=42, tol=1e-5,max_iter=3000)
clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))


from sklearn.ensemble import AdaBoostClassifier
# Training the classifier
parameter_c =2
kernel_svm = 'rbf'
clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')

clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))



# Training the classifier
parameter_c =10
kernel_svm = 'rbf'
clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')

clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy:" + str(score))
from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1']
print(classification_report(np.asarray(test_class), predict, target_names=target_names))


###################################################################################
 # COMBINING GABOR FEATURES AND OTHER FEATURES - GIVES 83.16 percent Accuracy
###################################################################################
train = np.load("train_l.npy")
test = np.load("test_l.npy")

train_feature = np.load("train_feat_gaborrgb3.npy")
test_feature = np.load("test_feat_gaborrgb4.npy")
train = train[:,:]
test = test[:,:]

# concatenating the two features

positive_train_class = np.zeros((len(train_nv), 1))
positive_test_class = np.zeros((len(test_nv), 1))
negative_train_class = np.ones((len(train_les), 1))
negative_test_class = np.ones((len(test_les), 1))

train_class = np.concatenate((positive_train_class,negative_train_class),axis=0)
test_class = np.concatenate((positive_test_class,negative_test_class),axis=0)

train_feature_n = train_feature[:,:]
test_feature_n = test_feature[:,:]
training = np.concatenate((train_feature_n,train),axis=1)
testing = np.concatenate((test_feature_n,test),axis=1)


training,train_class,mean_x,std_x = features_normalize(training,train_class)

# normalizing and shuffling the test features
testing = (testing - mean_x) / std_x

where_are_NaNs = np.isnan(training)
training[where_are_NaNs] = 0

where_are_NaNs = np.isnan(testing)
testing[where_are_NaNs] = 0

#
#from sklearn.metrics import classification_report
#target_names = ['class 0', 'class 1']
## Training the classifier
#from sklearn.ensemble import RandomForestClassifier
#
#parameter_c = 15
#kernel_svm = 'rbf'
##clf = svm.SVC(C=parameter_c, kernel=kernel_svm, class_weight='balanced')
#clf = LinearSVC(C = 2, random_state=42, tol=1e-5,max_iter=3000)
#
#clf.fit(training, train_class)
#
#predict = clf.predict(training)
#score = accuracy_score(np.asarray(train_class), predict)
#print("Accuracy Training:" + str(score))
#
## calculating the accuracy
#predict = clf.predict(testing)
#score = accuracy_score(np.asarray(test_class), predict)
#print("Accuracy Testing:" + str(score))
#print(classification_report(np.asarray(test_class), predict, target_names=target_names))
#
#

parameter_c = 4
kernel_svm = 'rbf'
clf = svm.SVC(C=parameter_c, kernel=kernel_svm, class_weight='balanced')
#clf = LinearSVC(C = 10, random_state=42, tol=1e-5,max_iter=3000)

clf.fit(training, train_class)

predict = clf.predict(training)
score = accuracy_score(np.asarray(train_class), predict)
print("Accuracy Training:" + str(score))

# calculating the accuracy
predict = clf.predict(testing)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy Testing:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))

################################################################
# Training for testing images
###############################################################


###################################################################################
 # COMBINING GABOR FEATURES AND OTHER FEATURES - GIVES 83.16 percent Accuracy
###################################################################################
train = np.load("train_l.npy")
test = np.load("test_l.npy")

train_feature = np.load("train_feat_gaborrgb3.npy")
test_feature = np.load("test_feat_gaborrgb4.npy")



# concatenating the two features

positive_train_class = np.zeros((len(train_nv), 1))
positive_test_class = np.zeros((len(test_nv), 1))
negative_train_class = np.ones((len(train_les), 1))
negative_test_class = np.ones((len(test_les), 1))

train_class = np.concatenate((positive_train_class,negative_train_class),axis=0)
test_class = np.concatenate((positive_test_class,negative_test_class),axis=0)

train_feature_n = train_feature[:,:]
test_feature_n = test_feature[:,:]
training = np.concatenate((train_feature_n,train),axis=1)
testing = np.concatenate((test_feature_n,test),axis=1)

training = np.concatenate((training,testing),axis=0)
train_class = np.concatenate((train_class,test_class),axis=0)

training,train_class,mean_x,std_x = features_normalize(training,train_class)

test = np.load("test_feature_rest.npy")
test_feature_n = np.load("test_feature_gabor.npy")
testing = np.concatenate((test_feature_n,test),axis=1)


# normalizing and shuffling the test features
testing = (testing - mean_x) / std_x

where_are_NaNs = np.isnan(training)
training[where_are_NaNs] = 0

where_are_NaNs = np.isnan(testing)
testing[where_are_NaNs] = 0

#
#from sklearn.metrics import classification_report
#target_names = ['class 0', 'class 1']
## Training the classifier
#from sklearn.ensemble import RandomForestClassifier
#
#parameter_c = 15
#kernel_svm = 'rbf'
##clf = svm.SVC(C=parameter_c, kernel=kernel_svm, class_weight='balanced')
#clf = LinearSVC(C = 1, random_state=42, tol=1e-5,max_iter=3000)
#
#clf.fit(training, train_class)
#
#predict = clf.predict(training)
#score = accuracy_score(np.asarray(train_class), predict)
#print("Accuracy Training:" + str(score))
#
## calculating the accuracy
#predict = clf.predict(testing)
#score = accuracy_score(np.asarray(test_class), predict)
#print("Accuracy Testing:" + str(score))
#print(classification_report(np.asarray(test_class), predict, target_names=target_names))
#
#

parameter_c = 4
kernel_svm = 'rbf'
#clf = svm.SVC(C=parameter_c, kernel=kernel_svm, class_weight='balanced')
clf = LinearSVC(C = 1, random_state=42, tol=1e-5,max_iter=3000)

clf.fit(training, train_class)

predict = clf.predict(training)
score = accuracy_score(np.asarray(train_class), predict)
print("Accuracy Training:" + str(score))

# calculating the accuracy
predict = clf.predict(testing)

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (predict)

## save to xlsx file

filepath = 'histo_results.xlsx'

df.to_excel(filepath, index=False)