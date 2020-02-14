##############################################################
# Extracting and Saving Color Features :)
#############################################################

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

METHOD = 'uniform'



def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(int)
    # compute the haralick texture feature vector

    haralick_mean = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick_mean


def mean_std_whole(img_orig):

    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = np.std(img_orig_n[:, :, 0] )
    std_g_1 = np.std(img_orig_n[:, :, 1] )
    std_b_1 = np.std(img_orig_n[:, :, 2] )

    return [std_r_1, std_g_1, std_b_1]



def skew_whole(img_orig):

    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = skew(img_orig_n[:, :, 0],axis=None )
    std_g_1 = skew(img_orig_n[:, :, 1],axis=None )
    std_b_1 = skew(img_orig_n[:, :, 2],axis=None )

    return [std_r_1,std_g_1,std_b_1]

def kurtosis_whole(img_orig):

    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = kurtosis(img_orig_n[:, :, 0] ,axis=None)
    std_g_1 = kurtosis(img_orig_n[:, :, 1] ,axis=None)
    std_b_1 = kurtosis(img_orig_n[:, :, 2] ,axis=None)


    return [std_r_1,std_g_1,std_b_1]



def glcm_features_whole(segment_region):
    glcm = greycomatrix(segment_region, [5], [0], 256)

    stats = ["dissimilarity", "correlation", "contrast", "homogeneity", "ASM", "energy"]
    dissimilarity = greycoprops(glcm, stats[0])[0, 0]
    correlation = greycoprops(glcm, stats[1])[0, 0]
    contrast = greycoprops(glcm, stats[2])[0, 0]
    homogeneity = greycoprops(glcm, stats[3])[0, 0]
    ASM = greycoprops(glcm, stats[4])[0, 0]
    energy = greycoprops(glcm, stats[5])[0, 0]

    temp_features = [dissimilarity, correlation,contrast,homogeneity,ASM,energy]
    return temp_features

def fd_histogram(image):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,0] = image[:,:,0]
    image[:,:,1] = image[:,:,1]
    image[:,:,2] = image[:,:,2]
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def lbp_gray_images(img_orig_grayscale):
    lbp = local_binary_pattern(img_orig_grayscale, 24, 3, METHOD)
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    if(len(hist) <26):
        hist = np.zeros((26,))
    return  hist

def extract_features(train_class):
    for img_no in range(len(train_class)):
        img_orig_grayscale = cv2.imread(train_class[img_no],cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread(train_class[img_no])
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        l, a, b = cv2.split(img_orig)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        a = clahe.apply(a)
        b = clahe.apply(b)
        img_orig = cv2.merge((l, a, b))


        std_f_w = mean_std_whole(img_orig)
        skew_f_w =skew_whole(img_orig)
        kurt_f_w = kurtosis_whole(img_orig)
        fd_histogram_l = fd_histogram(img_orig)
        fd_haralick_g = fd_haralick(img_orig)
        lbp_g = lbp_gray_images(img_orig_grayscale)

        feature_v = np.hstack((skew_f_w,kurt_f_w,std_f_w,fd_haralick_g,lbp_g,fd_histogram_l))

        feature_v = np.expand_dims(feature_v,0)
        if img_no == 0:
            feature_vector = feature_v
        else:
            feature_vector = np.concatenate((feature_vector,feature_v),axis=0)

        if (img_no % 100 == 0):
            print(img_no)
    return feature_vector


def features_normalize(feature_vectors,class_vectors):
    mean_x = np.mean(feature_vectors, axis=0)
    std_x = np.std(feature_vectors, axis=0)
    #print(std_x)
    where_are_zeros = (std_x == 0)
    std_x[where_are_zeros] = 1
    #print(std_x)
    randomize = np.arange(len(class_vectors))
    np.random.shuffle(randomize)
    feature_vectors_n = feature_vectors[randomize, :]
    feature_vectors_n = (feature_vectors_n - np.mean(feature_vectors_n, axis=0)) / np.std(feature_vectors_n, axis=0)
    class_vectors_n = class_vectors[randomize]
    return feature_vectors_n,class_vectors_n,mean_x,std_x


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
STEP = TOTAL_TRAINING/CROSS_VALIDATION




# Training with original Images
test_les = filenames_negative_val
test_nv = filenames_positive_val
train_les = filenames_negative
train_nv = filenames_positive

#Loading Testing Images


print("Training Positives")
# extracting features for positive class
positive_train_features = extract_features(train_nv)
positive_train_class = np.ones((len(train_nv),1))

print("Training Negatives")
# extracting features for negative class
negative_train_features=extract_features(train_les)
negative_train_class = np.zeros((len(train_les),1))

print("Training Positives Tests")
# extracting features for positive test class
positive_test_features =extract_features(test_nv)
positive_test_class = np.ones((len(test_nv),1))

print("Training Negative Tests")
# extracting features for negative test class
negative_test_features =extract_features(test_les)
negative_test_class = np.zeros((len(test_les),1))

# concatenating the two features
train_feature = np.concatenate((positive_train_features,negative_train_features),axis=0)
train_class = np.concatenate((positive_train_class,negative_train_class),axis=0)

np.save("train_l.npy",train_feature)

# normalizing and shuffling the train features
train_feature,train_class,mean_x,std_x = features_normalize(train_feature,train_class)

# concatenating the test features
test_feature = np.concatenate((positive_test_features,negative_test_features),axis=0)
test_class = np.concatenate((positive_test_class,negative_test_class),axis=0)
np.save("test_l.npy",test_feature)

# normalizing and shuffling the test features
test_feature = (test_feature - mean_x) / std_x

where_are_NaNs = np.isnan(train_feature)
train_feature[where_are_NaNs] = 0

where_are_NaNs = np.isnan(test_feature)
test_feature[where_are_NaNs] = 0

# Training the classifier
#parameter_c =5
#kernel_svm = 'rbf'
#clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']

clf = LinearSVC(C = 0.1, random_state=42, tol=1e-5,max_iter=3000)
clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))



# Training the classifier
parameter_c =25
kernel_svm = 'rbf'
clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')

clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("Accuracy:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))
