#######################################################################
# USE THIS FILE FOR TRAINING, VALIDATION AND TESTING SETS TO GENERATE
# IMAGES FOR TRAINING. THIS REMOVES HAIR AND CORNERS. WE USE THIS
# AS THE ORIGINAL IMAGE.

# REPLACE THE EXTRACT FEATURES FUNCTION AND STORE THE RESULTS IN THE FOLDER
# ONE BY ONE. GIVE THE LIST OF THE IMAGES AND THE MASKS ARE STORED
######################################################################

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

k = 0
METHOD = 'uniform'
threshold = 3000

def get_mask(s):
    mask = np.zeros(s, dtype=np.uint8)
    rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 2) - 1, round(s[1] / 2) - 1)
    mask[rr, cc, :] = 1
    return mask

def remove_hair(cl1,cl1_gray):
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(cl1_gray, cv2.MORPH_BLACKHAT, kernel)
    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # inpaint the original image depending on the mask



    img_thresh = cl1
    test = get_mask(img_thresh.shape)
    cl1 = img_thresh * test

    dst = cv2.inpaint(cl1, thresh2, 1, cv2.INPAINT_TELEA)

    l, a, b = cv2.split(dst)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    l = clahe.apply(l)
    a = clahe.apply(a)
    b = clahe.apply(b)
    dst = cv2.merge((l, a, b))

    return dst



def extract_features(train_class,train_class_seg):
    k = 0
    for img_no in range(len(train_class)):
        k = k+1
        img_orig_grayscale = cv2.imread(train_class[img_no],cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread(train_class[img_no])
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_orig= remove_hair(img_orig, img_orig_grayscale)
        img_orig_grayscale = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)

        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

        print("writing")
        path = "C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\les_val\\" + "les" + "_" + str(
            k) + ".jpg"

        print(path)
        cv2.imwrite(path, img_orig)

        print(img_no)




filenames_positive_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\nv_mask\\*.jpg")
filenames_negative_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\les_mask\\*.jpg")

filenames_positive = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\nv\\*.jpg")
filenames_negative = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\les\\*.jpg")


filenames_positive_val_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\nv_mask\\*.jpg")
filenames_negative_val_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\les_mask\\*.jpg")

filenames_positive_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\nv\\*.jpg")
filenames_negative_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\les\\*.jpg")

filenames_positive_seg = sorted( filenames_positive_seg, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )
filenames_negative_seg = sorted( filenames_negative_seg, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )

filenames_positive_val_seg = sorted( filenames_positive_val_seg, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )
filenames_negative_val_seg = sorted( filenames_negative_val_seg, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )

filenames_positive.sort()
filenames_negative.sort()

filenames_positive_val.sort()
filenames_negative_val.sort()

FEATURES_LENGTH = 6
features_neg = np.zeros((len(filenames_positive_seg),FEATURES_LENGTH))
features_pos = np.zeros((len(filenames_negative_seg),FEATURES_LENGTH))

CROSS_VALIDATION = 5
TOTAL_TRAINING = 2400
STEP = TOTAL_TRAINING/CROSS_VALIDATION




# Training with original Images
test_les = filenames_negative_val
test_nv = filenames_positive_val
train_les = filenames_negative
train_nv = filenames_positive

#Loading Testing Images
test_les_seg = filenames_negative_val_seg
test_nv_seg = filenames_positive_val_seg
train_les_seg = filenames_negative_seg
train_nv_seg = filenames_positive_seg

#Change it to generate for the different data

#print("Training Positives")
# extracting features for positive class
#positive_train_features = extract_features(train_nv, train_nv_seg)
#positive_train_class = np.ones((len(train_nv),1))
#
#print("Training Negatives")
## extracting features for negative class
#negative_train_features=extract_features(train_les, train_les_seg)
#negative_train_class = np.zeros((len(train_les),1))
#
#print("Training Positives Tests")
# extracting features for positive test class
#positive_test_features =extract_features(test_nv, test_nv_seg)
#positive_test_class = np.ones((len(test_nv),1))
#
print("Training Negative Tests")
# extracting features for negative test class
negative_test_features =extract_features(test_les, test_les_seg)
negative_test_class = np.zeros((len(test_les),1))

