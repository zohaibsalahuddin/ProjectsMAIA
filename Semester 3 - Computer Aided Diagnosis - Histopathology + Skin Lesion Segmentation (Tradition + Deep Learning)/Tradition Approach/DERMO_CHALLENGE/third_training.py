#########################################################################
# RUN THIS AFTER:
# SUPER PIXEL MASKS HAVE BEEN GENERATED
# MODIFIED ORIGINAL IMAGES HAVE BEEN GENERATED. THEY WILL BE USED INSTEAD OF ORIGINAL.
##########################################################################

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



def check_mask(img_seg, threshold):
    test_segment = np.zeros(img_seg.shape)
    s = img_seg.shape
    rr, cc = ellipse(round(s[0] / 2), round(s[1] / 2), round(s[0] / 4) - 1, round(s[1] / 4) - 1)
    test_segment[rr, cc] = 1

    if (np.sum(img_seg) < threshold or np.sum(img_seg * test_segment) == 0):
        img_seg = test_segment

    return  img_seg

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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    a = clahe.apply(a)
    b = clahe.apply(b)
    dst = cv2.merge((l, a, b))

    return dst

def fd_hu_moments(image_grayscale):
    feature = cv2.HuMoments(cv2.moments(image_grayscale)).flatten()
    return feature


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(int)
    # compute the haralick texture feature vector

    haralick_mean = mahotas.features.haralick(gray).mean(axis=0)
    haralick_std = mahotas.features.haralick(gray).std(axis=0)
    haralick_range = mahotas.features.haralick(gray).ptp(axis=0)
    # return the result
    test = np.concatenate((haralick_mean,haralick_std,haralick_range),axis=0)
    return test

def fd_haralick_l(gray):
    gray = gray.astype(int)

    # convert the image to grayscale
    # compute the haralick texture feature vector
    haralick_mean = mahotas.features.haralick(gray).mean(axis=0)
    haralick_std = mahotas.features.haralick(gray).std(axis=0)
    haralick_range = mahotas.features.haralick(gray).ptp(axis=0)
    # return the result

    test = np.concatenate((haralick_mean,haralick_std,haralick_range),axis=0)
    return test

def mean_std_whole(img_orig):
    std_r = np.std(img_orig[:, :, 0] )
    std_g = np.std(img_orig[:, :, 1] )
    std_b = np.std(img_orig[:, :, 2] )
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = np.std(img_orig_n[:, :, 0] )
    std_g_1 = np.std(img_orig_n[:, :, 1] )
    std_b_1 = np.std(img_orig_n[:, :, 2] )

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = np.std(img_orig[:, :, 0] )
    std_g_2 = np.std(img_orig[:, :, 1] )
    std_b_2 = np.std(img_orig[:, :, 2] )

    return [std_r, std_g, std_b, std_r_1, std_g_1, std_b_1, std_r_2, std_g_2, std_b_2]

def mean_std(img_orig,mask):
    
    std_r = np.std(img_orig[:, :, 0] * mask)
    std_g = np.std(img_orig[:, :, 1] * mask)
    std_b = np.std(img_orig[:, :, 2] * mask)
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = np.std(img_orig_n[:, :, 0] * mask)
    std_g_1 = np.std(img_orig_n[:, :, 1] * mask)
    std_b_1 = np.std(img_orig_n[:, :, 2] * mask)

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = np.std(img_orig[:, :, 0] * mask)
    std_g_2 = np.std(img_orig[:, :, 1] * mask)
    std_b_2 = np.std(img_orig[:, :, 2] * mask)

    return [std_r,std_g,std_b,std_r_1,std_g_1,std_b_1,std_r_2,std_g_2,std_b_2]

def skew_seg(img_orig,mask):
    
    std_r = skew(img_orig[:, :, 0] * mask,axis=None)
    std_g = skew(img_orig[:, :, 1] * mask,axis=None)
    std_b = skew(img_orig[:, :, 2] * mask,axis=None)
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = skew(img_orig_n[:, :, 0] * mask,axis=None)
    std_g_1 = skew(img_orig_n[:, :, 1] * mask,axis=None)
    std_b_1 = skew(img_orig_n[:, :, 2] * mask,axis=None)

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = skew(img_orig[:, :, 0] * mask,axis=None)
    std_g_2 = skew(img_orig[:, :, 1] * mask,axis=None)
    std_b_2 = skew(img_orig[:, :, 2] * mask,axis=None)

    return [std_r,std_g,std_b,std_r_1,std_g_1,std_b_1,std_r_2,std_g_2,std_b_2]

def skew_whole(img_orig):

    std_r = skew(img_orig[:, :, 0],axis=None )
    std_g = skew(img_orig[:, :, 1],axis=None )
    std_b = skew(img_orig[:, :, 2],axis=None )
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = skew(img_orig_n[:, :, 0],axis=None )
    std_g_1 = skew(img_orig_n[:, :, 1],axis=None )
    std_b_1 = skew(img_orig_n[:, :, 2],axis=None )

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = skew(img_orig[:, :, 0] ,axis=None)
    std_g_2 = skew(img_orig[:, :, 1] ,axis=None)
    std_b_2 = skew(img_orig[:, :, 2] ,axis=None)

    return [std_r,std_g,std_b,std_r_1,std_g_1,std_b_1,std_r_2,std_g_2,std_b_2]

def kurtosis_whole(img_orig):
 
    std_r = kurtosis(img_orig[:, :, 0] ,axis=None)
    std_g = kurtosis(img_orig[:, :, 1] ,axis=None)
    std_b = kurtosis(img_orig[:, :, 2] ,axis=None)
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = kurtosis(img_orig_n[:, :, 0] ,axis=None)
    std_g_1 = kurtosis(img_orig_n[:, :, 1] ,axis=None)
    std_b_1 = kurtosis(img_orig_n[:, :, 2] ,axis=None)

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = kurtosis(img_orig[:, :, 0] ,axis=None)
    std_g_2 = kurtosis(img_orig[:, :, 1] ,axis=None)
    std_b_2 = kurtosis(img_orig[:, :, 2] ,axis=None)

    return [std_r,std_g,std_b,std_r_1,std_g_1,std_b_1,std_r_2,std_g_2,std_b_2]

def kurtosis_seg(img_orig,mask):

    std_r = kurtosis(img_orig[:, :, 0] * mask,axis=None)
    std_g = kurtosis(img_orig[:, :, 1] * mask,axis=None)
    std_b = kurtosis(img_orig[:, :, 2] * mask,axis=None)
    img_orig_n = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)

    std_r_1 = kurtosis(img_orig_n[:, :, 0] * mask,axis=None)
    std_g_1 = kurtosis(img_orig_n[:, :, 1] * mask,axis=None)
    std_b_1 = kurtosis(img_orig_n[:, :, 2] * mask,axis=None)

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2LAB)

    std_r_2 = kurtosis(img_orig[:, :, 0] * mask,axis=None)
    std_g_2 = kurtosis(img_orig[:, :, 1] * mask,axis=None)
    std_b_2 = kurtosis(img_orig[:, :, 2] * mask,axis=None)

    return [std_r,std_g,std_b,std_r_1,std_g_1,std_b_1,std_r_2,std_g_2,std_b_2]

def glcm_features(img_orig,mask,img_gray):
    mask_idx = mask.astype('uint8')
    segment_region = np.multiply(img_gray, mask_idx)
    segment_region = segment_region.astype('uint8')
    comp_temp = ma.array(segment_region, mask=~mask_idx)
    comp_temp = comp_temp.astype('uint8')
    comp_temp = np.delete(comp_temp, np.argwhere(np.all(comp_temp[..., :] == 0, axis=0)), axis=1)
    comp_temp = np.delete(comp_temp, np.where(~comp_temp.any(axis=1))[0], axis=0)
    comp_temp = np.delete(comp_temp, np.argwhere(np.all(comp_temp[..., :] == 0, axis=0)), axis=1)
    comp_temp = np.delete(comp_temp, np.where(~comp_temp.any(axis=1))[0], axis=0)

    compressed_segment_region = ma.array(segment_region, mask=~mask_idx).compressed()
    glcm = greycomatrix(comp_temp, [5], [0], 256)

    stats = ["dissimilarity", "correlation", "contrast", "homogeneity", "ASM", "energy"]
    dissimilarity = greycoprops(glcm, stats[0])[0, 0]
    correlation = greycoprops(glcm, stats[1])[0, 0]
    contrast = greycoprops(glcm, stats[2])[0, 0]
    homogeneity = greycoprops(glcm, stats[3])[0, 0]
    ASM = greycoprops(glcm, stats[4])[0, 0]
    energy = greycoprops(glcm, stats[5])[0, 0]

    temp_features = [dissimilarity, correlation,contrast,homogeneity,ASM,energy]
    return temp_features

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

def fd_histogram_f(image, mask):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,0] = image[:,:,0] * mask
    image[:,:,1] = image[:,:,1] * mask
    image[:,:,2] = image[:,:,2] * mask
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def fd_histogram(image):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,0] = image[:,:,0]
    image[:,:,1] = image[:,:,1]
    image[:,:,2] = image[:,:,2]
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def lbp_gray_images(img_orig_grayscale):
    lbp = local_binary_pattern(img_orig_grayscale, 24, 3, METHOD)
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    return  hist

def extract_features(train_class,train_class_seg):
    for img_no in range(len(train_class)):
        img_orig_grayscale = cv2.imread(train_class[img_no],cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread(train_class[img_no])
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        img_seg = cv2.imread(train_class_seg[img_no],cv2.IMREAD_GRAYSCALE)
        img_seg = img_seg/255
        img_seg = check_mask(img_seg,threshold)
        #### INSERT FEATURES HERE ######

        std_f = mean_std(img_orig,img_seg) #76.41 use this
        #glcm_f = glcm_features(img_orig,img_seg,img_orig_grayscale) #74.5 use this
        std_f_w = mean_std_whole(img_orig) #64.08 dont use this
        skew_f = skew_seg(img_orig,img_seg)
        skew_f_w =skew_whole(img_orig)
        kurt_f = kurtosis_seg(img_orig,img_seg)
        kurt_f_w = kurtosis_whole(img_orig)
        #glcm_f_w = glcm_features_whole(img_orig_grayscale) #73.83 use this
        fd_hu_moments_g = fd_hu_moments(img_orig_grayscale) #63.25 dont use this
        fd_hu_moments_l = fd_hu_moments(img_seg) # 56.25 dont use this
        fd_histogram_g = fd_histogram(img_orig) #78.25
        fd_histogram_l = fd_histogram_f(img_orig, img_seg) #76
        fd_haralick_g = fd_haralick(img_orig)

        fd_haralick_lo = fd_haralick_l(img_orig_grayscale * img_seg)
        desc = mahotas.features.zernike_moments(img_seg,21)
        lbp_l = lbp_gray_images(img_orig_grayscale * img_seg)
        lbp_g = lbp_gray_images(img_orig_grayscale)
        
        #### COMBINE FEATURES HERE INTO A VECTOR #####
        feature_v = np.hstack((skew_f_w,kurt_f_w,std_f_w,fd_haralick_g,fd_histogram_g,fd_hu_moments_g,lbp_g,skew_f,kurt_f,std_f,fd_haralick_lo,fd_hu_moments_l))

        feature_v = np.expand_dims(feature_v,0)
        if img_no == 0:
            feature_vector = feature_v
        else:
            feature_vector = np.concatenate((feature_vector,feature_v),axis=0)

        print(img_no)
    return feature_vector


def features_normalize(feature_vectors,class_vectors):
    mean_x = np.mean(feature_vectors, axis=0)
    std_x = np.std(feature_vectors, axis=0)
    
    where_are_zeros = (std_x == 0)
    std_x[where_are_zeros] = 1
    
    randomize = np.arange(len(class_vectors))
    np.random.shuffle(randomize)
    feature_vectors_n = feature_vectors[randomize, :]
    feature_vectors_n = (feature_vectors_n - np.mean(feature_vectors_n, axis=0)) / np.std(feature_vectors_n, axis=0)
    class_vectors_n = class_vectors[randomize]
    return feature_vectors_n,class_vectors_n,mean_x,std_x


filenames_positive_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\nv_mask\\*.jpg")
filenames_negative_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train\\les_mask\\*.jpg")

filenames_positive = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\nv\\*.jpg")
filenames_negative = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\les\\*.jpg")


filenames_positive_val_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\nv_mask\\*.jpg")
filenames_negative_val_seg = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Validation\\val\\les_mask\\*.jpg")

filenames_positive_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\nv_val\\*.jpg")
filenames_negative_val = glob.glob("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\les_val\\*.jpg")


filenames_positive = sorted( filenames_positive, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )
filenames_negative = sorted( filenames_negative, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )

filenames_positive_val = sorted( filenames_positive_val, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )
filenames_negative_val = sorted( filenames_negative_val, key=lambda a: int((a.split("_")[-1]).split(".")[0]) )

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

print("Training Positives")
# extracting features for positive class
positive_train_features = extract_features(train_nv, train_nv_seg)
positive_train_class = np.ones((len(train_nv),1))

print("Training Negatives")
# extracting features for negative class
negative_train_features=extract_features(train_les, train_les_seg)
negative_train_class = np.zeros((len(train_les),1))

print("Training Positives Tests")
# extracting features for positive test class
positive_test_features =extract_features(test_nv, test_nv_seg)
positive_test_class = np.ones((len(test_nv),1))

print("Training Negative Tests")
# extracting features for negative test class
negative_test_features =extract_features(test_les, test_les_seg)
negative_test_class = np.zeros((len(test_les),1))

# concatenating the two features
train_feature = np.concatenate((positive_train_features,negative_train_features),axis=0)
train_class = np.concatenate((positive_train_class,negative_train_class),axis=0)


####################################################################################
#SAVING FEATURES
#####################################################################################
np.save("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train_p.npy",positive_train_features)
np.save("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\train_n.npy",negative_train_features)

# normalizing and shuffling the train features
train_feature,train_class,mean_x,std_x = features_normalize(train_feature,train_class)

# concatenating the test features
test_feature = np.concatenate((positive_test_features,negative_test_features),axis=0)
test_class = np.concatenate((positive_test_class,negative_test_class),axis=0)

np.save("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\test_p.npy",positive_test_features)
np.save("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\Training\\test_n.npy",negative_test_features)


# normalizing and shuffling the test features
test_feature = (test_feature - mean_x) / std_x

where_are_NaNs = np.isnan(train_feature)
train_feature[where_are_NaNs] = 0

where_are_NaNs = np.isnan(test_feature)
test_feature[where_are_NaNs] = 0


# Training the classifier
######################################################################


clf = LinearSVC(C = 0.1, random_state=42, tol=1e-5)
clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("LinearSVM Accuracy:" + str(score))


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
# Training the classifier
parameter_c =10
kernel_svm = 'rbf'
clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')

clf.fit(train_feature, train_class)

# calculating the accuracy
predict = clf.predict(test_feature)
score = accuracy_score(np.asarray(test_class), predict)
print("SVM Accuracy:" + str(score))
print(classification_report(np.asarray(test_class), predict, target_names=target_names))


#####################################################################################
#This code is for cross validation
####################################################################################

train = np.load("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\train_n.npy")
test = np.load("C:\\Users\\hp\\Desktop\\Lectures - Semester 3\\Labs\\Computer Aided Diagnosis Lab\\Dermoscopic\\test_n.npy")

# concatenating the two features
train_feature = np.concatenate((positive_train_features,negative_train_features),axis=0)
train_class = np.concatenate((positive_train_class,negative_train_class),axis=0)

test_feature = np.concatenate((positive_test_features,negative_test_features),axis=0)
test_class = np.concatenate((positive_test_class,negative_test_class),axis=0)

#positives = np.concatenate((np.concatenate((positive_train_features[:,:676],train[:2400,:]),axis=1),np.concatenate((positive_test_features[:,:676],test[:600,:]),axis=1)),axis=0)
#negatives = np.concatenate((np.concatenate((negative_train_features[:,:676],train[2400:,:]),axis=1),np.concatenate((negative_test_features[:,:676],test[600:,:]),axis=1)),axis=0)

positives = np.concatenate((positive_train_features[:,:676],positive_test_features[:,:676]),axis=0)
negatives = np.concatenate((negative_train_features[:,:676],negative_test_features[:,:676]),axis=0)


positives_class = np.concatenate((positive_train_class,positive_test_class),axis=0)
negatives_class = np.concatenate((negative_train_class,negative_test_class),axis=0)

folds = 8
step = 475
for i in range(0,folds):
    print(i+1)
    test_feature_n = np.concatenate((positives[(i*step):(i+1)*step, :], negatives[(i*step):(i+1)*step, :]), axis=0)
    test_class = np.concatenate((positives_class[(i*step):(i+1)*step, :], negatives_class[(i*step):(i+1)*step, :]), axis=0)

    train_feature_n = np.concatenate((positives[:i*step,:],positives[(i+1)*step:,:],negatives[:i*step,:],negatives[(i+1)*step:,:]),axis=0)
    train_class = np.concatenate((positives_class[:i*step,:],positives_class[(i+1)*step:,:],negatives_class[:i*step,:],negatives_class[(i+1)*step:,:]),axis=0)


    training = train_feature_n
    testing = test_feature_n

    training,train_class,mean_x,std_x = features_normalize(training,train_class)

    # normalizing and shuffling the test features
    testing = (testing - mean_x) / std_x

    where_are_NaNs = np.isnan(training)
    training[where_are_NaNs] = 0

    where_are_NaNs = np.isnan(testing)
    testing[where_are_NaNs] = 0

    training_n = (training)
    testing_n = (testing)

    from sklearn.metrics import classification_report
    target_names = ['class 0', 'class 1']
    # Training the classifier
    parameter_c = 12
    kernel_svm = 'rbf'
    clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')

    clf.fit(training_n, train_class)

    # calculating the accuracy
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("Accuracy:" + str(score))
    print(classification_report(np.asarray(test_class), predict, target_names=target_names))

#################################################################################
#PCA
################################################################################
for n in range(510,570,30):
    from sklearn.decomposition import PCA
    print('--------------------')
    print(n)
    pca = PCA(n_components=n) #480 give us the best accuracy
    pca.fit(train_feature)

    training_n = pca.transform(train_feature)
    testing_n = pca.transform(test_feature)


    # Linear Classifier
    target_names = ['class 0', 'class 1']
    clf = LinearSVC(C = 0.1, random_state=42, tol=1e-5)
    clf.fit(training_n, train_class)
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("LinearSVM Accuracy:" + str(score))


    from sklearn.metrics import classification_report
    target_names = ['class 0', 'class 1']
    
    # SVM classifier
    parameter_c = 10
    kernel_svm = 'rbf'
    clf = svm.SVC(C=parameter_c, kernel= kernel_svm, class_weight='balanced')
    clf.fit(training_n, train_class)
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("SVM Accuracy:" + str(score))
   # print(classification_report(np.asarray(test_class), predict, target_names=target_names))


    #RandomForest
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    clf.fit(training_n, train_class)
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("RandomForest Accuracy Testing:" + str(score))
   # print(classification_report(np.asarray(test_class), predict, target_names=target_names))

    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(training_n, train_class)
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("Logistic Regression Testing:" + str(score))
   # print(classification_report(np.asarray(test_class), predict, target_names=target_names))

    #adaboost
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(training_n, train_class)
    predict = clf.predict(testing_n)
    score = accuracy_score(np.asarray(test_class), predict)
    print("AdaBoost Testing:" + str(score))
   # print(classification_report(np.asarray(test_class), predict, target_names=target_names))
