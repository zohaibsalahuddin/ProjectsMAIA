# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:23:55 2019
README
The first functions get the images.
This code was executed in a folder where there was also the folders provided for this lab:
    training-images
    training-mask
    training-labels
    testing-images
    testing-mask
    testing-labels
    

@authors: Zohaib & Isaac (MAIA_3, 3rd Semester, Girona)

Tissue Model based on the Raw Brain (without registration)

"""

import os
import time
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion

def getImgs():
    '''
    This function explores the folders in the same directory path.
    '''
    t0 = time.time()
    print('reading training images, masks and labels...')
    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, 'training-images')
    img = load_images_from_folder(data_folder)
    
    data_folder = os.path.join(root_folder, 'training-mask')
    mask = load_images_from_folder(data_folder)
    
    data_folder = os.path.join(root_folder, 'training-labels')
    gt = load_images_from_folder(data_folder)
    print('Getting all data took ', round(time.time()-t0, 2), 'seconds')
    return img, mask, gt

def getImgsTest():
    '''
    This function explores the folders in the same directory path.
    '''
    t0 = time.time()
    print('reading testing images, masks and labels...')
    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, 'testing-images')
    img = load_images_from_folder(data_folder)
    
    data_folder = os.path.join(root_folder, 'testing-mask')
    mask = load_images_from_folder(data_folder)
    
    data_folder = os.path.join(root_folder, 'testing-labels')
    gt = load_images_from_folder(data_folder)
    print('Getting all data took ', round(time.time()-t0, 2), 'seconds')
    return img, mask, gt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            img = nib.load(os.path.join(folder, filename))
            images.append(img)
    return images


def load_gt_from_folder(folder):
    gt = []
    for filename in os.listdir(folder):
        if filename.endswith(".nii"):
#            img = cv.imread(os.path.join(folder, filename))
            img = nib.load(os.path.join(folder, filename))
            if img is not None:
                if filename.endswith("g.nii"):
                    gt.append(img)
    return gt


def N4(nifti_input, maskImage):
    print("N4 bias correction runs.")
    inputImage = sitk.ReadImage(nifti_input)
#    inputImage = sitk.ReadImage("06-t1c.nii.gz")
    # maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
#    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
#    sitk.WriteImage(maskImage, "06-t1c_mask3.nii.gz")

#    inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter();

    output = corrector.Execute(inputImage,maskImage)
#    sitk.WriteImage(output,"06-t1c_output3.nii.gz")
    corr = sitk.GetArrayFromImage(output)
    print("Finished N4 Bias Field Correction.....")
    return corr


def load_itk(filename):
    '''
    credits: kaggle.com/arnavkj95
    '''
    
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    imgArr = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
#    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
#    spacing = np.array(list(reversed(itkimage.GetSpacing())))

#    return ct_scan, origin, spacing
    return imgArr

def getArrays(img, mask, gt):
    '''
    This function gets the arrays of numbers from the nifti files
    '''
    gtA = []
    imgA = []
    maskA = []
    for i in range(len(img)):
        imgA.append(img[i].get_fdata())
    for i in range(len(gt)):
        maskA.append((mask[i].get_fdata()).astype(np.bool))
    for i in range(len(img)):
        gtA.append(gt[i].get_fdata())
    return imgA, maskA, gtA

def mask(gt, img):
    '''
    We use the Ground Truth for skull-stripping
    '''
    imgM = []
    lenGt = len(gt)
    for i in range(lenGt):
        imgM.append(img[i] * (gt[i]>0))
    
    return imgM



class Peak:
    '''
    https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
    '''
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]

def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)


def peakFind(img):
    '''
    Returns the 2 most disctinctive intensity peaks, and the valley in between
    '''
    t0 = time.time()
    if type(img) == np.ndarray:
        img = list([img])
    pkall = [0,0,0]
    for imm in range(len(img)):
        mxI = int(np.max(img[imm]))
        asd = np.histogram(img[imm], range = (1,mxI), bins = mxI)[0]
        asd2 = get_persistent_homology(asd)[:3]
        pks=[]
        for i in range(3):
#            print(asd2[i].born)
            pks.append(asd2[i].born)
        p1 = min(pks[1], pks[0])
        p2 = max(pks[1], pks[0])
        p3 = p1 + get_persistent_homology(p2-asd[p1:p2])[0].born
        pks[2] = p3
        
        if 0:
            plt.plot(asd)
            for i in range(3):
                plt.plot(pks[i], asd[pks[i]], 'ro')
            plt.show()
        pkall = np.vstack((pkall, pks))
    print('Finding all peaks took ', round(time.time()-t0, 2), 'seconds')
    return pkall[1:]
    

def peakEq(img, pkall, gam = False):
    t0 = time.time()
    if type(img) == np.ndarray:
        img = list([img])
    imtr = []
    for imm in range(len(img)):
#        print('processing image ', imm, '...')
        pks2 = np.sort(pkall[imm,:])
        imgSc = (((img[imm]) - pks2[0])/(pks2[2]-pks2[0]))
        if gam:
            gamma = (np.log(0.5)/np.log((pks2[1]-pks2[0])/(pks2[2]-pks2[0])))
            imgSc[imgSc>0] = imgSc[imgSc>0]**gamma
            imgSc[imgSc<0] = -abs(imgSc[imgSc<0])**gamma
        imtr.append(imgSc)
    print('Transforming all brains took ', round(time.time()-t0, 2), 'seconds')
    return imtr

def tisMod(imgEq, gtA, mod = 2, b=300):
    t0 = time.time()
    allHist = 0
    t1, t2, t3 = 0, 0, 0
    for imm in range(len(imgEq)):
#        print('processing image ', imm, '...')
        totI = int(np.sum(imgEq[imm]>0))
        mxI = int(np.max(imgEq[imm][imgEq[imm]>0]))
        mnI = int(np.min(imgEq[imm][imgEq[imm]>0]))
        if mod == 1:
            t1 = t1 + np.histogram(imgEq[imm][gtA[imm] == 1], range = (mnI,mxI), bins = b)[0]/totI
#            plt.plot(t1), plt.show()
            t2 = t2 + np.histogram(imgEq[imm][gtA[imm] == 2], range = (mnI,mxI), bins = b)[0]/totI
#            plt.plot(t2), plt.show()
            t3 = t3 + np.histogram(imgEq[imm][gtA[imm] == 3], range = (mnI,mxI), bins = b)[0]/totI
#            plt.plot(t3), plt.show()
        if mod == 2:
            t1 = t1 + np.histogram(imgEq[imm][gtA[imm] == 1], range = (-3,3), bins = b)[0]/totI
            t2 = t2 + np.histogram(imgEq[imm][gtA[imm] == 2], range = (-3,3), bins = b)[0]/totI
            t3 = t3 + np.histogram(imgEq[imm][gtA[imm] == 3], range = (-3,3), bins = b)[0]/totI
    allHist = t1 + t2 + t3
    for i in np.where(allHist>0)[0]:
        t1[i] = t1[i]/allHist[i]
        t2[i] = t2[i]/allHist[i]
        t3[i] = t3[i]/allHist[i]
    print('Getting the tissue models took ', round(time.time()-t0, 2), 'seconds')
    return t1, t2, t3

def getTMB(imgM, t1, t2, t3, mode = 2, binss = 100):
    '''
    This function returns the 3 tissue probabilities of the given masked brain
    modes:
        0: raw histogram (not active)
        1: histogram stretching
        2: 2 peak adaptive
    '''
    t0 = time.time()
    b1 = np.zeros(imgM.shape)
    b2 = np.zeros(imgM.shape)
    b3 = np.zeros(imgM.shape)
    if mode == 0: # raw histogram
        print('this mode is not active')
    elif mode == 1: # histogram stretching
#        totI = int(np.sum(imgM>0))
        mxI = int(np.max(imgM[imgM>0]))
        mnI = int(np.min(imgM[imgM>0]))
        b = (mxI-mnI)/binss
        for v in range(binss):
            lb = imgM >= (mnI+b*v)
            hb = imgM <(mnI+b*(v+1))
            wh = lb*hb
            b1[wh] = t1[v]
            b2[wh] = t2[v]
            b3[wh] = t3[v]
    elif mode == 2: # 2-peak histogram matching
        # We get the main peaks from the image
        pkall0 = peakFind(imgM)
        
        # We transform the brain to have similar distributions (True = apply gamma)
        imgEq0 = peakEq(imgM, pkall0, False)
        imgEq0 = imgEq0[0]
#        hst = np.histogram(imgEq0[imgEq0>0], range = (-3,3), bins = binss)[0]
#        plt.plot(hst)
#        plt.figure
#        plt.plot(t1, 'r', t2, 'g', t3, 'b')
#        mxI = int(np.max(imgEq0[imm]))
#        t1 = np.histogram(imgEq[imm], range = (-3,3), bins = binss)[0]
        # Associate each voxel to its respective tissue probability
        b = (3-(-3))/binss
        for v in range(binss):
            lb = imgEq0 >= (-3+b*v)
            hb = imgEq0 <(-3+b*(v+1))
            wh = lb*hb
            b1[wh] = t1[v]
            b2[wh] = t2[v]
            b3[wh] = t3[v]
        
    else:
        print('mode has to be a number form 0 to 2: \n0: raw histogram\n1: histogram stretching\n2: 2 peak adaptive\n')
        b1, b2, b3 = [], [], []
    print('Applying the Tissue Models took ', round(time.time()-t0, 2), 'seconds')
    return b1, b2, b3
    
def cleanTM(t1, t2, t3):
    t0 = time.time()
    h = int(np.where(t2 == max(t2))[0][0])
    t2[h:] = 1
    t3[h:] = 0
    l = int(np.where(t1 == max(t1))[0][0])
    t1[:l] = 1
    print('Cleaning the tissue models took ', round(time.time()-t0, 2), 'seconds')
    return t1, t2, t3


def get1Dice(gt, img):
    ''' Returns the Dice scores for all tissues (labels > 0) for 1 case
    gt : Ground Truth array
    img: Segmented images array
    '''
#    labs = len(np.unique(gt))
    dices = []
    for l in range(1, 4):
        dices.append(2*np.sum((gt==l)*(img==l))/(np.sum(gt==l)+np.sum(img==l)))
    return dices

def ani_dif(img, niter=5, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    print('Applying Anisotropic Diffusion...')
    t0 = time.time()
    if type(img) == np.ndarray:
        img = list([img])
    im_dif = []
    for imm in range(len(img)):    
        im_dif.append(anisotropic_diffusion(img[imm], niter, kappa, gamma, voxelspacing, option)*(img[imm]>0))
    print('Applying Anisotropic Diffusion took ', round(time.time()-t0, 2), 'seconds')
    return im_dif

'''
END of function definitions --------------------------------------------------
'''


'''
START OF CODE
'''
img, imask, gt = getImgs()

# Saving the actual arrays inside the nifti files
imgA, imask, gtA = getArrays(img, imask, gt)

# Apply N4 bias correction (discarded for this set)

# Masking of the brain using the GT (yes, using the GT)
imgM = mask(imask, imgA)

#Pre-proc using anisotr. diff (discarded for this set)
#im_dif = ani_dif(imgM, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
#imgM = ani_dif(imgM)

# We get the main peaks from the images
pkall = peakFind(imgM)

# We transform the brains to have similar distributions (True = apply gamma)
imgEq = peakEq(imgM, pkall, False)

# We get the different tissue probabilities
num_bins = 100
mode = 2
t1, t2, t3 = tisMod(imgEq, gtA, mode, num_bins)
#t1, t2, t3 = tisMod(imgM, gtA, mode, num_bins)
#plt.plot(t1, 'r', t2, 'g', t3, 'b')

# Cleaning of the tissue models
t1, t2, t3 = cleanTM(t1, t2, t3)
plt.figure()
plt.plot(t1, 'r', t2, 'g', t3, 'b')

# Reading test images
timg, timask, tgt = getImgsTest()


# Saving the actual arrays inside the nifti files
timgA, timask, tgtA = getArrays(timg, timask, tgt)


# Masking of the brain using the GT (yes, using the GT)
timgM = mask(timask, timgA)

# Applying Anisotropic Diffusion (slightly worse results, unfortunately)
#timgM = ani_dif(timgM)

# get the tissue probabilities for a given msked brain, and save the 3 prob. 3d maps
root_folder = os.getcwd()

save_predictions = 0
if save_predictions:
    for x in range(0, len(timgM)):
        b1, b2, b3 = getTMB(timgM[x], t1, t2, t3, mode, num_bins)
        
        header = timg[x].header
        affine = timg[x].affine
        
        ni_CSF = nib.Nifti1Image(b1, affine, header)
        ni_WM = nib.Nifti1Image(b2, affine, header)
        ni_GM = nib.Nifti1Image(b3, affine, header)
        
        newpath = r'/M3/MISA/labs/3/TM/mod1_500/'+str(x)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(os.path.join(root_folder, '/M3/MISA/labs/3/TM/mod1_500/'+str(x)))
        
        nib.save(ni_CSF, 'CSF.nii')
        nib.save(ni_GM, 'GM.nii')
        nib.save(ni_WM, 'WM.nii')




# Here we get all the dices
alldices = []
for x in range(len(timgM)):
    b1, b2, b3 = getTMB(timgM[x], t1, t2, t3, mode, num_bins)
    b1[timask[x]==0] = 0
    b2[timask[x]==0] = 0
    b3[timask[x]==0] = 0
    pr = np.zeros(timgM[x].shape)
    pr[timask[x]>0] = 1
    pr[(b2>b1) * (b2>b3)] = 2
    pr[(b3>b1) * (b3>b2)] = 3
    dice = get1Dice(tgtA[x], pr)
    alldices.append(dice)

alldices = np.asarray(alldices)