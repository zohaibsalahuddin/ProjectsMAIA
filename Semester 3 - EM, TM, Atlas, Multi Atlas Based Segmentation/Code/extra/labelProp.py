# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:23:55 2019
README
    > The first functions get the images.
    > This code was executed in a folder where there was also the folders provided for this lab:
        training-images
        training-mask
        training-labels
        testing-images
        testing-mask
        testing-labels
    > Reference image/atlas, associated mask and labels are changed starting on line 133
    

@authors: Zohaib & Isaac (MAIA_3, 3rd Semester, Girona)

Label propagation

"""




import os
import time
import numpy as np
import nibabel as nib
import SimpleITK as sitk


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


def getGTA(gt):
    '''
    This function gets the array of numbers from the nifti GT files
    '''
    gtA = []
    for i in range(len(gt)):
        gtA.append(gt[i].get_fdata())
    return gtA


def mask(gt, img):
    '''
    We use the Ground Truth for skull-stripping
    '''
    imgM = []
    lenGt = len(gt)
    for i in range(lenGt):
        imgM.append(img[i] * (gt[i]>0))
    
    return imgM


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


# ===========================================================================

'''
    Here we register the reference image (atlas?) to the target image (test)
'''



if 1:
    t0 = time.time()
    rootdir = "D:\\M3\\MISA\\labs\\3\\testing-images"   # test images
    base = "D:\\M3\\MISA\\labs\\3"                      # main folder
    
    # Change these to try a new reference
    if 1: # we get a single training image as reference
        reff = '1008'
        basedir = "D:\\M3\\MISA\labs\\3\\reg\\ref" + reff         # New place for results #-----------------------RESULTS FOLDER
        moving = "D:\\M3\\MISA\\labs\\3\\training-images\\" + reff + ".nii.gz"
        moving_mask = "D:\\M3\\MISA\\labs\\3\\training-mask\\" + reff + "_1C.nii.gz"
        transformThis = "D:\\M3\\MISA\\labs\\3\\training-labels\\" + reff + "_3C.nii.gz" #----------------------LABELS------------------
        maskFlag = 1
    else:
        basedir = "D:\\M3\\MISA\labs\\3\\reg\\maiaMNI"        # New place for results #---------------------------- RESULTS FOLDER
        moving = "D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas\\template.nii.gz"
        moving_mask = "D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas\\templ_mask.nii"
        transformThis = "D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas\\templ_lab.nii" #---------------------------LABELS-------------
        maskFlag = 1
    
    affine = base + "\\par0000affine.txt"
    bspline = base + "\\par0000bspline.txt"
#    fixed_reference_image = 'mni'

    command = "mkdir " + basedir
    os.system(command)
    cnt = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            cnt = cnt + 1
            #print(file)
            fullpath = os.path.join(subdir, file)
            fname = file.split(".")
#            if (fname[0] != str(fixed_reference_image)):
            print('registering', fname[0], '[', cnt, '/20]')
            result_dir = basedir + "\\" + fname[0]
            command= "mkdir " + result_dir
            os.system(command)

#            fixed = rootdir + "\\" + str(fixed_reference_image) + ".nii"
#            fixed_mask = base +"\\training-mask" + "\\" + str(fixed_reference_image) + "_1C.nii"
            fixed = rootdir + "\\" + fname[0] + ".nii.gz"
            fixed_mask = base + "\\testing-mask" + "\\" + fname[0] + "_1C.nii.gz"
            if maskFlag:
                command = "elastix -f " + fixed + " -fMask " + fixed_mask + " -m " + moving + " -mMask " + moving_mask + " -p " +  affine + " -p " +  bspline + " -out " + result_dir
            else:
                command = "elastix -f " + fixed + " -m " + moving + " -p " +  affine + " -p " +  bspline + " -out " + result_dir
#            print(command)
            os.system(command)
            print('about ', round((time.time()-t0)/60*((20-cnt)/cnt),0), 'minutes left')
    print("Registering to all test images took about ", (time.time()-t0), " seconds in total")



''' 
    We change the transformation to be Nearest Neigh.
'''
if 1:
    # Here we change the parameters of the transformation in order to apply nearest neighbors
    rootdir = basedir
    transform = "TransformParameters.1.txt"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
#            print(file)
            if (file == "TransformParameters.0.txt" or file == "TransformParameters.1.txt"):
                fullpath = os.path.join(subdir, file)
                with open(fullpath, 'r+') as f:
                    content = f.read()
                    f.seek(0)
                    f.truncate()
                    f.write(content.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0'))
            
    
''' 
    Here we apply label propagation
'''
if 1:
    rootdir = basedir
    transform = "TransformParameters.1.txt"
#    transformThis = "D:\\M3\\MISA\\labs\\3\\training-labels\\1006_3C.nii.gz" #---------------------------LABELS-------------------------
    for subdir, dirs, files in os.walk(rootdir):
        for fol in dirs:
            if fol != "labProp":
                print('Propagating atlas labels to test image', fol, '...')
                result_dir = basedir + "\\" + fol  + "\\labProp"
                
                command= "mkdir " + result_dir
                os.system(command)
    
                transform_dir = rootdir + "\\" + fol + "\\" + transform
                command = "transformix -in " + transformThis + " -out " + result_dir + " -tp " + transform_dir
                os.system(command) 


''' 
    Here we get the dice scores
'''

if 1:
    images = []
    cnt = 0
    noImg = []
    for subdir, dirs, files in os.walk(basedir):
        for fol in dirs:
            if fol != "labProp":
                print('Processing', fol, '...')
                try:
                    result_dir = basedir + "\\" + fol  + "\\labProp\\result.mhd"
                    
                    itkimage = sitk.ReadImage(result_dir)
                
                    # Get array
                    imgArr = sitk.GetArrayFromImage(itkimage)
                except:
                    print('ALERT! Could not find the image')
                    imgArr = np.zeros((5,5,5))
                    noImg.append(cnt)
                images.append(imgArr)
                cnt = cnt + 1

if 'tgtA' not in locals():
    print('Reading test GT to compute Dice scores...')
    # Reading test images
    timg, timask, tgt = getImgsTest()
    
    # Saving the actual arrays inside the nifti files
    tgtA = getGTA(tgt)

# Here I try to get all the dices
alldices = []
im2 = images.copy()
for x in range(len(tgtA)):
    print('Calculating the Dice scores...')
    if x not in noImg:
        im2[x] = np.swapaxes(im2[x], 0,2)
#        commented lines for future merging of methods
    #    b1, b2, b3 = getTMB(timgM[x], t1, t2, t3, mode, num_bins)
    #    b1[timask[x]==0] = 0
    #    b2[timask[x]==0] = 0
    #    b3[timask[x]==0] = 0
    #    pr = np.zeros(timgM[x].shape)
    #    pr[timask[x]>0] = 1
    #    pr[(b2>b1) * (b2>b3)] = 2
    #    pr[(b3>b1) * (b3>b2)] = 3
        im2[x][(im2[x] == 0) * (tgtA[x] > 0)] = 3
        im2[x] = im2[x] * (tgtA[x] > 0)
        dice = get1Dice(tgtA[x], im2[x])
    else:
        dice = np.asarray([0,0,0])    
    alldices.append(dice)

alldices = np.asarray(alldices)


print("Running everything took about ", (time.time()-t0), " seconds in total")















# This is how we got the mask and label map of the given MNI space (perhaps on future work we could be more strict on the threshold)
if 0:
    templ = nib.load("D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas\\template.nii.gz")
    templA = templ.get_fdata()    # now we have the 4 layers in an extra dimension
    atlas = nib.load("D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas\\atlas.nii.gz")
    atlasA = atlas.get_fdata()    # now we have the 4 layers in an extra dimension
    #we get the mask
    maskA = atlasA[:,:,:,0]<1
    b1 = atlasA[:,:,:,1]
    b2 = atlasA[:,:,:,3]
    b3 = atlasA[:,:,:,2]
    
    # we build the labels (careful! they are: out, csf, gm, wm... and we need out, csf, wm, gm)
    pr = maskA*1 + (b2>b1) * (b2>b3)*1 + (b3>b1) * (b3>b2)*2
    
#    we save the labels
    header = templ.header
    affine = templ.affine
    
    ni_lab = nib.Nifti1Image(pr, affine, header)
    ni_mask = nib.Nifti1Image(maskA, affine, header)
    
    os.chdir("D:\\M3\\MISA\\labs\\3\\MNI_sp\\MNITemplateAtlas")
    
    nib.save(ni_lab, 'templ_lab.nii')
    nib.save(ni_mask, 'templ_mask.nii')
    