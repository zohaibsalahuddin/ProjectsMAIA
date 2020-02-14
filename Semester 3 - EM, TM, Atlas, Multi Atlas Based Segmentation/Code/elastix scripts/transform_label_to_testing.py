###########################################################################
#
#  Requirements:
# The fixed image name is defined as fixed_image_reference either mean image or
# the reference image used for creating the corresponding atlas
# THe current directory should contain:
# Training directory and testing directory as well as
# the atlas directory with CSF.nii, WM.nii and GM.nii and mean_image.nii
#
# All the registered test labels will be stored in the directory as
# predictions/CSF, predictions/WM and predictions/GM
#
#
###########################################################################
your_dir = "C:\\Users\\hp\\Desktop\\registered\\"
import os

rootdir = your_dir + "testing\\testing-images"
base = your_dir
basedir = your_dir + "registered_tests"
affine = base + "\\par0000affine.txt"
bspline = base + "\\par0000bspline.txt"

fixed_reference_image = 1007



command = "mkdir " + basedir
os.system(command)
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print(file)
        fullpath = os.path.join(subdir, file)
        fname = file.split(".")
        if (fname[0] != str(fixed_reference_image)):
            print(fname[0])
            result_dir = basedir + "\\" + fname[0]
            command= "mkdir " + result_dir
            os.system(command)

            moving = base + "training\\training-images\\" + str(fixed_reference_image) + ".nii"
            moving_mask = base + "training\\training-mask\\" + str(fixed_reference_image) + "_1C.nii"
            fixed = rootdir + "\\" + fname[0] + ".nii"
            fixed_mask = base + "testing" + "\\testing-mask" + "\\" + fname[0] + "_1C.nii"
            command = "elastix -f " + fixed + " -fMask " + fixed_mask + " -m " + moving + " -mMask " + moving_mask + " -p " +  affine + " -p " + bspline + " -out " + result_dir
            os.system(command)



import os
rootdir = your_dir + "registered_tests"

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        fullpath = os.path.join(subdir, file)
        if (file == "TransformParameters.0.txt" or file == "TransformParameters.1.txt"):
            with open(fullpath, 'r+') as f:
                content = f.read()
                f.seek(0)
                f.truncate()
                #f.write(content.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0'))
                f.write(content.replace('ResultImagePixelType \"short\"', 'ResultImagePixelType \"float\"'))

import os

labeldir = your_dir + "atlas\\"
registeredtestdir = your_dir + "registered_tests\\"
basedir = your_dir + "predictions\\CSF"
test_img = your_dir + "testing\\testing-images"
transform = "TransformParameters.1.txt"

fixed_reference_image = 1007

command = "mkdir " + basedir
os.system(command)

for subdir, dirs, files in os.walk(test_img):
    for file in files:
        #print(file)
        fullpath = os.path.join(subdir, file)
        fname = file.split(".")
        if (fname[0] != (str(fixed_reference_image)+"_3C")):
            print(fname[0])
            result_dir = basedir + "\\" + fname[0]
            command= "mkdir " + result_dir
            os.system(command)

            transform_dir = registeredtestdir + "\\" +fname[0] + "\\" + transform
            label = labeldir + "CSF" + ".nii"
            command = "transformix -in " + label + " -out " + result_dir + " -tp " + transform_dir
            os.system(command)


import os

labeldir = your_dir + "atlas\\"
registeredtestdir = your_dir + "registered_tests\\"
basedir = your_dir + "predictions\\GM"
test_img = your_dir + "testing\\testing-images"
transform = "TransformParameters.1.txt"

fixed_reference_image = 1007

command = "mkdir " + basedir
os.system(command)

for subdir, dirs, files in os.walk(test_img):
    for file in files:
        #print(file)
        fullpath = os.path.join(subdir, file)
        fname = file.split(".")
        if (fname[0] != (str(fixed_reference_image)+"_3C")):
            print(fname[0])
            result_dir = basedir + "\\" + fname[0]
            command= "mkdir " + result_dir
            os.system(command)

            transform_dir = registeredtestdir + "\\" +fname[0] + "\\" + transform
            label = labeldir + "GM" + ".nii"
            command = "transformix -in " + label + " -out " + result_dir + " -tp " + transform_dir
            os.system(command)


import os

labeldir = your_dir + "atlas\\"
registeredtestdir = your_dir + "registered_tests\\"
basedir = your_dir + "predictions\\WM"
test_img = your_dir + "testing\\testing-images"
transform = "TransformParameters.1.txt"

fixed_reference_image = 1007

command = "mkdir " + basedir
os.system(command)

for subdir, dirs, files in os.walk(test_img):
    for file in files:
        #print(file)
        fullpath = os.path.join(subdir, file)
        fname = file.split(".")
        if (fname[0] != (str(fixed_reference_image)+"_3C")):
            print(fname[0])
            result_dir = basedir + "\\" + fname[0]
            command= "mkdir " + result_dir
            os.system(command)

            transform_dir = registeredtestdir + "\\" +fname[0] + "\\" + transform
            label = labeldir + "WM" + ".nii"
            command = "transformix -in " + label + " -out " + result_dir + " -tp " + transform_dir
            os.system(command)