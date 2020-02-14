###########################################################################
#
#  Requirements:
# The fixed image is specified in the training_images lists
# Corresponding to each fixed image, all the images will be registered in
# the folder multi in the current directory in path ./multi/fixed_image_no
#
# FILES REQUIRED : training folder and testing folder in the current directory
#                  along with affine and b-spline text files
###########################################################################

import os

training_images = ["00", "01", "02", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "17", "36"]
## CHANGE THIS PATH
your_dir = "C:\\Users\\hp\\Desktop\\registered\\"
for i in training_images:

    rootdir = your_dir + "training\\training-images"
    base = your_dir
    basedir = your_dir + "multi\\10" + str(i) + "\\registered_images"
    affine = your_dir + "par0000affine.txt"
    bspline = your_dir + "\\par0000bspline.txt"

    fixed_reference_image = str(10) + str(i)

    command = "mkdir " + basedir
    os.system(command)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print(file)
            fullpath = os.path.join(subdir, file)
            fname = file.split(".")
            if (fname[0] != str(fixed_reference_image)):
                print(fname[0])
                result_dir = basedir + "\\" + fname[0]
                command = "mkdir " + result_dir
                os.system(command)

                fixed = rootdir + "\\" + str(fixed_reference_image) + ".nii"
                fixed_mask = base + "training" + "\\training-mask" + "\\" + str(fixed_reference_image) + "_1C.nii"
                moving = rootdir + "\\" + fname[0] + ".nii"
                moving_mask = base + "training" + "\\training-mask" + "\\" + fname[0] + "_1C.nii"
                command = "elastix -f " + fixed + " -fMask " + fixed_mask + " -m " + moving + " -mMask " + moving_mask + " -p " + affine + " -p " + bspline + " -out " + result_dir
                os.system(command)

    import os

    rootdir = "C:\\Users\\hp\\Desktop\\registered\\multi\\10" + str(i) + "\\registered_images"

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            fullpath = os.path.join(subdir, file)
            if (file == "TransformParameters.0.txt" or file == "TransformParameters.1.txt"):
                with open(fullpath, 'r+') as f:
                    content = f.read()
                    f.seek(0)
                    f.truncate()
                    f.write(content.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0'))

    import os

    rootdir = your_dir + "training\\training-labels"
    registered_img = your_dir + "multi\\10" + str(i) + "\\registered_images"
    basedir = your_dir + "multi\\10" + str(i) + "\\registered_labels"
    transform = "TransformParameters.1.txt"

    fixed_reference_image = str(10) + str(i)

    command = "mkdir " + basedir
    os.system(command)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print(file)
            fullpath = os.path.join(subdir, file)
            fname = file.split(".")
            if (fname[0] != (str(fixed_reference_image) + "_3C")):
                print(fname[0])
                ref_img_dir = fname[0].split("_")
                result_dir = basedir + "\\" + ref_img_dir[0]
                command = "mkdir " + result_dir
                os.system(command)

                transform_dir = registered_img + "\\" + ref_img_dir[0] + "\\" + transform
                label = rootdir + "\\" + str(fname[0]) + ".nii"
                command = "transformix -in " + label + " -out " + result_dir + " -tp " + transform_dir
                os.system(command)


