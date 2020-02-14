############################################################
# This function registers the multi atlas to common space.
# You should have msi/registered_images in the current directory
#
# After this, u need to register the average mean image
# and average atlas to the all the testing images like normal
# procedure.
#
############################################################

import os
your_dir = "C:\\Users\\hp\\Desktop\\registered\\"
rootdir = your_dir + "msi\\registered_images"

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

tdir = your_dir + "msi\\registered_images\\"
atlasdir = your_dir + "average_atlas\\"
result_dir = your_dir + "avg_atlas_transform"
transform = "TransformParameters.1.txt"



t_dir = os.listdir(tdir)
t_dir.sort()
print(t_dir)

a_dir = os.listdir(atlasdir)
a_dir.sort()
print(a_dir)

for t_file in t_dir:
        transform_file = os.path.join(tdir, t_file,transform)
        print(transform_file)
        csf_file = os.path.join(atlasdir, t_file,"CSF.nii")
        wm_file = os.path.join(atlasdir, t_file,"WM.nii")
        gm_file = os.path.join(atlasdir, t_file,"GM.nii")
        print(csf_file)
        print(wm_file)
        print(gm_file)
        result_file_csf = os.path.join(result_dir, t_file,"csf")
        command = "mkdir " + result_file_csf
        os.system(command)

        result_file_gm = os.path.join(result_dir, t_file,"wm")
        command = "mkdir " + result_file_gm
        os.system(command)

        result_file_wm = os.path.join(result_dir, t_file,"gm")
        command = "mkdir " + result_file_wm
        os.system(command)

        command = "transformix -in " + csf_file + " -out " + result_file_csf + " -tp " + transform_file
        os.system(command)
        command = "transformix -in " + wm_file + " -out " + result_file_gm + " -tp " + transform_file
        os.system(command)
        command = "transformix -in " + gm_file + " -out " + result_file_wm + " -tp " + transform_file
        os.system(command)
        print("#########################")
















