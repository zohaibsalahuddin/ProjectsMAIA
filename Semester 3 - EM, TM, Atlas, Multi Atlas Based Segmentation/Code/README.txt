SEGMENTATION GUIDELINES:

- Altas Segmentation experiments.ipynb : Standalone file, just upload on google colab and run it. No data download required


REGISTRATION GUIDELINES:
The three python scripts in the elastix scripts folders are used for 
registering and transforming as well as registering and transforming
multi atlas for average atlas

The notebook atlas segmentation experiment's first part is atlas generation.
The average atlas notebook generates the average atlas. average_atlas.ipynb is a 
standalone file : Upload on google drive and just run it.


NOTE: The notebook has been pre-run so that the output is already there for visualization


EXTRA FILES
We include two python files in the "extra" folder:

- getTM: upon executing it..
	> it creates a Tissue Model using the training set images
	> it returns the dice scores for all tissues for all the images in the test set

-labelProp:
	> Reference image (atlas) is registered to each test image
	> The labels are then transformed to match the target
	> Dice scores for all tissues for all the images in the test set are computed