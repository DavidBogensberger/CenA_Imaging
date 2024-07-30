# Cen A Imaging
This code is for analyzing Chandra subpixel images of Centaurus A, to look for variation and proper motions. All these files can be adapted for use with other observations, other objects, and other data sets. 

### Selectimageregion_createsubpixelimages.py
This code is used to create the subpixel images from the different observations, which are used later in the rest of the code. 

### CompressFitsFileData.py
This compresses the subpixel images to reduce the file sizes while retaining all the information needed for the later fits. The later codes read in the compressed images produced by this script. 

### Align_xcorr
The three align_xcorr codes are used to align different observations. The Align_xcorr.py is the basic version of this, which cross-correlates a grid of regions around the aligning sources against a similar grid of one particular observation that is selected for this task, due to having the longest exposure time. 

Align_xcorr_vsallimg.py expands on this, by performing the cross-correlation of each grid of regions for each observation against the merged grid of all other observations. It iteratively improves on the previous alignments, which results in more accurate cross-correlations. 

Align_xcorr_vsallimg_smth.py does the same thing as Align_xcorr_vsallimg.py, but smoothes the images first, before performing the cross-correlation. 

### Propermotion.py
This is the code that runs the proper motion algorithm on the set of images, with alignments as found by one of the align_xcorr codes. It outputs a fits file containing the results of the iterative proper motion fits. 

### Bestfitpropermotions.py
This code extracts the relevant data from the fits file outputted by Propermotion.py. It corrects these for possible offsets, and adjusts the uncertainties in the measurements to accommodate the uncertainty of the alignment. 
