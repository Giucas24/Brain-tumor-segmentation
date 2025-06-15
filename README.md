MATLAB script that performs segmentation of brain tumor images acquired by Magnetic Resonance. 
The algorithm has been developed for the Image Processing course on A.A 2024 - 2025 at Politecnico di Bari. 
The specific work is my personal implementation of an already existing solution proposed by Rasha Khilkhal and Mustafa Ismael in their work named "Brain Tumor Segmentation Utilizing Thresholding and K-Means Clustering".
This work has been develeped and tested on a BRATS dataset which contains NIfTI files representing the 3D volume of acquisitions.
In the datased used there were 4 sequences of MRI acquired: FLAIR, T1, T1-gd and T2. This implementation is thought of being used just with FLAIR sequence. Other sequences may not perform correctly.

The segmentation process is divided in 4 parts:
  1. Preprocessing: application of a Median Filter with a window size of 5.
  2. K-means Clustering: segmentation of the FLAIR image with k = 3.
  3. Thresholding: extraction of the cluster with the brightest centroid (good extraction thanks to the high contrast provided by the FLAIR sequence).
  4. Morphological Operations: sequence of opening and closing operations (those MO may vary based on the size of the tumors). 3 different Structuring Elements have been tested.
    4.1 The result of the MO of each of the 3 SE have been further cleaned using the bwconncomp function in order to count the pixel inside each connected component and retain only the one with more than 200 px (this value may also be refined as well as the sequence of MO).
  5. Test: Jaccard index, Dice Coefficient and F1-score computed to understand how the segmented region is similar to the ground truth provided in the dataset.
