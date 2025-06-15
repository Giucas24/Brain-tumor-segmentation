clear
close all
clc
%% INSTRUCTION TO TEST THE PROGRAM
% This program is configured to open the NIfTI file and the corresponding
% ground truth using only the img_index.

% To avoid malfunctioning, it is therefore recommended
% to set up the files as follows:
%
% 1. All NIfTI files to be used with the segmentation algorithm must
% be placed in the Current Folder and must be named BRATS_XXX.nii
% with XXX being a number from 001 to 484.
%
% 2. The Current Folder must also contain a folder named "labels training"
% (the exact name is important) which contains the ground truth files
% (also named BRATS_XXX.nii).

%% VISUALIZATION OF THE IMAGES
% Images used to evaluate the results: 82, 151, 20, 34, 117, 170, 402, 48, 379, 474

% Initialization of the parameters
img_index = 66;  % Choose an image in the range 1 - 484

% Check on availabilty of the img_index
if (img_index > 0 && img_index <= 484)
    img_filename = sprintf('BRATS_%03d.nii', img_index);
else                                                      
    error('Choose an image in the range 1 - 484.');      

end


% Opening of the selected NIfTI image
img_nifti = niftiread(img_filename);
fprintf('-------OPENING BRATS_%03d.nii--------\n' , img_index);

% Visualization of the selected NIfTI file dimensions
dim = size(img_nifti);
fprintf('NIfTI file dimensions: %d x %d x %d x %d\n', dim(1), dim(2), dim(3), dim(4));
fprintf('Actual dimensions of the image: %d x %d\n', dim(1), dim(2));
fprintf('Available axial slices: %d\n', dim(3));
fprintf('Available MRI sequences: %d\n 1: FLAIR \n 2: T1 \n 3: T1-Gd \n 4: T2\n\n', dim(4));


% Selection of the slice
slice_index = 78;

% Check on availabilty of the slice_index
if slice_index < 1 | slice_index > 155
    error('Choose a slice in the range 1–155. A middle value (e.g., 78) is recommended to ensure full brain tissue coverage.')
end


% Visualization of the 4 sequences
seq_names = {'FLAIR', 'T1', 'T1-Gd', 'T2'};
figure;
sgtitle(sprintf('Slice %d of BRATS\\_%03d', slice_index, img_index));

for seq= 1:4
    img_slice_sequence = img_nifti(:,:, slice_index, seq);

    subplot(2, 2, seq);
    imshow(img_slice_sequence, []);
    title(seq_names{seq});
end

pause;
%% APPLICATION OF MEDIAN FILTER
% Selection of the sequence to segment
seq = 1;

% Check on availabilty of the seq
if slice_index < 1 && slice_index > 4
    error('Choose a sequence in the range 1–4.')
end


% Image selection based on specified parameters
img_selected = img_nifti(:,:, slice_index, seq);
img_norm = mat2gray(img_selected);

figure;
sgtitle(sprintf('Application of median filter on the original %s image', seq_names{seq}));
subplot(1,2,1);
imshow(img_norm);
title(sprintf('Original %s image', seq_names{seq}));

% Median filter
window_size = [5 5];    % Value taken by the reference article
img_median = medfilt2(img_norm, window_size);

subplot(1, 2, 2);
imshow(img_median);
title('Median filtered image');
pause;
%% K-MEANS CLUSTERING  WITH K = 3
% Number of desired clusters
k = 3;  % Value taken by the reference article

% Unroll img_median as a vector 1x1
img_unrolled = img_median(:);

% K-means clustering
[kmeans_index, C] = kmeans(img_unrolled, k);

% Get back the original img_median shape
img_segmented = reshape(kmeans_index, [dim(1) dim(2)]);

% Sort the centroids from darkest to brightest
[sorted_C, sorted_idx] = sort(C);

% Assign a label to each cluster for the colorbar
labels = strings(1, k);
labels(sorted_idx(1)) = "Background";
labels(sorted_idx(2)) = "Healty tissue";
labels(sorted_idx(3)) = "Tumor region";

figure;
sgtitle('K-means clustering, segmented tumor mask and region');
subplot(1, 3, 1);
imagesc(img_segmented);
title(sprintf('n. of cluster: k = %d', k));
axis image off; 

colormap(gray(k));
colorbarTicks = 1:k;
cb = colorbar('Location', 'southoutside');
cb.Ticks = colorbarTicks;
cb.TickLabels = labels; 

% Find the brightest centroid (the whole tumor region)
[~, tumor_cluster] = max(C);   % ~ ignores the first output of max which is the value of the brightest centroid.

% Creation of the binary mask
tumor_mask = img_segmented == tumor_cluster;

subplot(1, 3, 2);
imshow(tumor_mask);
title('Segmented tumor mask');


% Overlay of segmented tumor region over the median filtered image
overlay = labeloverlay(img_median, tumor_mask, 'Colormap', [1 0 0], 'Transparency', 0.7);

subplot(1, 3, 3);
imshow(overlay);
title('Segmented tumor region')
pause;

%% MORPHOLOGICAL OPERATIONS WITH DISK STRUCTURING ELEMENT: RADIUS = 5
% Disk Structuring Element with radius = 5
disk_se = strel('disk', 5);

% Opening Disk SE
tumor_open_disk = imopen(tumor_mask, disk_se);
figure;
sgtitle('Morphological Operations with Disk structuring element: r = 5')
subplot(1, 2, 1);
imshow(tumor_open_disk);
title('Opening');

% Closing Disk SE
tumor_close_disk = imclose(tumor_open_disk, disk_se);
subplot(1, 2, 2);
imshow(tumor_close_disk);
title('Closing');
pause;

%% MORPHOLOGICAL OPERATIONS WITH SQUARE 3X3 STRUCTURING ELEMENT
% Square 3x3 Structuring Element
square_se = strel('square', 3);

% Opening Square SE
tumor_open_square = imopen(tumor_mask, square_se);
figure;
sgtitle('Morphological Operations with Square 3x3 structuring element')
subplot(1, 2, 1);
imshow(tumor_open_square);
title('Opening');

% Closing Square SE
tumor_close_square = imclose(tumor_open_square, square_se);
subplot(1, 2, 2);
imshow(tumor_close_square);
title('Closing');
pause;

%% MORPHOLOGICAL OPERATIONS WITH RECTANGULAR 3X9 STRUCTURING ELEMENT
% Rectangular 3x9 Structuring Element
rect_se = strel('rectangle', [3 9]);

% Opening Square SE
tumor_open_rect = imopen(tumor_mask, rect_se);
figure;
sgtitle('Morphological Operations with Rectangular 3x9 structuring element')
subplot(1, 2, 1);
imshow(tumor_open_rect);
title('Opening');

% Closing Square SE
tumor_close_rect = imclose(tumor_open_rect, rect_se);
subplot(1, 2, 2);
imshow(tumor_close_rect);
title('Closing');
pause;

%% EXTRACT THE CORRECT REGION(S)
% Find the connected components on each of the three results
CC_disk = bwconncomp(tumor_close_disk);
CC_square = bwconncomp(tumor_close_square);
CC_rect = bwconncomp(tumor_close_rect);

% Extract the number of pixels for each of the detected components
numPixels_disk = cellfun(@numel, CC_disk.PixelIdxList);
numPixels_square = cellfun(@numel, CC_square.PixelIdxList);
numPixels_rect = cellfun(@numel, CC_rect.PixelIdxList);

% Extract the components with more than 200 pixels on the disk result
keep = numPixels_disk >= 200;
idxToKeep = CC_disk.PixelIdxList(keep);
allIdx = cell2mat(idxToKeep(:));
B_mask_disk = false(size(tumor_close_disk));
B_mask_disk(allIdx) = true;

% Extract the components with more than 200 pixels on the square result
keep = numPixels_square >= 200;
idxToKeep = CC_square.PixelIdxList(keep);
allIdx = cell2mat(idxToKeep(:));
B_mask_square = false(size(tumor_close_square));
B_mask_square(allIdx) = true;

% Extract the components with more than 200 pixels on the rectangle result
keep = numPixels_rect >= 200;
idxToKeep = CC_rect.PixelIdxList(keep);
allIdx = cell2mat(idxToKeep(:));
B_mask_rect = false(size(tumor_close_rect));
B_mask_rect(allIdx) = true;

% The results of those operations will be shown starting from line 294 in
% order to be compared with the ground truth

%% LABEL (GROUND TRUTH)
% Opening the label file corresponding to the current NIfTI image
gt_filename = sprintf('labels training/BRATS_%03d.nii', img_index);
gt_nifti = niftiread(gt_filename);

% Visualization of the selected label dimensions
dim_gt = size(gt_nifti);
fprintf('Ground Truth dimensions: %d x %d x %d\n', dim_gt(1), dim_gt(2), dim_gt(3));
fprintf('Actual dimensions of the label image: %d x %d\n', dim_gt(1), dim_gt(2));
fprintf('Available axial slices: %d\n\n', dim_gt(3));

% Matching the slice used in the original image
gt_selected = gt_nifti(:, :, slice_index);

% Visualization of the three regions of the tumor
gt_whole = label2rgb(gt_selected, 'sky', 'w');


regions = [1, 2, 3];
label = strings(1, 3);
% Assign a label to each cluster for the legend (colorbar)
label(regions(1)) = "Edema";
label(regions(2)) = "Necrotic core";
label(regions(3)) = "Enhancing Tumor";

% Division done to be included in the written report
figure;
imagesc(gt_whole);
title('Ground Truth divided in its three regions')
axis image off
clim([0.5, 3.5]);

colormap(sky(3));
colorbarTicks = 1:3;
cb = colorbar('Location', 'southoutside');
cb.Ticks = colorbarTicks;
cb.TickLabels = label; 
cb.TickLabelInterpreter = 'none';
cb.Ruler.TickLabelRotation = 0;
pause;

% Creation of the label's binary mask containing the whole tumor region
gt_binary_mask_complete = gt_selected > 0;

figure;
sgtitle('Comparison between Ground Truth and the obtained results')
subplot(2, 2, 1);
imshow(gt_binary_mask_complete);
title('Ground Truth')

subplot(2, 2, 2);
imshow(B_mask_disk);
title('Disk r = 5 SE');

subplot(2, 2, 3);
imshow(B_mask_square);
title('Square 3x3 SE');

subplot(2, 2, 4);
imshow(B_mask_rect);
title('Rectangle 3x9 SE');
pause;
%% DISK SE METRICES
fprintf('----------DISK SE METRICES----------\n');

% Jaccard index for disk SE
jaccard_index_disk = jaccard(gt_binary_mask_complete, B_mask_disk);
fprintf('Jaccard Index: %.4f\n', jaccard_index_disk);

% Dice coefficient for disk SE
dice_disk = dice(gt_binary_mask_complete, B_mask_disk);
fprintf('Dice coefficient: %.4f\n', dice_disk);

% True Positives (TP)
TP = sum(B_mask_disk(:) & gt_binary_mask_complete(:));

% False Positives (FP)
FP = sum(B_mask_disk(:) & ~gt_binary_mask_complete(:));

% False Negatives (FN)
FN = sum(~B_mask_disk(:) & gt_binary_mask_complete(:));

% Precision
precision = TP / (TP + FP + eps);
% Recall
recall = TP / (TP + FN + eps);

% F1-score disk SE
f1_score = (2 * precision * recall) / (recall + precision + eps);

fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1_score);
fprintf('------------------------------------\n\n');
%% SQUARE SE METRICES
fprintf('---------SQUARE SE METRICES---------\n');

% Jaccard index for square SE
jaccard_index_square = jaccard(gt_binary_mask_complete, B_mask_square);
fprintf('Jaccard Index: %.4f\n', jaccard_index_square);

% Dice coefficient for square SE
dice_square = dice(gt_binary_mask_complete, B_mask_square);
fprintf('Dice coefficient: %.4f\n', dice_square);

% True Positives (TP)
TP = sum(B_mask_square(:) & gt_binary_mask_complete(:));

% False Positives (FP)
FP = sum(B_mask_square(:) & ~gt_binary_mask_complete(:));

% False Negatives (FN)
FN = sum(~B_mask_square(:) & gt_binary_mask_complete(:));

% Precision
precision = TP / (TP + FP + eps);
% Recall
recall = TP / (TP + FN + eps);

% F1-score disk SE
f1_score = (2 * precision * recall) / (recall + precision + eps);

fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1_score);
fprintf('------------------------------------\n\n');

%% RECTANGULAR SE METRICES
fprintf('-------RECTANGULAR SE METRICES------\n');

% Jaccard distance for rectangle SE
jaccard_index_rect = jaccard(gt_binary_mask_complete, B_mask_rect);
fprintf('Jaccard Index: %.4f\n', jaccard_index_rect);

% Dice coefficient for rectangle SE
dice_rect = dice(gt_binary_mask_complete, B_mask_rect);
fprintf('Dice coefficient: %.4f\n', dice_rect);

% True Positives (TP)
TP = sum(B_mask_rect(:) & gt_binary_mask_complete(:));

% False Positives (FP)
FP = sum(B_mask_rect(:) & ~gt_binary_mask_complete(:));

% False Negatives (FN)
FN = sum(~B_mask_rect(:) & gt_binary_mask_complete(:));

% Precision
precision = TP / (TP + FP + eps);
% Recall
recall = TP / (TP + FN + eps);

% F1-score disk SE
f1_score = (2 * precision * recall) / (recall + precision + eps);

fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1_score);
fprintf('------------------------------------\n\n');

%% VISUALIZATION OF THE RESULTS
% Dice Coefficient values for the 10 random chosen images
dice_values= [0.8840 0.8850 0.8833;
              0.8878 0.8997 0.8796;
              0.9021 0.8063 0.8957;
              0.7660 0.8349 0.7886;
              0.8424 0.8602 0.8504;
              0.0947 0.0827 0.0869;
              0.7074 0.7268 0.7114;
              0.8799 0.8613 0.8912;
              0.7759 0.9225 0.8314;
              0.9232 0.9194 0.9179
              ];
% Image indices: 82, 151, 20, 34, 117, 170, 402, 48, 379, 474
figure;
boxplot(dice_values, 'Labels', {'Disk', 'Square', 'Rectangle'});
title('Performance of the three structuring elements');
ylabel('Dice');
xlabel('SE');



