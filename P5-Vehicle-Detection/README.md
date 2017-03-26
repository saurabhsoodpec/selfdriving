**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/normalization.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/hog-grey.png
[image5]: ./output_images/heatmap-1.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/hog_output_orient_8.png
[image9]: ./output_images/hog_output_orient_9.png
[image10]: ./output_images/sliding_windows.png
[image11]: ./output_images/HSV-S-Channel.png
[image12]: ./output_images/YCrCb-Y-Channel.png
[image13]: ./output_images/color-space.png
[video1]: ./project_video_out.mp4

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the method named `_get_hog_features`  of the Jupyter notebook `CarND-Vehicle-Detection`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I experimented with various parameters available for tuning and improve image feature extraction. Here the the various feature used -

1. Spatial binning
2. Color histogram
3. Histogram of oriented gradients
	1. hog_orientations
	2. hog_pix_per_cell
	3. hog_cell_per_block

*Spatial binning*
For spacial binning I used the small value 16,16 to have small size evaluations.

*Color histogram*
Refer to the attached jupyter notebook "CarND-Vehicle-Detection.ipynb" for the code. I plotted individual car and non-car image and converted it to various color channels - RGB, HSV, YCrCb and YUV. I noticed that S channel of HSV and multiple channels of YCrCb are displaying good results.

![alt text][image13]

Also, the image of the car shows better results - 
![alt text][image11]
![alt text][image12]

I experimented with various values of hog parameters, and here are the details - 

1. hog_orientations=8, hog_pix_per_cell=8, hog_cell_per_block=2
2. hog_orientations=9, hog_pix_per_cell=8, hog_cell_per_block=2
3. hog_orientations=8, hog_pix_per_cell=16, hog_cell_per_block=2
4. hog_orientations=8, hog_pix_per_cell=16, hog_cell_per_block=1
5. hog_orientations=9, hog_pix_per_cell=16, hog_cell_per_block=2
6. hog_orientations=9, hog_pix_per_cell=16, hog_cell_per_block=1
7. hog_orientations=9, hog_pix_per_cell=8, hog_cell_per_block=1

*Results*
![alt text][image8]
![alt text][image9]

Based on the above results I decided to use the following values - 

 1. hog_orientations = 8
 2. hog_pix_per_cell = 8
 3. hog_cell_per_block = 2
 
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC`. This code is in the method named `train_classifier` in jupyter notebook .

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search contains 3 different scales and range of windows

windows = get_windows(img, x_range=(None, None), y_range=(400, 500), window_size=(96, 96), overlap=(0.75, 0.75))
windows += get_windows(img, x_range=(None, None), y_range=(400, 500), window_size=(144, 144), overlap=(0.75, 0.75))
windows += get_windows(img, x_range=(None, None), y_range=(430, 550), window_size=(192, 192), overlap=(0.75, 0.75))
windows += get_windows(img, x_range=(None, None), y_range=(460, 580), window_size=(192, 192), overlap=(0.75, 0.75))

![Sliding Windows][image10]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I extracted the features from the image and then normalized the data. Here is the normalized image. Then I trained the model and tested it with test images.

![alt text][image2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are many false positives detected in the images. I used some thresholding approaches to remove the false positives but still there is a scope of improvement in this area.  

