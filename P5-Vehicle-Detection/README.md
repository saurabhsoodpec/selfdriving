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
[image14]: ./output_images/false-positive-detection.png
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

**Spatial binning**
For spacial binning I used the small value 16,16 to have small size evaluations.

**Color histogram**
Refer to the attached jupyter notebook "CarND-Vehicle-Detection.ipynb" for the code. I plotted individual car and non-car image and converted it to various color channels - RGB, HSV, YCrCb and YUV. I noticed that S channel of HSV and multiple channels of YCrCb are displaying good results.

![alt text][image13]

Also, the image of the car shows better results - 
![alt text][image11]
![alt text][image12]

After training the model with both "HSV" and "YCrCb" I found that the accuracy of "YCrCb" is better and hence I used "YCrCb" in my final training model.

**Histogram of oriented gradients**
I experimented with various values of hog parameters, and here are the details - 

1. hog_orientations=8, hog_pix_per_cell=8, hog_cell_per_block=2
2. hog_orientations=9, hog_pix_per_cell=8, hog_cell_per_block=2
3. hog_orientations=8, hog_pix_per_cell=16, hog_cell_per_block=2
4. hog_orientations=8, hog_pix_per_cell=16, hog_cell_per_block=1
5. hog_orientations=9, hog_pix_per_cell=16, hog_cell_per_block=2
6. hog_orientations=9, hog_pix_per_cell=16, hog_cell_per_block=1
7. hog_orientations=9, hog_pix_per_cell=8, hog_cell_per_block=1

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

Code - 
1. windows = get_windows(img, x_range=(None, None), y_range=(400, 500), window_size=(96, 96), overlap=(0.75, 0.75))
2. windows += get_windows(img, x_range=(None, None), y_range=(400, 500), window_size=(144, 144), overlap=(0.75, 0.75))
3. windows += get_windows(img, x_range=(None, None), y_range=(430, 550), window_size=(192, 192), overlap=(0.75, 0.75))
4. windows += get_windows(img, x_range=(None, None), y_range=(460, 580), window_size=(192, 192), overlap=(0.75, 0.75))

![Sliding Windows][image10]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I extracted the features from the image and then normalized the data. Here is the normalized image. Then I trained the model and tested it with test images.

![alt text][image2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Here are the approaches applied to detect and remove false positives - 

1. Heatmap creation - This allows to create a heatmap of all the sliding window detection and this highlights the areas where a detection was found by multiple sliding windows. If there is a detection by one sliding window but not its corresponding windows then it can be a false positive.
2. Motion averaging - This keeps track of the detection in the previous image and then creates a averaged map of the current frame. If there is a detection in only one frame but not in its next frame then it will be a false positive. 

To understand the complete approach please refer to 'execute' and 'blur_boxes' method of 'CarND-Vehicle-Detection.ipynb'

### Here are six frames and their corresponding final heatmaps:

1. Original Image
2. All Raw Windows detected
3. Final heatmap after applying weight heatmap and and motion averaging (using blur boxes)  

![alt text][image14]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The HOG and SVM model used during this project seems to be limited in generalizing the results. I think there will be many placed where which model will also. Also there are some false positives even in this video itself.
2. Hard coded "region of interest" will not be applicable for all scenarios. Even within the same video each frame can have its own region of interest.

**Suggestions for improvement**
1. Using diversed dataset will make sure the model is generalized for various scenarios. Also we can experiment with other model apart from SVM to check if others perform better in certain situations.
2. Adding a neural network model layer to learn more aspects of the frame and not just focus on fixed training information.
3. Smart detection of the region of interest. Currently the region of interest if fixed and is only applicable for this video. The region of interest should smart and should be able to find correct values specific to each frame and each condition.
4. Template matching based on pre detected cars can be one approach. This should be adaptive i.e. when a car is detected then it should be template matched for only few frames (as car orientation remains the same) and then after few frames new car features should be saved for future detection.  
