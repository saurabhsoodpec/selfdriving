import numpy as np
import cv2
import pickle
import math
import time
from datetime import datetime
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the camera calibration matrix from the pickle file.
def get_camera_calibration():
    calibration_pickle = pickle.load( open( "./camera_cal/camera-calibration.p", "rb" ) )

    mtx = calibration_pickle["mtx"]
    dist = calibration_pickle["dist"]
    return mtx, dist


def plot_original_and_newImage(original, new, new_title="New Image", destCmap='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original, destCmap)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(new, destCmap)
    ax2.set_title(new_title, fontsize=30)

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def pre_process_image(img):
    ksize = 15
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', thresh=(5, 100))
    grady = abs_sobel_thresh(img, orient='y', thresh=(5, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(10, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined_binary = np.zeros_like(dir_binary)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    s_binary = hls_select(img, thresh=(128, 255))

    full_combined_binary = np.zeros_like(s_binary)
    # full_combined_binary[((((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) & (s_binary == 1))] = 1

    full_combined_binary[((combined_binary == 1) & (s_binary == 1))] = 1
    return full_combined_binary


# Image dimentions (720, 1280, 3)
def draw_poly(img):
    imshape = img.shape
    yMax = (imshape)[0]
    color = [0, 255, 0]
    new_image = img.copy()
    # cv2.line(new_image, (180, yMax), (580, 450), color, 10)
    # cv2.line(new_image, (580, 450), (710, 450), color, 10)
    # cv2.line(new_image, (710, 450), (1150, yMax), color, 10)

    # cv2.line(new_image, (300, 661), (560, 476), color, 10)
    # cv2.line(new_image, (560, 476), (734, 476), color, 10)
    # cv2.line(new_image, (734, 476), (1010, 656), color, 10)

    cv2.line(new_image, (300, 661), (560, 476), color, 10)
    cv2.line(new_image, (560, 476), (754, 476), color, 10)
    cv2.line(new_image, (754, 476), (1100, 656), color, 10)
    return new_image


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def region_of_not_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image "NOT" defined by the polygon
    formed from `vertices`. Image inside the Polygon is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask2 = cv2.bitwise_not(mask)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask2)
    return masked_image


def apply_perspective(img, grayscale='false'):
    if grayscale == 'true':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_size = (img.shape[1], img.shape[0])
    yMax = (img_size)[0]

    """
    #This time we are defining a four sided polygon to mask
    imshape = image.shape

    # Adjusting the co-ordinates according to the resolution of the image
    x0Vertice1 = math.ceil((120/960)*imshape[1])
    x1Vertice1 = math.ceil((445/960)*imshape[1])
    x2Vertice1 = math.ceil((520/960)*imshape[1])
    yVertice1 = math.ceil((325/540)*imshape[0])

    #vertices1 = np.array([[(x0Vertice1,imshape[0]),(x1Vertice1, yVertice1), (x2Vertice1, yVertice1), (imshape[1],imshape[0])]], dtype=np.int32)
    #src =  np.float32([[(x0Vertice1,imshape[0]),(x1Vertice1, yVertice1), (x2Vertice1, yVertice1), (imshape[1],imshape[0])]])

    """
    """
    src = np.float32([[300, 661], [560, 476], [734, 476], [1010, 656]])
    dst = np.float32([[offset, yMax],
                      [offset, 0],
                      [offset+400, 0],
                      [offset+400, yMax]])
    """

    src = np.float32([[300, 661], [560, 476], [734, 476], [1010, 656]])

    # src = np.float32([[300, 1000], [360, 476], [534, 476], [1010, 1000]])
    # src = np.float32([[100, 1000], [160, 476], [734, 476], [1010, 1000]])

    offset = 300
    dst = np.float32([[offset, 720],
                      [offset, 0],
                      [offset + 400, 0],
                      [offset + 400, 720]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # M = cv2.getPerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    # delete the next two lines
    # M = None
    # warped = np.copy(img)
    return warped, M, Minv

def get_real_world_angles_from_points(binary_warped, leftx, lefty, rightx, righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)
    #print("leftx= ", leftx.shape)
    #print("ploty= ", ploty.shape)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad


def fit_poly(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    arg_max_left = np.argmax(histogram[:midpoint])
    arg_max_right = np.argmax(histogram[midpoint:])

    if arg_max_left <= 40 or arg_max_right <= 40:
        print("arg_max_left=", arg_max_left, "arg_max_right=", arg_max_right)
        return

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # print("leftx_base=", leftx_base, ", rightx_base=",rightx_base)

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_curverad, right_curverad = get_real_world_angles_from_points(binary_warped, leftx, lefty, rightx, righty)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return (left_curverad, right_curverad, left_fit, right_fit, out_img, left_lane_inds, right_lane_inds)


def unwarp_image(img, warped_binary, Minv, left_fit, right_fit):

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result