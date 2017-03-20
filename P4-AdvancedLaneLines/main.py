#importing some useful packages
import numpy as np
import cv2
import pickle
import math
import time
from datetime import datetime
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from Line import *
from lane_util import *

left_Lane_Line = Line()
right_Lane_Line = Line()

def get_left_line():
    return left_Lane_Line

def get_right_line():
    return right_Lane_Line

def process_clip(clip_image):
    """
    currDT = datetime.now().strftime('%Y-%m-%d-%H-%M-%s')
    fileName = './out_images/pic_' + currDT + '_in.png'
    im = Image.fromarray(clip_image)
    im.save(fileName)
    """

    mtx, dist = get_camera_calibration()
    undist_binary = cv2.undistort(clip_image, mtx, dist, None, mtx)

    s_binary = hls_select(undist_binary, thresh=(128, 255))

    final_binary = s_binary  # pre_process_image(undist_binary)

    # image_with_poly = draw_poly(clip_image)

    # This time we are defining a four sided polygon to mask
    imshape = clip_image.shape

    # Adjusting the co-ordinates according to the resolution of the image
    x0Vertice1 = math.ceil((120 / 960) * imshape[1])
    x1Vertice1 = math.ceil((445 / 960) * imshape[1])
    x2Vertice1 = math.ceil((520 / 960) * imshape[1])
    yVertice1 = math.ceil((325 / 540) * imshape[0])

    vertices1 = np.array(
        [[(x0Vertice1, imshape[0]), (x1Vertice1, yVertice1), (x2Vertice1, yVertice1), (imshape[1], imshape[0])]],
        dtype=np.int32)
    masked_image = region_of_interest(final_binary, vertices1)

    x0Vertice2 = math.ceil((250 / 960) * imshape[1])
    x1Vertice2 = math.ceil((475 / 960) * imshape[1])
    x2Vertice2 = math.ceil((490 / 960) * imshape[1])
    yVertice2 = math.ceil((380 / 540) * imshape[0])
    xnVertice2 = math.ceil((140 / 960) * imshape[1])

    vertices2 = np.array([[(x0Vertice2, imshape[0]), (x1Vertice2, yVertice2), (x2Vertice2, yVertice2),
                           (imshape[1] - xnVertice2, imshape[0])]], dtype=np.int32)
    masked_image = region_of_not_interest(masked_image, vertices2)

    binary_warped, M, Minv = apply_perspective(masked_image)
    return_poly = fit_poly(binary_warped)

    right_line = get_right_line()
    left_line = get_left_line()

    if return_poly == None:
        print("No Object Returned")
        right_curverad = None
        left_fit = None
        right_fit = None
        out_img = None
        left_lane_inds = None
        right_lane_inds = None

        #plot_original_and_newImage(image, binary_warped, 'Perspective Warped')
        #histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        #plt.plot(histogram)

        #TODO:: Add a condition if the first image is not detected properly
        #if(len(right_line.current_fit) == 0 or len(left_line.current_fit)==0):
        #    return clip_image
    else:
        left_curverad = return_poly[0]
        right_curverad = return_poly[1]
        left_fit = return_poly[2]
        right_fit = return_poly[3]
        out_img = return_poly[4]
        left_lane_inds = return_poly[5]
        right_lane_inds = return_poly[6]

        #print("Curve Coff Intercept Diff= ", (right_fit[2] - left_fit[2]))
        intercept_diff = right_fit[2] - left_fit[2]
        if intercept_diff >350 and intercept_diff<450:
            right_line.current_fit = right_fit
            left_line.current_fit = left_fit
            left_line.radius_of_curvature=left_curverad
            right_line.radius_of_curvature=right_curverad
        else:
            print("Lines far apart or too close. Intercept diff=", intercept_diff)

    unwarpped_image = unwarp_image(clip_image, binary_warped, Minv, left_line.current_fit, right_line.current_fit)

    text_position = (math.ceil(imshape[1] / 2) - 100, 100)
    text = "Curves=" + str(left_line.radius_of_curvature)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(unwarpped_image, text, text_position, font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    """
    outFileName = './out_images/pic_' + currDT + '_out.png'
    im = Image.fromarray(unwarpped_image)
    im.save(outFileName)
    """

    return unwarpped_image

def process_video():

    # print("White 1")
    last_saved_clip = None
    project_video_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_clip)  # NOTE: this function expects color images!!
    white_clip.write_videofile(project_video_output, audio=False)


if __name__ == '__main__':
    process_video()