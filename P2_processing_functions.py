#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# import array
import glob
import sys
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from collections import deque


def cvt_bgr2rgb(_img_bgr):
    return cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2RGB)

def plot_2(_img_1, _img_2, heading_1="", heading_2=""):
    # plotting
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(_img_1, cmap='gray')
    ax1.set_title(heading_1, fontsize=50)
    ax2.imshow(_img_2, cmap='gray')
    ax2.set_title(heading_2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def draw_line(self, img, x1, y1, x2, y2, color=[255, 0, 0], thickness=3):
    return cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
# class to store the characteristics of each line detection
class line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.last_detected = False  
        self.cnt_last_invalid = 1000
        # number of stored lines detected n
        self.n = 10
        self.n_fit = 10
        self.n_curvature = 60
        # x values of the last n fits of the line
        self.recent_xfitted_buffer = deque()
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.recent_fit_buffer = deque()
        self.recent_fit = [] 
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.recent_radius_of_curvature_buffer = deque()
        self.recent_radius_of_curvature = []
        self.avg_radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        # color for visualization
        self.color = [255, 255, 255]
    


    def update_with_new_values(self, my_deque, new_values):
        my_deque.appendleft(new_values)
        while len(my_deque) > self.n:
            my_deque.pop()
        my_values = []
        for i in range(0, len(my_deque)):
            my_values.append(my_deque[i])
        return my_values
    
    def calc_average(self, last_n_values, avg_over_n):
        x_avg = []
        len_last_n = len(last_n_values)
        if len_last_n != 0:
            if any(isinstance(el, list) for el in last_n_values):
                for y in range(0,len(last_n_values[0])):
                    x_sum = 0
                    for n in range(0, min(len_last_n, avg_over_n)):
                        x_sum += last_n_values[n][y]
                    x_avg.append(x_sum/min(len_last_n, avg_over_n))
            else:
                x_sum = 0
                for n in range(0, min(len_last_n, avg_over_n)):
                    x_sum += last_n_values[n]
                x_avg = x_sum/min(len_last_n, avg_over_n)
        return x_avg

    def calc_offset(self):
        pass

    ## Determine the curvature of the lane_markings and vehicle position with respect to center.
    def calc_curvature_real(self, y_eval, fit_cr):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3/(63) # 63(53-69)pixel; meters per pixel in y dimension
        xm_per_pix = 3.7/611 # 611(605-620)pixel; meters per pixel in x dimension
        
        # Implement the calculation of R_curve (radius of curvature)
        curverad = (1+(2*fit_cr[0]*y_eval*ym_per_pix+fit_cr[1])**2)**(3/2)/(np.absolute(2*fit_cr[0]))  
        curverad = curverad * 0.5 # parameter tuning
    
        return curverad 

    def update(self, img_shape, line_valid, xfitted, fit, radius=0, xvalues=0, yvalues=0):
        # update values:
        if line_valid:
            self.cnt_last_invalid = 0
            self.recent_xfitted = self.update_with_new_values(self.recent_xfitted_buffer, xfitted)
            self.bestx = self.calc_average(self.recent_xfitted, self.n_fit)
            self.recent_fit = self.update_with_new_values(self.recent_fit_buffer, fit)
            self.best_fit = self.calc_average(self.recent_fit, self.n)
            # calculate metrics:
            curvature_radius = self.calc_curvature_real(img_shape[0], self.best_fit) 
            self.recent_radius_of_curvature = self.update_with_new_values(self.recent_radius_of_curvature_buffer, curvature_radius)
            self.avg_radius_of_curvature = self.calc_average(self.recent_radius_of_curvature, self.n_curvature)
            # offset = calc_offset()
        else:
            self.cnt_last_invalid += 1
        
        return self.avg_radius_of_curvature, self.bestx, self.best_fit

class lane_markings_detection:
    def __init__(self):
        self.print_output = False
        self.write_output = False
        self.left_fit = np.array([0.0, 0.0, 0.0])
        self.right_fit = np.array([0.0, 0.0, 0.0])
        self.line_left = line()
        self.line_right = line()
        # Initialize lines
        self.line_left.color = [255,0,0]
        self.line_right.color = [0,0,255]
        # Hyperparameters
        self.max_cnt_last_invalid = 5
        
    ## Compute Camera Calibration
    def compute_camera_calibration(self):
        # prepare object points
        # number of inside corners in x and y
        ny = 6 
        nx = 9

        # Make a list of calibration images (calibration1..20)
        images_cal = glob.glob('camera_cal/calibration*.jpg', recursive=False)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane_markings.

        # create objp
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # get chessboard corners for all calibration images
        for fname in images_cal:
            img_cal = plt.imread(fname)

            # Convert to grayscale
            gray_cal = cv2.cvtColor(img_cal, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_cal, (nx, ny), None)

            # If corners found, draw corners
            if ret == True:
                objpoints.append(objp)
        #         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)
            
                # draw & display corners
                cv2.drawChessboardCorners(img_cal, (nx, ny), corners, ret)
            
        # calibrate camera
        ret, mtx, dist, rvecs, tvecs  = cv2.calibrateCamera(objpoints, imgpoints, img_cal.shape[1:], None,None)
        return ret, mtx, dist, rvecs, tvecs
    ## Apply distortion correction
    def undistort_image(self, img, _mtx, _dist):
        return cv2.undistort(img, _mtx, _dist, None, _mtx)
        
    ## Generate thesholded binary image (color transform, gradients etc...)
    def extract_single_color(self, img, channel='gray'):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  
        if(channel == 'r'):
            return img[:,:,0]
        elif(channel == 'g'):
            return img[:,:,1]
        elif(channel == 'b'):
            return img[:,:,2]
        elif(channel == 'gray'):
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif(channel == 'h'):
            return hls[:,:,0]
        elif(channel == 'l'):
            return hls[:,:,1]
        elif(channel == 's'):
            return hls[:,:,2]
    
    def abs_sobel_thresh(self, image_binary, orient='x', sobel_kernel=3, thresh=(0, 255)):
        self.image_binary = image_binary
        self.orient = orient
        self.sobel_kernel = sobel_kernel
        self.thresh = thresh
        # Calculate directional gradient
        if orient == 'x':
            sobel_orient = cv2.Sobel(image_binary, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel_orient = cv2.Sobel(image_binary, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel_orient)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Apply threshold
        grad_binary = np.zeros_like(scaled_sobel)        
        grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 255 #imshow accepts 1 not!
        return grad_binary

    def abs_magn_thresh(self, image_binary, magn_sobel_kernel=3, thresh_2=(0, 255)):
        # Calculate gradient magnitude
        self.image_binary = image_binary
        self.magn_sobel_kernel = magn_sobel_kernel
        self.thresh_2 = thresh_2
        sobel_x = cv2.Sobel(image_binary, cv2.CV_64F, 1, 0, ksize=magn_sobel_kernel)
        sobel_y = cv2.Sobel(image_binary, cv2.CV_64F, 0, 1, ksize=magn_sobel_kernel)
        # magn = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        magn = np.sqrt(np.power(sobel_x,2) + np.power(sobel_y,2))
        scaled_magn = np.uint8(255*magn/np.max(magn))
        # Apply threshold
        magn_binary = np.zeros_like(scaled_magn)       
        magn_binary[(scaled_magn > (thresh_2[0])) & (scaled_magn < thresh_2[1])] = 255
        return magn_binary
    
    def abs_dir_threshold(self, image_binary, dir_sobel_kernel=3, dir_thresh=(-np.pi/2, np.pi/2)):
        self.image_binary = image_binary
        self.dir_sobel_kernel = dir_sobel_kernel
        self.dir_thresh = dir_thresh
        # Calculate gradient direction
        sobel_x = cv2.Sobel(image_binary, cv2.CV_64F, 1, 0, ksize=dir_sobel_kernel)
        sobel_y = cv2.Sobel(image_binary, cv2.CV_64F, 0, 1, ksize=dir_sobel_kernel)
        abs_grad_x = np.absolute(sobel_x)
        abs_grad_y = np.absolute(sobel_y)
        direction_grad = np.arctan2(abs_grad_y, abs_grad_x)
        # Apply threshold
        dir_binary = np.zeros_like(direction_grad)
        dir_binary[(direction_grad > dir_thresh[0]) & (direction_grad < dir_thresh[1])] = 1
        return dir_binary
    
    def abs_average(self, binary_image, filter_size=3):
        output_image = cv2.blur(binary_image, (filter_size, filter_size))       
        # output_image = cv2.medianBlur(binary_image, filter_size)

        return output_image
    
    def abs_threshold(self, image, thres_low, thres_high=255):
        binary_image = np.zeros_like(image)
        binary_image[(image > thres_low) & (image < thres_high)] = 255
        return binary_image

    def image_to_thresholded_binary(self, _img_undist):
        _h_channel_low = 96
        _h_channel_high = 102
        _l_channel_low = 220
        _l_channel_high = 255

        _img_undist_extract = _img_undist.copy()
        single_channel_h = self.extract_single_color(_img_undist_extract, 'h')
        single_channel_l = self.extract_single_color(_img_undist_extract, 'l')
        binary_channel_h = self.abs_threshold(single_channel_h, _h_channel_low, _h_channel_high)
        binary_channel_l = self.abs_threshold(single_channel_l, _l_channel_low, _l_channel_high)

        channels_binary = np.zeros_like(binary_channel_h)
        channels_binary[(binary_channel_h > 0) | (binary_channel_l > 0)] = 255
        # channels_binary[(binary_channel_h > 0)] = 255
    
        # filter parameters 1
        _sobelx_low = 12 #(12,18)
        _sobelx_high = 255
        _sobelx_filter = 3

        _sobely_low = 24 #(24,0)
        _sobely_high = 255
        _sobely_filter = 3

        _magn_low = 15 #(x, 103)
        _magn_high = 255 
        _magn_filter = 3

        _direction_low = 229
        _direction_high = 287 #269 (0,0)
        _direction_filter = 15
        _direction_avg_filter = 11 #(1)
        _direction_thresh = 225


        _post_avg_filter = 1#9 (x,5)
        _post_thresh = 126#80  (x, 158)
        # _sobelx_low = 5 #(12,18)
        # _sobelx_high = 50
        # _sobelx_filter = 3

        # _sobely_low = 6 #(24,0)
        # _sobely_high = 50
        # _sobely_filter = 3

        # _magn_low = 15 #(x, 103)
        # _magn_high = 0 
        # _magn_filter = 3

        # _direction_low = 229
        # _direction_high = 287 #269 (0,0)
        # _direction_filter = 15
        # _direction_avg_filter = 11 #(1)
        # _direction_thresh = 225


        # _post_avg_filter = 1#9 (x,5)
        # _post_thresh = 126#80  (x, 158)

        # filter parameters 2
        _2_sobelx_low =  18#(12,18)
        _2_sobelx_high = 255
        _2_sobelx_filter = 3

        _2_sobely_low = 0 #(24,0)
        _2_sobely_high = 255
        _2_sobely_filter = 3

        _2_magn_low = 103 #(x, 103)
        _2_magn_high = 255 
        _2_magn_filter = 3

        _2_direction_low = 229
        _2_direction_high = 0 #269 (0,0)
        _2_direction_filter = 15
        _2_direction_avg_filter = 11 #(1)
        _2_direction_thresh = 255

        _2_post_avg_filter = 5#9 (x,5)
        _2_post_thresh = 158#80  (x, 158)

        # cut image in two sections
        crop_y_border = _img_undist.shape[0]//2 + 120

        _img_undist_1 = _img_undist.copy()
        _img_undist_2 = _img_undist.copy()
        # _img_undist_1 = _img_undist_1[0:crop_y_border, 0:_img_undist.shape[1]]
        # _img_undist_2 = _img_undist_2[crop_y_border:_img_undist.shape[0], 0:_img_undist.shape[1]]
        

        # use functions to generate binary image 1
        _img_undist_copy3 = _img_undist.copy()
        single_channel_gray = self.extract_single_color(_img_undist_copy3, 's')
        _sobelx_binary = self.abs_sobel_thresh(single_channel_gray, 'x', _sobelx_filter, (_sobelx_low, _sobelx_high))
        _sobely_binary = self.abs_sobel_thresh(single_channel_gray, 'y', _sobely_filter, (_sobely_low, _sobely_high))
        _mag_binary = self.abs_magn_thresh(single_channel_gray, _magn_filter, (_magn_low, _magn_high))
        _dir_binary = self.abs_dir_threshold(single_channel_gray, _direction_filter, (_direction_low, _direction_high))
        _avg_img = self.abs_average(_dir_binary, _direction_avg_filter)
        _thres_img = self.abs_threshold(_avg_img, _direction_thresh)

        # combine results of different filters
        combined_binary = np.zeros_like(_sobelx_binary)
        combined_binary[((_sobelx_binary == 255) & (_sobely_binary == 255)) | ((_mag_binary == 255) & (_thres_img == 255))] = 255
        # combined_binary[((_sobelx_binary == 255) & (_sobely_binary == 255)) | (_thres_img == 255)] = 255

        _post_avg_img = self.abs_average(combined_binary, _post_avg_filter)
        _post_thres_img = self.abs_threshold(_post_avg_img, _post_thresh)

        # # use functions to generate binary image 2
        # _2_sobelx_binary = self.abs_sobel_thresh(_img_undist_2, 'x', _2_sobelx_filter, (_2_sobelx_low, _2_sobelx_high))
        # _2_sobely_binary = self.abs_sobel_thresh(_img_undist_2, 'y', _2_sobely_filter, (_2_sobely_low, _2_sobely_high))
        # _2_mag_binary = self.abs_magn_thresh(_img_undist_2, _2_magn_filter, (_2_magn_low, _2_magn_high))
        # _2_dir_binary = self.abs_dir_threshold(_img_undist_2, _2_direction_filter, (_2_direction_low, _2_direction_high))
        # _2_avg_img = self.abs_average(_2_dir_binary, _2_direction_avg_filter)
        # _2_thres_img = self.abs_threshold(_2_avg_img, _2_direction_thresh)

        # # combine results of different filters
        # _2_combined_binary = np.zeros_like(_2_sobelx_binary)
        # # combined_binary[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | ((_2_mag_binary == 255) & (_2_thres_img == 255))] = 255
        # _2_combined_binary[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | (_2_thres_img == 255)] = 255

        # _2_post_avg_img = self.abs_average(_2_combined_binary, _post_avg_filter)
        # _2_post_thres_img = self.abs_threshold(_2_post_avg_img, _post_thresh)

        # concatenated_binary = np.concatenate((_post_thres_img, _2_post_thres_img), axis=0)
        # concatenated_binary = combined_binary
        # concatenated_binary = _post_thres_img
        concatenated_binary = channels_binary

        if self.print_output:
            plt.imshow(concatenated_binary, cmap='gray')
            plt.show()
        if self.write_output:
            cv2.imwrite("output_images/combined_binary.jpg", concatenated_binary)
        return concatenated_binary

    ## Apply a perspective transform to rectify binary image (bird-eye view)
    def unwarp(self, _img):
        # define 4 source points 
        src = np.float32([[200,_img.shape[0]],[597,447],[686,447],[1124,_img.shape[0]]])

        # define 4 destination points 
        dst = np.float32([[300,_img.shape[0]],[300,0],[950,0],[950,_img.shape[0]]])
        
        # get perspective transform
        M =  cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        # warp your image to a top-down view
        img_warped = cv2.warpPerspective(_img, M, _img.shape[1::-1], flags=cv2.INTER_LINEAR)

        if self.print_output:
            plt.imshow(cvt_bgr2rgb(img_warped))
            plt.show()
        # debugging: draw_lines to tune distortion
            # draw_line(img_warped, 312,img_undist.shape[0], 312,0, [255, 255, 255], 3)
            # draw_line(img_warped, 940,img_undist.shape[0], 940,0, [255, 255, 255], 3)
        return img_warped, M, Minv

    ## Detect lane_markings pixels and fit to find the lane_markings boundary.
    # search lane_markings from scratch
    def find_lane_marking_pixels(self, binary_warped, base_x):
       
        # HYPERPARAMETERS
        # Number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create an output image to draw on and visualize the result
        visual_search_korridor = np.dstack((binary_warped, binary_warped, binary_warped))

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = base_x

        # Create empty lists to receive lane_markings marking pixel indices
        lane_markings_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            # Find the four below boundaries of the window ###
            win_xlow = x_current - margin
            win_xhigh = x_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(visual_search_korridor,(win_xlow,win_y_low),
            (win_xhigh,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzerox >= win_xlow) & (nonzerox < win_xhigh) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            
            # Append these indices to the lists
            lane_markings_inds.append(good_inds)
             
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_markings_inds = np.concatenate(lane_markings_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract line pixel positions
        x = nonzerox[lane_markings_inds]
        y = nonzeroy[lane_markings_inds] 

        return x, y, visual_search_korridor

    def search_new_polynomial_from_scratch(self, binary_warped, line, base_x):
        success = True
        # Find our lane_markings pixels first
        x, y, visual_search_korridor = self.find_lane_marking_pixels(binary_warped, base_x)

        # Fit a second order polynomial to each using `np.polyfit` #
        try:
            fit = np.polyfit(y, x, 2)
        except TypeError:
            success = False

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        except TypeError:
            # Avoids an error if `_fit` is still none or incorrect
            print('The function failed to fit a line!')
            success = False
            fitx = 1*ploty**2 + 1*ploty

        # Visualization #
        # Colors in line regions
        visual_search_korridor[y, x] = line.color
        points_lane_markings = np.zeros_like(visual_search_korridor)
        points_lane_markings[y, x] = line.color

        # Plots the polynomial on the lane_markings lines
        plt.plot(fitx, ploty, color='yellow')

        return  points_lane_markings, visual_search_korridor, fit, fitx, ploty, success

    # search lane_markings based on Polynomial fit values from the previous frame
    def fit_poly(self, img_shape, x, y):
        success = True
        # Fit a second order polynomial to each with np.polyfit() #
        try:
            fit = np.polyfit(y, x, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
            # Calc both polynomials using ploty, fit and right_fit #
            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
            return fitx, ploty, fit, success
        except TypeError:
            # Avoids an error if `_fit` is still none or incorrect
            print('The function failed to fit a line!')
            success = False
            fitx = 1*ploty**2 + 1*ploty
            return fitx, ploty, fit, success

    def search_polynomial_from_prior(self, binary_warped, line):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        success = True
        margin = 100
        min_pixels_per_line = 500

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        fit_last = line.best_fit

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function 
        lane_markings_marking_inds = ((nonzerox >= (fit_last[0]*(nonzeroy**2) + fit_last[1]*nonzeroy + fit_last[2] - margin)) & 
                        (nonzerox < (fit_last[0]*(nonzeroy**2) + fit_last[1]*nonzeroy + fit_last[2] + margin)))

        x = nonzerox[lane_markings_marking_inds]
        y = nonzeroy[lane_markings_marking_inds] 

        if (x.size > min_pixels_per_line) & (y.size > min_pixels_per_line):
            # fit new polynomial
            fitx, ploty, fit_new, success_fit_poly = self.fit_poly(binary_warped.shape, x, y)
            
            if not success_fit_poly:
                success = False
        
            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            visual_search_korridor = np.dstack((binary_warped, binary_warped, binary_warped))*255
            points_lane_markings = np.zeros_like(visual_search_korridor)
            
            # Color in line pixels
            points_lane_markings[nonzeroy[lane_markings_marking_inds], nonzerox[lane_markings_marking_inds]] = line.color
            
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, 
                                    ploty])))])
            line_pts = np.hstack((line_window1, line_window2))

            # Draw the lane_markings onto the warped blank image
            cv2.fillPoly(visual_search_korridor, np.int_([line_pts]), (0,255, 0))
            # Plot the polynomial lines onto the image
            plt.plot(fitx, ploty, color='yellow')
            ## End visualization steps ##
        else:
            # reset and start searching from scratch 
            success = False
            # return False, False, False, False, False, False, False, False

        return points_lane_markings, visual_search_korridor, fit_new, fitx, ploty, success

    def get_starting_points_for_search(self, binary_warped_):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped_[binary_warped_.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        base_leftx_ = np.argmax(histogram[:midpoint])
        base_rightx_ = np.argmax(histogram[midpoint:]) + midpoint
        return base_leftx_, base_rightx_

    def get_line_from_scatch(self, binary_warped_, line,  base_x):
        points_lane_markings, visual_search_korridor, fit, fitx, ploty, success = self.search_new_polynomial_from_scratch(binary_warped_, line, base_x)
        if success: #polyline found
            return  points_lane_markings, visual_search_korridor, fit, fitx, ploty, True 

        elif not success and line.cnt_last_invalid < self.max_cnt_last_invalid: # ignore this sample, return last one;
             return points_lane_markings, visual_search_korridor, line.recent_fit_buffer[-1], line.recent_xfitted_buffer[-1], ploty, False 

        elif not success and line.cnt_last_invalid > self.max_cnt_last_invalid: # ignore this sample, return last one; TODO idealy recompute binary_image with coarser parameteres and start new line search
             return points_lane_markings, visual_search_korridor, line.recent_fit_buffer[-1], line.recent_xfitted_buffer[-1], ploty, False 
            # generate new binary_img with coarser params for new line search!

    def detect_single_line(self, binary_warped_, line, base_x):
        if line.cnt_last_invalid < self.max_cnt_last_invalid:
            # searching lane_markings lines from prior 
            points_lane_markings, visual_search_korridor, fit, fitx, ploty, success = self.search_polynomial_from_prior(binary_warped_, line)  
            if success: #polyline found
                return points_lane_markings, visual_search_korridor, fit, fitx, ploty, True

            elif not success and line.cnt_last_invalid < self.max_cnt_last_invalid: # ignore this sample, return last one
                return points_lane_markings, visual_search_korridor, line.recent_fit_buffer[-1], line.recent_xfitted_buffer[-1], ploty, False

            elif not success and line.cnt_last_invalid > self.max_cnt_last_invalid: # start searching from scratch
                points_lane_markings, visual_search_korridor, fit, fitx, ploty, success_from_scratch = get_line_from_scatch(binary_warped_, line, base_x)
                return  points_lane_markings, visual_search_korridor, fit, fitx, ploty, False
        else:
            # searching lane_markings line from scratch (sliding window) for startup and fallback
            points_lane_markings, visual_search_korridor, fit, fitx, ploty, success = self.get_line_from_scatch(binary_warped_, line, base_x)
            return  points_lane_markings, visual_search_korridor, fit, fitx, ploty, success
            

    def combine_and_show_search_korridor(self, binary_warped_, points_lane_markings_left, points_lane_markings_right, korridor_left, korridor_right):
        korridor_combined = cv2.addWeighted(korridor_left, 1, korridor_right, 1, 0)
        lane_markings_combined = cv2.addWeighted(points_lane_markings_left, 1, points_lane_markings_right, 1, 0)
        annotated_search_korridor = cv2.addWeighted(lane_markings_combined, 1, korridor_combined, 0.3, 0)
        # visualization
        # binar_warped_3_channel = cv2.cvtColor(binary_warped_, cv2.COLOR_GRAY2BGR)
        # binary_with_lane_markings = cv2.addWeighted(binar_warped_3_channel, 0.5, result, 1, 0)
        if self.print_output:
            plt.imshow(annotated_search_korridor)
            plt.show()

    def combine_lane_markings(self, binary_warped_, markings_left , markings_right):
        lane_markings = cv2.addWeighted(markings_left, 1, markings_right, 1, 0)
        return lane_markings

    def check_curvature_left_right_ok(self, binary_warped_, fit_left, fit_right):
        max_diff_curvature = 1000 #really rough check
        curverad_left = self.line_left.calc_curvature_real(binary_warped_.shape[0], fit_left)
        curverad_right = self.line_left.calc_curvature_real(binary_warped_.shape[0], fit_right)
        return math.isclose(curverad_left, curverad_right, abs_tol=max_diff_curvature)

    def check_lines_horiz_distance_left_right_ok(self, binary_warped_, fitx_left, fitx_right):
        valid_dist_in_pixel = 650
        max_diff_pixel = 50
        dist_in_pixel = fitx_left[binary_warped_.shape[0]-1] - fitx_right[binary_warped_.shape[0]-1]
        return math.isclose(dist_in_pixel, valid_dist_in_pixel, abs_tol=max_diff_pixel)

    def check_curvature_new_last_ok(self, binary_warped_, line, fit):
        max_diff_curvature = 500 #really rough check
        curverad_new = line.calc_curvature_real(binary_warped_.shape[0], fit)
        if len(line.recent_radius_of_curvature_buffer) > 0:
            curverad_avg = line.avg_radius_of_curvature
        else: # if buffer is empty (after init), ignore check
            return True 
        return math.isclose(curverad_avg, curverad_new, abs_tol=max_diff_curvature)

    def check_line_horiz_pos_new_last_ok(self, binary_warped_, line, fitx):
        max_diff_pos_pixel = 50
        line_pos_new = fitx[binary_warped_.shape[0]-1]
        if len(line.recent_xfitted_buffer) > 0:
            line_pos_avg = line.bestx[binary_warped_.shape[0]-1]        
        else: # if buffer is empty (after init), ignore check
            return True 
        return math.isclose(line_pos_avg, line_pos_new, abs_tol=max_diff_pos_pixel)


    def validation_check_lane(self, binary_warped_, fit_left, fitx_left, fit_right, fitx_right):
        curverad_left_right_ok = self.check_curvature_left_right_ok(binary_warped_, fit_left, fit_right)
        horiz_dist_left_right_ok = self.check_lines_horiz_distance_left_right_ok(binary_warped_, fitx_left, fitx_right)
        return curverad_left_right_ok & horiz_dist_left_right_ok

    def validation_check_line(self, binary_warped_, line, fit, fitx):
        curverad_new_last_ok = self.check_curvature_new_last_ok(binary_warped_, line, fit)
        horiz_pos_new_last_ok = self.check_line_horiz_pos_new_last_ok(binary_warped_, line, fitx)
        return curverad_new_last_ok & horiz_pos_new_last_ok

    def update_with_valid_line(self, binary_warped_, line, fit, fitx, base_x):
        # check if left line is valid in relation to its avg
            if self.validation_check_line(binary_warped_, line, fit, fitx):
                line.update(binary_warped_.shape, True, fitx, fit)
            else:
                # search line from scratch 
                points_lane_markings, visual_search_korridor, fit, fitx, ploty, success_ = self.get_line_from_scatch(binary_warped_, line, base_x)
                if success_: # new line found, or last avg used
                    line.update(binary_warped_.shape, True, fitx, fit)
                else:
                    # if not found: but last line used, update !with is_valid=False! is needed
                    line.update(binary_warped_.shape, False, fitx_, fit_)

    def ensure_valid_lane_and_smoothen(self, binary_warped_, fit_left, fitx_left, fit_right, fitx_right, base_leftx, base_rightx):
        # check realtion
        lane_valid = self.validation_check_lane(binary_warped_, fit_left, fitx_left, fit_right, fitx_right)
        # if relation check ok: update prior lines 
        if lane_valid:
            self.line_left.update(binary_warped_.shape(), lane_valid, fitx_left, fit_left)
            self.line_right.update(binary_warped_.shape(), lane_valid, fitx_right, fit_right)
        else:
            # check if left line is valid in relation to its avg
            self.update_with_valid_line(binary_warped_, self.line_left, fit_left, fitx_left, base_leftx)

            # check if right line is valid in relation to its avg
            self.update_with_valid_line(binary_warped_, self.line_right, fit_right, fitx_right, base_rightx)

        # provide valid lines
        return self.line_left.best_fit, self.line_left.bestx, self.line_right.best_fit, self.line_right.bestx
    
    def detect_lines(self, binary_warped_):
        # seperate left and right
        base_leftx, base_rightx = self.get_starting_points_for_search(binary_warped_)
        # detect line left
        points_lane_markings_left, visual_search_korridor_left, fit_left, fitx_left, ploty_left, success_left = self.detect_single_line(binary_warped_, self.line_left, base_leftx)
        # detect line right
        points_lane_markings_right, visual_search_korridor_right, fit_right, fitx_right, ploty_right, success_right = self.detect_single_line(binary_warped_, self.line_right, base_rightx)
        
        # validation checker here!! Update cnt_is_valid_value
        fit_left, fitx_left, fit_right, fitx_right = self.ensure_valid_lane_and_smoothen(binary_warped_, fit_left, fitx_left, fit_right, fitx_right, base_leftx, base_rightx)

        colored_lane_markings = self.combine_lane_markings(binary_warped_, points_lane_markings_left, points_lane_markings_right)
        self.combine_and_show_search_korridor(binary_warped_, points_lane_markings_left, points_lane_markings_right, visual_search_korridor_left, visual_search_korridor_right)
        return  colored_lane_markings, fit_left, fit_right, fitx_left, fitx_right, ploty_left

    ## Determine the curvature of the lane_markings and vehicle position with respect to center.
    def measure_curvature_real(self, y_eval, left_fit_cr, right_fit_cr):

        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3/(63) # 63(53-69)pixel; meters per pixel in y dimension
        xm_per_pix = 3.7/611 # 611(605-620)pixel; meters per pixel in x dimension
        
        # Implement the calculation of R_curve (radius of curvature)
        left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2)/(np.absolute(2*left_fit_cr[0]))  ## Implement the calculation of the left line here
        right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2)/(np.absolute(2*right_fit_cr[0]))  ## Implement the calculation of the right line here
        
        return left_curverad, right_curverad
    
    ## Warp the detected lane_markings boundaries back onto the original image.
    def warp_back_on_original(self, binary_warped_, _img_undist, _left_fitx, _right_fitx, _ploty, _perspective_Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped_).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([_left_fitx, _ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([_right_fitx, _ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane_markings onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        rewarp_annotated = cv2.warpPerspective(color_warp, _perspective_Minv, (binary_warped_.shape[1], binary_warped_.shape[0])) 

        # Combine the result with the original image
        result_lane_markings = cv2.addWeighted(_img_undist, 1, rewarp_annotated, 0.3, 0)
        
        result_bgr = cv2.cvtColor(result_lane_markings, cv2.COLOR_RGB2BGR)
        # if self.print_output:
        #     plt.imshow(result_lane_markings)
        #     plt.show()
        if self.write_output:
            cv2.imwrite("output_images/result.jpg", result_bgr)
        return result_lane_markings

    ## Output visual display of the lane_markings boundaries and numerical estimation of lane_markings curvature and vehicle position
    # warp output back to original perspective (for overlay annotation)
    def annotate_lane_markings_markings(self, _img_result_lane_markings, _lane_markings, _perspective_Minv, _curverad, _offset_lane_markings_center, fit_left, fitx_left, fit_right, fitx_right, binary_warped_):
        rewarp_lane_markings_markings = cv2.warpPerspective(_lane_markings, _perspective_Minv, (_img_result_lane_markings.shape[1], _img_result_lane_markings.shape[0])) 
        result_annotated = cv2.addWeighted(_img_result_lane_markings, 0.7, rewarp_lane_markings_markings, 1, 0)

        # create text output for numerial estimations
        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontscale = 1.7
        color = (0, 0, 255) 
        thickness = 4
        text_curvature = "Radius of Curvature: %4.0fm" % _curverad
        if _offset_lane_markings_center >= 0:
            text_offset = "Vehicle is %2.2fm right of center" % _offset_lane_markings_center
        else:
            _offset_lane_markings_center = abs(_offset_lane_markings_center)
            text_offset = "Vehicle is %2.2fm left of center" % _offset_lane_markings_center

        #TODO: DELETE AFTER - TESTING:
        curverad_left = self.line_left.calc_curvature_real(binary_warped_.shape[0], fit_left)
        curverad_right = self.line_left.calc_curvature_real(binary_warped_.shape[0], fit_right)
        horz_dist_lines_pixel = fitx_right[binary_warped_.shape[0]-1] - fitx_left[binary_warped_.shape[0]-1]
        pos_left_pixel = fitx_left[binary_warped_.shape[0]-1]
        pos_right_pixel = fitx_right[binary_warped_.shape[0]-1]

        text_rad_left = "Rad_left: %4.0fm" % curverad_left
        text_rad_right = "Rad_right: %4.0fm" % curverad_right
        # text_dist_lines = "pos_line_left: %1.2f pixel  " % pos_left_pixel
        cv2.putText(result_annotated, text_rad_left, (50,200),font,fontscale,color,thickness)
        cv2.putText(result_annotated, text_rad_right, (600,200),font,fontscale,color,thickness)
        # cv2.putText(result_annotated, text_dist_lines, (50,300),font,fontscale,color,thickness)
        #TODO: DELETE ABOVE

        cv2.putText(result_annotated, text_curvature, (50,50),font,fontscale,color,thickness)
        cv2.putText(result_annotated, text_offset, (50,100),font,fontscale,color,thickness)
        result_annotated_bgr = cv2.cvtColor(result_annotated, cv2.COLOR_RGB2BGR)
        
        if self.print_output:
            plt.imshow(result_annotated)
            plt.show()
        if self.write_output:
            cv2.imwrite("output_images/result_2.jpg", result_annotated_bgr)
        return result_annotated
    
    def calc_offset_lane_markings_center(self, _left_best_fitx, _right_best_fitx, _img_shape, _roi_offset):
        meter_per_pixel = 3.7 / (_right_best_fitx[_img_shape[0]-1] - _left_best_fitx[_img_shape[0]-1])  # lane_markings is roughly 3.7m wide
        middle_of_lane_markings_in_roi = _left_best_fitx[_img_shape[0]-1] + (_right_best_fitx[_img_shape[0]-1] - _left_best_fitx[_img_shape[0]-1])/2 
        middle_of_image_in_roi = _img_shape[1]//2
        offset_lane_markings_center_roi = middle_of_lane_markings_in_roi - middle_of_image_in_roi
        offset_lane_markings_center_img = offset_lane_markings_center_roi - _roi_offset 
        offset_in_meter = offset_lane_markings_center_img * meter_per_pixel
        return offset_in_meter

    ## Video lane_markings detection ##
    def process_image(self, _img_input):
        img_shape = _img_input.shape
        roi_offset = abs(200 - (img_shape[1] - 1124))/2 # depending on warp points
        
        img_undist = self.undistort_image(_img_input, mtx, dist)
        img_combined_binary = self.image_to_thresholded_binary(img_undist)
        imgbinary_warped_, perspective_M, perspective_Minv = self.unwarp(img_combined_binary)
        lane_markings, left_fit, right_fit, left_fitx, right_fitx, ploty = self.detect_lines(imgbinary_warped_)
        # delete update:
        # left_curverad, left_best_fitx, left_best_fit = self.line_left.update(img_shape, True, left_fitx, left_fit)
        # right_curverad, right_best_fitx, right_best_fit = self.line_right.update(img_shape,True, right_fitx, right_fit)
        img_result_lane_markings = self.warp_back_on_original(imgbinary_warped_, img_undist, left_fitx, right_fitx, ploty, perspective_Minv)
        offset_lane_markings_center = self.calc_offset_lane_markings_center(left_fitx, right_fitx, img_shape, roi_offset)
        result_annotated = self.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, self.line_right.avg_radius_of_curvature, offset_lane_markings_center, left_fit, left_fitx, right_fit, right_fitx,_img_input)
        # result_annotated = self.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, right_curverad, offset_lane_markings_center)
        return result_annotated

    def save_frames_of_video(self, _frame):
        self.iteration_cnt += 1
        _frame_BGR = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"video_frames/frame{self.iteration_cnt}.jpg", _frame_BGR)
        return _frame

    def execute_video_pipeline(self):
        white_output = 'test_videos_output/project_video.mp4'
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer va|ues representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        # clip1 = VideoFileClip("project_video.mp4").subclip(24,26)
        clip1 = VideoFileClip("project_video.mp4").subclip(38,42)
        # clip1 = VideoFileClip("project_video.mp4").subclip(32,42)
        # clip1 = VideoFileClip("project_video.mp4").subclip(39,41)
        # clip1 = VideoFileClip("project_video.mp4").subclip(50)
        white_clip = clip1.fl_image(self.process_image) #NOTE: this function expects color images!!
        # white_clip = clip1.fl_image(self.save_frames_of_video) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)    

#%%
if __name__ == "__main__":
    # setup
    plt.rcParams["figure.figsize"] = (25,15)
    os.chdir("/home/nbenndo/Documents/Programming/Udacity/SelfDrivingCarND/CarND-Advanced-Lane-Lines/")
    ld = lane_markings_detection()
    ld.print_output = False
    video_pipeline = True
    # initialisation
    cret, mtx, dist, rvecs, tvecs = ld.compute_camera_calibration()
    
#%%

    if video_pipeline:
        ld.execute_video_pipeline()
    else:
        # load input image
        
        img_input = plt.imread('video_frames/frame1.jpg')
        img_undist = ld.undistort_image(img_input, mtx, dist)
        img_shape = img_undist.shape
        roi_offset = abs(200 - (img_shape[1] - 1124))/2 # depending on warp points

        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        imgbinary_warped_, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lane_markings, lane_markings, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lines(imgbinary_warped_)
        left_curverad, left_best_fitx, left_best_fit = ld.line_left.update(img_shape, True, left_fitx, left_fit)
        right_curverad, right_best_fitx, right_best_fit = ld.line_right.update(img_shape,True, right_fitx, right_fit)
        img_result_lane_markings = ld.warp_back_on_original(imgbinary_warped_, img_undist, left_best_fitx, right_best_fitx, ploty, perspective_Minv)
        offset_lane_markings_center = ld.calc_offset_lane_markings_center(left_best_fitx, right_best_fitx, img_shape, roi_offset)
        result_annotated = ld.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, right_curverad, offset_lane_markings_center)

        img_input = plt.imread('video_frames/frame2.jpg')
        img_undist = ld.undistort_image(img_input, mtx, dist)

        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        imgbinary_warped_, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lane_markings, lane_markings, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lines(imgbinary_warped_)
        left_curverad, left_best_fitx, left_best_fit = ld.line_left.update(img_shape, True, left_fitx, left_fit)
        right_curverad, right_best_fitx, right_best_fit = ld.line_right.update(img_shape,True, right_fitx, right_fit)
        img_result_lane_markings = ld.warp_back_on_original(imgbinary_warped_, img_undist, left_best_fitx, right_best_fitx, ploty, perspective_Minv)
        offset_lane_markings_center = ld.calc_offset_lane_markings_center(left_best_fitx, right_best_fitx, img_shape, roi_offset)
        result_annotated = ld.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, right_curverad, offset_lane_markings_center)


        img_input = plt.imread('video_frames/frame3.jpg')
        img_undist = ld.undistort_image(img_input, mtx, dist)

        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        imgbinary_warped_, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lane_markings, lane_markings, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lines(imgbinary_warped_)
        left_curverad, left_best_fitx, left_best_fit = ld.line_left.update(img_shape, True, left_fitx, left_fit)
        right_curverad, right_best_fitx, right_best_fit = ld.line_right.update(img_shape,True, right_fitx, right_fit)
        img_result_lane_markings = ld.warp_back_on_original(imgbinary_warped_, img_undist, left_best_fitx, right_best_fitx, ploty, perspective_Minv)
        offset_lane_markings_center = ld.calc_offset_lane_markings_center(left_best_fitx, right_best_fitx, img_shape, roi_offset)
        result_annotated = ld.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, right_curverad, offset_lane_markings_center)

#         # load input image
#         img_undist = ld.undistort_image(img_input, mtx, dist)
#         img_combined_binary = ld.image_to_thresholded_binary(img_undist)
#         imgbinary_warped_, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
#         img_fittet_lane_markings, lane_markings, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lane_markings(imgbinary_warped_)
#         left_curverad, right_curverad = ld.measure_curvature_real(img_fittet_lane_markings.shape[0], left_fit, right_fit)
#         img_result_lane_markings = ld.warp_back_on_original(imgbinary_warped_, img_undist, left_fitx, right_fitx, ploty, perspective_Minv)
#         result_annotated = ld.annotate_lane_markings_markings(img_result_lane_markings, lane_markings, perspective_Minv, left_curverad, right_curverad)
        




# %%


