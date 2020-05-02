#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import array
import glob

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


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
        
class lane_detection:
    iteration_cnt = 0
    print_output = False
    write_output = False
    left_fit = np.array([0.0, 0.0, 0.0])
    right_fit = np.array([0.0, 0.0, 0.0])
    def init(self):
        pass
        
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
        imgpoints = [] # 2d points in image plane.

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
    def extract_single_color(self, img):
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        return s_channel

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = self.extract_single_color(image)
        if orient == 'x':
            sobel_orient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel_orient = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel_orient)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Apply threshold
        grad_binary = np.zeros_like(scaled_sobel)        
        grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 255 #imshow accepts 1 not!
        return grad_binary

    def abs_magn_thresh(self, image, magn_sobel_kernel=3, thresh_2=(0, 255)):
        # Calculate gradient magnitude
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = self.extract_single_color(image)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=magn_sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=magn_sobel_kernel)
        # magn = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        magn = np.sqrt(np.power(sobel_x,2) + np.power(sobel_y,2))
        scaled_magn = np.uint8(255*magn/np.max(magn))
        # Apply threshold
        magn_binary = np.zeros_like(scaled_magn)       
        magn_binary[(scaled_magn > (thresh_2[0])) & (scaled_magn < thresh_2[1])] = 255
        return magn_binary

    def abs_dir_threshold(self, image, dir_sobel_kernel=3, dir_thresh=(-np.pi/2, np.pi/2)):
        # Calculate gradient direction
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = self.extract_single_color(image)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=dir_sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=dir_sobel_kernel)
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

    def abs_threshold(self, image, threshold):
        binary_image = np.zeros_like(image)
        binary_image[image > threshold/255] = 255
        return binary_image

    def image_to_thresholded_binary(self, _img_undist):

        # filter parameters 1
        _sobelx_low = 5 #(12,18)
        _sobelx_high = 50
        _sobelx_filter = 3

        _sobely_low = 6 #(24,0)
        _sobely_high = 50
        _sobely_filter = 3

        _magn_low = 15 #(x, 103)
        _magn_high = 0 
        _magn_filter = 3

        _direction_low = 229
        _direction_high = 287 #269 (0,0)
        _direction_filter = 15
        _direction_avg_filter = 11 #(1)
        _direction_thresh = 225


        _post_avg_filter = 1#9 (x,5)
        _post_thresh = 126#80  (x, 158)

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
        _img_undist_1 = _img_undist_1[0:crop_y_border, 0:_img_undist_1.shape[1]]
        _img_undist_2 = _img_undist_2[crop_y_border:_img_undist_2.shape[0], 0:_img_undist_2.shape[1]]
        

        # use functions to generate binary image 1
        _sobelx_binary = self.abs_sobel_thresh(_img_undist_1, 'x', _sobelx_filter, (_sobelx_low, _sobelx_high))
        _sobely_binary = self.abs_sobel_thresh(_img_undist_1, 'y', _sobely_filter, (_sobely_low, _sobely_high))
        _mag_binary = self.abs_magn_thresh(_img_undist_1, _magn_filter, (_magn_low, _magn_high))
        _dir_binary = self.abs_dir_threshold(_img_undist_1, _direction_filter, (_direction_low, _direction_high))
        _avg_img = self.abs_average(_dir_binary, _direction_avg_filter)
        _thres_img = self.abs_threshold(_avg_img, _direction_thresh)

        # combine results of different filters
        combined_binary = np.zeros_like(_sobelx_binary)
        # combined_binary[((_sobelx_binary == 255) & (_sobely_binary == 255)) | ((_mag_binary == 255) & (_thres_img == 255))] = 255
        combined_binary[((_sobelx_binary == 255) & (_sobely_binary == 255)) | (_thres_img == 255)] = 255

        _post_avg_img = self.abs_average(combined_binary, _post_avg_filter)
        _post_thres_img = self.abs_threshold(_post_avg_img, _post_thresh)

        # use functions to generate binary image 2
        _2_sobelx_binary = self.abs_sobel_thresh(_img_undist_2, 'x', _2_sobelx_filter, (_2_sobelx_low, _2_sobelx_high))
        _2_sobely_binary = self.abs_sobel_thresh(_img_undist_2, 'y', _2_sobely_filter, (_2_sobely_low, _2_sobely_high))
        _2_mag_binary = self.abs_magn_thresh(_img_undist_2, _2_magn_filter, (_2_magn_low, _2_magn_high))
        _2_dir_binary = self.abs_dir_threshold(_img_undist_2, _2_direction_filter, (_2_direction_low, _2_direction_high))
        _2_avg_img = self.abs_average(_2_dir_binary, _2_direction_avg_filter)
        _2_thres_img = self.abs_threshold(_2_avg_img, _2_direction_thresh)

        # combine results of different filters
        _2_combined_binary = np.zeros_like(_2_sobelx_binary)
        # combined_binary[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | ((_2_mag_binary == 255) & (_2_thres_img == 255))] = 255
        _2_combined_binary[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | (_2_thres_img == 255)] = 255

        _2_post_avg_img = self.abs_average(_2_combined_binary, _post_avg_filter)
        _2_post_thres_img = self.abs_threshold(_2_post_avg_img, _post_thresh)

        concatenated_binary = np.concatenate((_post_thres_img, _2_post_thres_img), axis=0)
        # concatenated_binary = _2_post_thres_img

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

    ## Detect lane pixels and fit to find the lane boundary.
    # search lanes from scratch
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        img_lanes = np.copy(out_img) # copy output_image for selected visualization
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            # Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img, img_lanes

    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img, img_lanes = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit` #
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        # Visualization #
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        img_lanes[lefty, leftx] = [255, 0, 0]
        img_lanes[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        return out_img, img_lanes, left_fit, right_fit, left_fitx, right_fitx, ploty

    # search lanes based on Polynomial fit values from the previous frame
    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit() #
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit #
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        return left_fitx, right_fitx, ploty, left_fit, right_fit

    def search_around_poly(self, binary_warped, left_fit_last, right_fit_last):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        success = True
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function 
        left_lane_inds = ((nonzerox >= (left_fit_last[0]*(nonzeroy**2) + left_fit_last[1]*nonzeroy + left_fit_last[2] - margin)) & 
                        (nonzerox < (left_fit_last[0]*(nonzeroy**2) + left_fit_last[1]*nonzeroy + left_fit_last[2] + margin)))
        right_lane_inds = ((nonzerox >= (right_fit_last[0]*(nonzeroy**2) + right_fit_last[1]*nonzeroy +  right_fit_last[2] - margin)) & 
                        (nonzerox < (right_fit_last[0]*(nonzeroy**2) + right_fit_last[1]*nonzeroy + right_fit_last[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        print("leftx.size: ",leftx.size)
        print("rightx.size: ", rightx.size)
        if (leftx.size > 500) & (lefty.size > 500) & (rightx.size > 500) & (righty.size > 500):
            # fit new polynomials
            left_fitx, right_fitx, ploty, left_fit_new, right_fit_new = ld.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ##
        else:
            # reset and start searching from scratch 
            return False, False, False, False, False, False, False, False

        return result, out_img, left_fit_new, right_fit_new, left_fitx, right_fitx, ploty, success

    def detect_lanes(self, _binary_warped):
        lane_detection.iteration_cnt += 1
        if lane_detection.iteration_cnt <= 1:
            # searching lane lines from scratch (sliding window) for startup and fallback
            img_fittet_lanes, lanes, self.left_fit, self.right_fit, self.left_fitx, self.right_fitx, ploty = self.fit_polynomial(_binary_warped)
        else:
            # searching lane lines from prior 
            img_fittet_lanes, lanes, self.left_fit, self.right_fit, self.left_fitx, self.right_fitx, ploty, success = self.search_around_poly(_binary_warped, self.left_fit, self.right_fit)
            if not success: # start searching from scratch
                img_fittet_lanes, lanes, self.left_fit, self.right_fit, self.left_fitx, self.right_fitx, ploty = self.fit_polynomial(_binary_warped)    
        if self.print_output:
            plt.imshow(img_fittet_lanes)
            plt.show()
        return img_fittet_lanes, lanes, self.left_fit, self.right_fit, self.left_fitx, self.right_fitx, ploty

    ## Determine the curvature of the lane and vehicle position with respect to center.
    def measure_curvature_real(self, y_eval, left_fit_cr, right_fit_cr):

        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3/63 # 63(53-69)pixel; meters per pixel in y dimension
        xm_per_pix = 3.7/611 # 611(605-620)pixel; meters per pixel in x dimension
        
        # Implement the calculation of R_curve (radius of curvature)
        left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2)/(np.absolute(2*left_fit_cr[0]))  ## Implement the calculation of the left line here
        right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2)/(np.absolute(2*right_fit_cr[0]))  ## Implement the calculation of the right line here
        
        return left_curverad, right_curverad
    
    ## Warp the detected lane boundaries back onto the original image.
    def warp_back_on_original(self, _binary_warped, _img_undist, _left_fitx, _right_fitx, _ploty, _perspective_Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(_binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([_left_fitx, _ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([_right_fitx, _ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        rewarp_annotated = cv2.warpPerspective(color_warp, _perspective_Minv, (_binary_warped.shape[1], _binary_warped.shape[0])) 

        # Combine the result with the original image
        result_lane = cv2.addWeighted(_img_undist, 1, rewarp_annotated, 0.3, 0)
        
        result_bgr = cv2.cvtColor(result_lane, cv2.COLOR_RGB2BGR)
        # if self.print_output:
        #     plt.imshow(result_lane)
        #     plt.show()
        if self.write_output:
            cv2.imwrite("output_images/result.jpg", result_bgr)
        return result_lane

    ## Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
    # warp output back to original perspective (for overlay annotation)
    def annotate_lane_markings(self, _img_result_lane, _lanes, _perspective_Minv, _left_curverad, _right_curverad):
        rewarp_lane_markings = cv2.warpPerspective(_lanes, _perspective_Minv, (_img_result_lane.shape[1], _img_result_lane.shape[0])) 
        result_annotated = cv2.addWeighted(_img_result_lane, 0.7, rewarp_lane_markings, 1, 0)

        # create text output for numerial estimations
        x = 200
        y = 200
        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontscale = 0.8
        color = (0, 0, 255) 
        thickness = 2
        text_curvature = f"left_curverad: {_left_curverad}, right_curverad: {_right_curverad}"

        cv2.putText(result_annotated, text_curvature,(x,y),font,fontscale,color,thickness)
        result_annotated_bgr = cv2.cvtColor(result_annotated, cv2.COLOR_RGB2BGR)
        
        if self.print_output:
            plt.imshow(result_annotated)
            plt.show()
        if self.write_output:
            cv2.imwrite("output_images/result_2.jpg", result_annotated_bgr)
        return result_annotated
    

    ## Video lane detection ##
    def process_image(self, _img_input):
        img_undist = ld.undistort_image(_img_input, mtx, dist)
        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        img_binary_warped, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lanes, lanes, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lanes(img_binary_warped)
        left_curverad, right_curverad = ld.measure_curvature_real(img_fittet_lanes.shape[0], left_fit, right_fit)
        img_result_lane = ld.warp_back_on_original(img_binary_warped, img_undist, left_fitx, right_fitx, ploty, perspective_Minv)
        result_annotated = ld.annotate_lane_markings(img_result_lane, lanes, perspective_Minv, left_curverad, right_curverad)
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
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(self.process_image) #NOTE: this function expects color images!!
        # white_clip = clip1.fl_image(self.save_frames_of_video) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)    
#%%
if __name__ == "__main__":
    # setup
    plt.rcParams["figure.figsize"] = (25,15)

    ld = lane_detection()
    ld.print_output = False
    video_pipeline = True
    # initialisation
    cret, mtx, dist, rvecs, tvecs = ld.compute_camera_calibration()

#%%
    if video_pipeline:
        ld.execute_video_pipeline()
    else:
        # load input image
        img_input = plt.imread('test_images/straight_lines2.jpg')
        img_undist = ld.undistort_image(img_input, mtx, dist)
        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        img_binary_warped, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lanes, lanes, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lanes(img_binary_warped)
        left_curverad, right_curverad = ld.measure_curvature_real(img_fittet_lanes.shape[0], left_fit, right_fit)
        img_result_lane = ld.warp_back_on_original(img_binary_warped, img_undist, left_fitx, right_fitx, ploty, perspective_Minv)
        result_annotated = ld.annotate_lane_markings(img_result_lane, lanes, perspective_Minv, left_curverad, right_curverad)
            

        # load input image
        img_undist = ld.undistort_image(img_input, mtx, dist)
        img_combined_binary = ld.image_to_thresholded_binary(img_undist)
        img_binary_warped, perspective_M, perspective_Minv = ld.unwarp(img_combined_binary)
        img_fittet_lanes, lanes, left_fit, right_fit, left_fitx, right_fitx, ploty = ld.detect_lanes(img_binary_warped)
        left_curverad, right_curverad = ld.measure_curvature_real(img_fittet_lanes.shape[0], left_fit, right_fit)
        img_result_lane = ld.warp_back_on_original(img_binary_warped, img_undist, left_fitx, right_fitx, ploty, perspective_Minv)
        result_annotated = ld.annotate_lane_markings(img_result_lane, lanes, perspective_Minv, left_curverad, right_curverad)
        




# %%
