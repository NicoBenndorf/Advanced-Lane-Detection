#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle
import glob
import sys

class ParameterFinder:
    def __init__(self, image, sobelx_filter=1, sobelx_low=0, sobelx_high=0, 
                sobely_filter=1, sobely_low=0, sobely_high=0, 
                magn_filter=1, magn_low=0, magn_high=0, 
                direction_filter=1, direction_low=0, direction_high=0,
                direction_avg_filter=3, direction_thresh=0, load_params_path= "do_not_load"):
        self.image = image
        self._sobelx_filter = sobelx_filter
        self._sobelx_low = sobelx_low
        self._sobelx_high = sobelx_high
        self._sobely_filter = sobely_filter
        self._sobely_low = sobely_low
        self._sobely_high = sobely_high
        self._magn_filter = magn_filter
        self._magn_low = magn_low
        self._magn_high = magn_high
        self._direction_filter = direction_filter
        self._direction_low = direction_low
        self._direction_high = direction_high
        self._direction_avg_filter = direction_avg_filter
        self._direction_thresh = direction_thresh
        self._post_avg_filter = 1
        self._post_thresh = 1

        if load_params_path != "do_not_load":
            [self._sobelx_filter, self._sobelx_low, self._sobelx_high, self._sobely_filter, self._sobely_low, self._sobely_high, self._magn_filter, self._magn_low, self._magn_high, self._direction_filter, self._direction_low, self._direction_high, self._direction_avg_filter, self._direction_thresh] = self.load_params(load_params_path, [self._sobelx_filter, self._sobelx_low, self._sobelx_high, self._sobely_filter, self._sobely_low, self._sobely_high, self._magn_filter, self._magn_low, self._magn_high, self._direction_filter, self._direction_low, self._direction_high, self._direction_avg_filter, self._direction_thresh])
            print("self._sobelx_filter: ", self._sobelx_filter)


        def onchange_sobelx_low(pos):
            self._sobelx_low = pos
            self._render()
        def onchange_sobelx_high(pos):
            self._sobelx_high = pos
            self._render()
        def onchange_sobelx_filter(pos):
            self._sobelx_filter = pos
            self._sobelx_filter += (self._sobelx_filter + 1) % 2
            self._render() 
        
        def onchange_sobely_low(pos):
            self._sobely_low = pos
            self._render()
        def onchange_sobely_high(pos):
            self._sobely_high = pos
            self._render()
        def onchange_sobely_filter(pos):
            self._sobely_filter = pos
            self._sobely_filter += (self._sobely_filter + 1) % 2
            self._render()
        
        def onchange_magn_low(pos):
            self._magn_low = pos
            self._render()
        def onchange_magn_high(pos):
            self._magn_high = pos
            self._render()
        def onchange_magn_filter(pos):
            self._magn_filter = pos
            self._magn_filter += (self._magn_filter + 1) % 2
            self._render()
            
        def onchange_direction_low(pos):
            self._direction_low = (pos/100)-(np.pi/2)
            self._render()
        def onchange_direction_high(pos):
            self._direction_high = (pos/100)-(np.pi/2)
            self._render()
        def onchange_direction_filter(pos):
            self._direction_filter = pos
            self._direction_filter += (self._direction_filter + 1) % 2
            self._render()
        def onchange_direction_avg_filter(pos):
            self._direction_avg_filter = pos
            self._direction_avg_filter += (self._direction_avg_filter + 1) % 2
            self._render()
        def onchange_direction_thresh(pos):
            self._direction_thresh = pos
            self._render()

        def onchange_post_avg_filter(pos):
            self._post_avg_filter = pos
            self._post_avg_filter += (self._post_avg_filter + 1) % 2
            self._render()
        def onchange_post_thresh(pos):
            self._post_thresh = pos
            self._render()
    
        cv2.namedWindow('output')

        cv2.createTrackbar('sobelx_low', 'output', self._sobelx_low, 255, onchange_sobelx_low)
        cv2.createTrackbar('sobelx_high', 'output', self._sobelx_high, 255, onchange_sobelx_high)
        cv2.createTrackbar('sobelx_filter', 'output', self._sobelx_filter, 21, onchange_sobelx_filter)

        cv2.createTrackbar('sobely_low', 'output', self._sobely_low, 255, onchange_sobely_low)
        cv2.createTrackbar('sobely_high', 'output', self._sobely_high, 255, onchange_sobely_high)
        cv2.createTrackbar('sobely_filter', 'output', self._sobely_filter, 21, onchange_sobely_filter)

        cv2.createTrackbar('magn_low', 'output', self._magn_low, 255, onchange_magn_low)
        cv2.createTrackbar('magn_high', 'output', self._magn_high, 255, onchange_magn_high)
        cv2.createTrackbar('magn_filter', 'output', self._magn_filter, 21, onchange_magn_filter)
        
        cv2.createTrackbar('direction_low(rad)', 'output', self._direction_low, 314, onchange_direction_low)
        cv2.createTrackbar('direction_high(rad)', 'output', self._direction_high, 314, onchange_direction_high)
        cv2.createTrackbar('direction_filter', 'output', self._direction_filter, 21, onchange_direction_filter)
        cv2.createTrackbar('direction_avg_filter', 'output', self._direction_avg_filter, 21, onchange_direction_avg_filter)
        cv2.createTrackbar('direction_thresh', 'output', self._direction_thresh, 255, onchange_direction_thresh)


        cv2.createTrackbar('post_avg_filter', 'output', self._post_avg_filter, 21, onchange_post_avg_filter)
        cv2.createTrackbar('post_thresh', 'output', self._post_thresh, 255, onchange_post_thresh)
        
        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('output')
        self.save_params([self._sobelx_filter, self._sobelx_low, self._sobelx_high, self._sobely_filter, self._sobely_low, self._sobely_high, self._magn_filter, self._magn_low, self._magn_high, self._direction_filter, self._direction_low, self._direction_high, self._direction_avg_filter, self._direction_thresh])


    def sobelx_low(self):
        return self._sobelx_low
    def sobelx_high(self):
        return self._sobelx_high
    def sobelx_filter(self):
        return self._sobelx_filter
    
    def sobely_low(self):
        return self._sobely_low
    def sobely_high(self):
        return self._sobely_high
    def sobely_filter(self):
        return self._sobely_filter
    
    def magn_low(self):
        return self._magn_low
    def magn_high(self):
        return self._magn_high
    def magn_filter(self):
        return self._magn_filter
    
    def direction_low(self):
        return self._direction_low
    def direction_high(self):
        return self._direction_high
    def direction_filter(self):
        return self._direction_filter
    def direction_avg_filter(self):
        return self._direction_avg_filter
    def direction_thresh(self):
        return self._direction_thresh
    
    def sobelxImage(self):
        return self._sobelx_binary
    def sobelyImage(self):
        return self._sobely_binary
    def magImage(self):
        return self._mag_binary
    def dirImage(self):
        return self._dir_binary
    def averageImg(self):
        return self._avg_img
    def thresholdImg(self):
        return self._thres_img
    def postAverageImg(self):
        return self._post_avg_img
    def postThresholdImg(self):
        return self._post_thres_img

    def setImage(self, img):
        self.image = img

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
        self.image = image
        self.orient = orient
        self.sobel_kernel = sobel_kernel
        self.thresh = thresh
        # Calculate directional gradient
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
        self.image = image
        self.magn_sobel_kernel = magn_sobel_kernel
        self.thresh_2 = thresh_2
        gray = self.extract_single_color(image)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=magn_sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=magn_sobel_kernel)
    #     magn = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        magn = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
        scaled_magn = np.uint8(255*magn/np.max(magn))
        # Apply threshold
        magn_binary = np.zeros_like(scaled_magn)       
        magn_binary[(scaled_magn > (thresh_2[0])) & (scaled_magn < thresh_2[1])] = 255
        return magn_binary
    
    def abs_dir_threshold(self, image, dir_sobel_kernel=3, dir_thresh=(-np.pi/2, np.pi/2)):
        self.image = image
        self.dir_sobel_kernel = dir_sobel_kernel
        self.dir_thresh = dir_thresh
        # Calculate gradient direction
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
        # non_binary= np.zeros_like(binary_image)
        # non_binary[(binary_image > 0)] = 255
        # binary_image.convertTo(binary_image,CV_8U, 1/255)
        # non_binary = binary_image.view('float32')
        # non_binary[:] = binary_image 

        np.set_printoptions(threshold=sys.maxsize)
        # print("binary_image: ", binary_image)
        non_binary = np.zeros_like(binary_image)
        non_binary[binary_image > 0] = 255
        non_binary[binary_image == 0] = 1
        # print("non_binary: ", non_binary)
        # non_binary = zeros

        output_image = cv2.blur(non_binary, (filter_size, filter_size))       
#         output_image = cv2.medianBlur(binary_image, filter_size)
        return output_image
    
    def abs_threshold(self, image, threshold):
        binary_image = np.zeros_like(image)
        binary_image[image > threshold] = 255
        return binary_image

    def _render(self, save_name="no_file_name"):

            
        crop_y_border = self.image.shape[0]//2 + 120

        image_1 = self.image.copy()
        image_2 = self.image.copy()
        image_1 = image_1[0:crop_y_border-1, 0:image_1.shape[1]]
        image_2 = image_2[crop_y_border:image_2.shape[0], 0:image_2.shape[1]]
        
        # cv2.imshow('image1', image_1)
        # cv2.waitKey(0)
        # cv2.imshow('image2', image_2)
        # cv2.waitKey(0)

        # image 1
        self._sobelx_binary = self.abs_sobel_thresh(image_1, 'x', self._sobelx_filter, (self._sobelx_low, self._sobelx_high))
        self._sobely_binary = self.abs_sobel_thresh(image_1, 'y', self._sobely_filter, (self._sobely_low, self._sobely_high))
        self._mag_binary = self.abs_magn_thresh(image_1, self._magn_filter, (self._magn_low, self._magn_high))
        self._dir_binary = self.abs_dir_threshold(image_1, self._direction_filter, (self._direction_low, self._direction_high))
        self._avg_img = self.abs_average(self._dir_binary, self._direction_avg_filter)
        self._thres_img = self.abs_threshold(self._avg_img, self._direction_thresh)
        self.combined = np.zeros_like(self._sobelx_binary)
        # self.combined[((self._sobelx_binary == 255) & (self._sobely_binary == 255)) | ((self._mag_binary == 255) & (self._thres_img == 255))] = 255
        # self.combined[((self._sobelx_binary == 255) & (self._sobely_binary == 255)) | ((self._thres_img == 255))] = 255
        self.combined[((self._sobelx_binary == 255) & (self._sobely_binary == 255)) | (self._mag_binary == 255) | (self._thres_img == 255)] = 255
        # self.combined[((self._sobelx_binary == 255) & (self._sobely_binary == 255)) | (self._mag_binary == 255)] = 255

        self._post_avg_img = self.abs_average(self.combined, self._post_avg_filter)
        self._post_thres_img = self.abs_threshold(self._post_avg_img, self._post_thresh)


        #  image 2
        _2_sobelx_filter = 3
        _2_sobelx_low = 18
        _2_sobelx_high = 255

        _2_sobely_filter = 3
        _2_sobely_low = 0
        _2_sobely_high = 255

        _2_magn_filter = 3
        _2_magn_low = 103
        _2_magn_high = 255

        _2_direction_filter = 15
        _2_direction_low = 227
        _2_direction_high = 287
        _2_direction_avg_filter = 11
        _2_direction_thresh = 255
        
        _2_post_avg_filter = 5
        _2_post_thresh = 158

        _2_sobelx_binary = self.abs_sobel_thresh(image_2, 'x', _2_sobelx_filter, (_2_sobelx_low, _2_sobelx_high))
        _2_sobely_binary = self.abs_sobel_thresh(image_2, 'y', _2_sobely_filter, (_2_sobely_low, _2_sobely_high))
        _2_mag_binary = self.abs_magn_thresh(image_2, _2_magn_filter, (_2_magn_low, _2_magn_high))
        _2_dir_binary = self.abs_dir_threshold(image_2, _2_direction_filter, (_2_direction_low, _2_direction_high))
        _2_avg_img = self.abs_average(_2_dir_binary, _2_direction_avg_filter)
        _2_thres_img = self.abs_threshold(_2_avg_img, _2_direction_thresh)
        _2_combined = np.zeros_like(_2_sobelx_binary)
        # self.combined[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | ((_2_mag_binary == 255) & (_2_thres_img == 255))] = 255
        # self.combined[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | ((_2_thres_img == 255))] = 255
        _2_combined[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | (_2_mag_binary == 255) | (_2_thres_img == 255)] = 255
        # self.combined[((_2_sobelx_binary == 255) & (_2_sobely_binary == 255)) | (_2_mag_binary == 255)] = 255

        _2_post_avg_img = self.abs_average(_2_combined, _2_post_avg_filter)
        _2_post_thres_img = self.abs_threshold(_2_post_avg_img, _2_post_thresh)

        # # concatenate both pictures
        concatenated_image = np.concatenate((self._post_thres_img, _2_post_thres_img), axis=0)
        # concatenated_image = _2_post_thres_img
        # concatenated_image =  self._post_thres_img

        if save_name == "no_file_name":
            cv2.imshow('sobelx_binary', self._sobelx_binary)
            cv2.imshow('sobely_binary', self._sobely_binary)
            cv2.imshow('mag_binary', self._mag_binary)
            cv2.imshow('direction_binary', self._dir_binary)
            cv2.imshow('direction_&_avg', self._avg_img)
            cv2.imshow('direction_&_avg_thresh', self._thres_img)
        self.color_binary = np.dstack(( np.zeros_like(self._sobelx_binary),((self._sobelx_binary == 255) & (self._sobely_binary == 255)), ((self._mag_binary == 255) & (self._thres_img == 255)))) * 255
        if save_name == "no_file_name":
            cv2.imshow('output', concatenated_image)
        else: 
            cv2.imwrite(f"test_output/{save_name}_output", concatenated_image)
        # cv2.imshow('output', self.color_binary)

    def save_params(self, var_list):
        with open("store_params/params_new",'wb') as f:
            pickle.dump(var_list, f)

    def load_params(self, param_file, var_list):
        with open(param_file, 'rb') as f:
            var_list = pickle.load(f)
        return var_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    # parser.add_argument('FILENAME')

    # args = parser.parse_args()
    WORKING_DIR = "/home/nbenndo/Documents/Programming/Udacity/SelfDrivingCarND/CarND-Advanced-Lane-Lines/"
    os.chdir(WORKING_DIR)
    FILENAME = 'test_images/frame30.jpg'

    IMG = cv2.imread(FILENAME)#, cv2.IMREAD_GRAYSCALE)

    cv2.imshow('input', IMG)

    param_finder = ParameterFinder(IMG, sobelx_filter=3, sobelx_low=16, sobelx_high=255, 
                                        sobely_filter=3, sobely_low=36, sobely_high=255, 
                                        magn_filter=3, magn_low=15, magn_high=255, 
                                        direction_filter=15, direction_low=229, direction_high=287,
                                        direction_avg_filter=11, direction_thresh=143)#, load_params_path="store_params/params_new")
    # calculate all images with last parameter
    os.chdir(f"{WORKING_DIR}/test_images")
    images_test = glob.glob('*.jpg', recursive=False)
    os.chdir(WORKING_DIR)
    for image_path in images_test:
        image = cv2.imread(f"test_images/{image_path}")
        param_finder.setImage(image)
        param_finder._render(image_path)


    # print("Edge parameters:")
    # print("GaussianBlur Filter Size: %f" % param_finder.filterSize())
    # print("Threshold1: %f" % param_finder.threshold1())
    # print("Threshold2: %f" % param_finder.threshold2())

    # (head, tail) = os.path.split(args.FILENAME)

    # (root, ext) = os.path.splitext(tail)

    # smoothed_filename = os.path.join("output_images", root + "-smoothed" + ext)
    # edge_filename = os.path.join("output_images", root + "-edges" + ext)

    # cv2.imwrite(smoothed_filename, param_finder.smoothedImage())
    # cv2.imwrite(edge_filename, param_finder.edgeImage())

    cv2.destroyAllWindows()