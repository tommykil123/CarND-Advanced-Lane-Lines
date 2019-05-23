## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)


# Project Overview

### Step 0: My Pipeline

My pipeline consisted of 10 steps. 
    1. Obtain distorted camera images
    2. Calibrate and Undistort camera images
    3. Transform/warp the camera images to project same to dimensions
    4. Export lane images
    5. Undistort the lane images
    6. Apply thresholds (Sobel X, Sobel Y, Sobely XY, Gradient, and HLS)
    7. Find the region of interest (ROI)
    8. Warp binary image
    9. Apply histogram filter and moving window to find lanes
    10. Find curvatures, offsets of the image
    
This pipeline has been applied to both images and videos

# Camera Calibration

### Step 1: Camera Image
![Camera Image](1_camera_cal/calibration20.jpg)

### Step 2: Calibrate and Undistort Raw Image
![Undistort Image](2_camera_undist/undist20.jpg)

### Step 3: Transform/Warp Image
![Warpcamera Image](3_camera_transform/transform20.jpg)


# Lane Line Images

### Step 4: Lane Line Image
![Laneline Image](4_test_images/straight_lines1.jpg)

### Step 5: Undistort Lane Line Image
![Undistort Laneline Image](5_test_images_undist/straight_lines1_undist.jpg)

### Step 6: Apply thresholds (Sobel, Gradient, HLS, etc)
![Sobelx Image](6A_test_images_sobelx/straight_lines1_undist_sobelx.jpg)
![Sobely Image](6B_test_images_sobely/straight_lines1_undist_sobely.jpg)
![Sobelxy Image](6C_test_images_sobelxy/straight_lines1_undist_sobelxy.jpg)
![Gradient Image](6D_test_images_gradient/straight_lines1_undist_gradient.jpg)
![Combined Sobel Image](6E_test_images_combinedthr/straight_lines1_undist_combintedthr.jpg)

![HLS Image](6F_test_images_hls/straight_lines1_undist_hls.jpg)

![Sobel HLS Image](6G_test_images_combined_all/straight_lines1_undist_combinedall.jpg)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



