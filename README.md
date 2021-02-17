## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

[//]: # (Image References)

[image1]: ./writeup_images/orginal_chess.jpg "Orginal"
[image2]: ./writeup_images/Undistorted_chess.jpg "Undistorted"
[image3]: ./writeup_images/orginal_test_image.jpg "Orginal Test image"
[image4]: ./writeup_images/Undistorted_test_image.jpg "Undistorted Test image"
[image5]: ./writeup_images/prespective.jpg "veryfying src points"
[image6]: ./writeup_images/warped.jpg "warped Test image"
[image7]: ./writeup_images/binary_image.png "Binary Example"
[image8]: ./writeup_images/h_binary.PNG "H_channel Binary Example"
[image9]: ./writeup_images/histogram.PNG "Histogram Example"
[image10]: ./writeup_images/sliding_windows.PNG "sliding_windows Example"
[image11]: ./writeup_images/tracing_lines.PNG "Tracing lines Example"
[image12]: ./writeup_images/draw_line.PNG "draw lines Example"

[video1]: ./project_video_output.mp4 "Video"


### Camera Calibration

I started by preparing objpoints and imgpoints. The following steps were followed:

1.Grayscale the image
2.Find Chessboard Corners. It returns two values ret,corners. ret stores whether the corners were returned or not
3.If the corners were found, append corners to image points.Object points are same in this case bcause all images represent real chessboard.

With these steps we will be able to get image points and object points which will be required to calculate the camera calibration and distortion coefficients.

We call the calibrateCamera function which returns us a bunch of parameters, but the ones we are interested are the camera matrix (mtx) and distortion coefficient (dist).I've also saved the results in 'cam_calibration.p' pickle file for later use on our images.

We then use the distortion coefficient to undistort our image.

```python
##Calculating objpoints and imgpoints
objp = np.zeros((6*9,3), np.float32)
#mgrid returns coordinate value for given grid size and ill shape those coordinates back into 2 cols one for x an one for y.
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

for index,image in enumerate(images):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners  (for an 8x6 board)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points
    if ret == True:
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        objpoints.append(objp)
        #from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        ax2.imshow(img)
        ax2.set_title('Chessboard Corners Image', fontsize=30)
```

```python
#Calculating distortion coefficients

img = cv2.imread('camera_cal/calibration1.jpg')

img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

distortion_pickle = {}
distortion_pickle["mtx"] = mtx
distortion_pickle["dist"] = dist
pickle.dump( distortion_pickle, open( "cam_calibration.p", "wb" ) )

dst = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
```

![alt text][image1]    ![alt text][image2]


### Pipeline (single images)

From the results of previous steps, we can perform distortion correction to the test images using the function below:

```python
def undistort_images(img):
    distor_pickle = pickle.load( open( "cam_calibration.p", "rb" ) )
    mtx = distor_pickle["mtx"]
    dist = distor_pickle["dist"]
    return cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][image3]                ![alt text][image4]

#### Prespective transformation

The code for my perspective transform includes a function called `wraped_image()`.I chose the hardcode the source and destination points in the following manner:

```python
left=[150,720]
right=[1250,720] 
apex_left=[590,450] 
apex_right=[700,450] 

src=np.float32([left,apex_left,apex_right,right]) # Source Points for Image Warp
dst= np.float32([[200 ,720], [200  ,0], [980 ,0], [980 ,720]]) # Destination Points for Image Warp
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Using these points I got prespective transform Matrix to get bird's eye view of an image. The points are chosen with trail and error basis.

```python
M = cv2.getPerspectiveTransform(src, dst)
```

![alt text][image4]                     ![alt text][image5]                  ![alt text][image6]


#### Color transforms, gradients or other methods to create a thresholded binary image. 

I used a combination of color and gradient thresholds to generate a binary image.

The below function is used to convert image into various colorspaces and extract desired channel and the binary image is created by selecting the pixels within the threshold values.

```python
def colorspaces_binary(warped,colorspace,threshold,channel):
    color_img = cv2.cvtColor(warped, colorspace)
    channel_values = color_img[:,:,channel]
    binary_channel = np.zeros_like(channel_values)
    binary_channel[(channel_values > threshold[0]) & (channel_values <= threshold[1])] = 1
    return binary_channel
```
Below image is an example of H_channel and its respective binary image.
![alt text][image8]

I used below functions to get binary image with required thresholds. 

1.Sobel x-gradient.

```python
def abs_sobel_thresh(img, orient='x',sobel_kernel=3,thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return sxbinary
```

2. Applied a threshold to the overall magnitude of the gradient, in both x and y which is nothing but the square root of the squares of the individual x and y gradients.

```python
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):  
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    return sxbinary
```

3. The gradient magnitude picks up the lane lines well, but with a lot of other stuff detected too.In the case of lane lines, we're interested only in edges of a particular orientation. So now we will calculated the direction, or orientation, of the gradient.The direction of the gradient is simply the inverse tangent (arctangent) of the yy gradient divided by the xx gradient.

```python
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    sxbinary = np.zeros_like(arctan)
    sxbinary[(arctan > thresh[0]) & (arctan < thresh[1])] = 1  
    return sxbinary
```
When tested seperately the images where certain places had more lightining, the lane lines where not extracted properly. After certain trail and error I found optimal combination to get the desired binary image like the example below.

```python
combined_binary = np.zeros_like(S_binary)
combined_binary[((S_binary==1)&(L_binary==1))| (gradx == 1)]=1
```

![alt text][image6]              ![alt text][image7]

I have also created a function putting all the above preprocessing functions, which takes original raw images and then undistort the image and wrap the image and finally provide the combined binary image after combining color transforms and gradients.The function also returns Minv which is inverse of prespective transform which we will use later for unwraping the image.

```python
def preprocessing_image(img):
    undistorted_image = undistort_images(img)
    warped_img,M,Minv,roi = wraped_image(undistorted_image)
    ksize = 3
    # Applying the thresholding functions
    gradx = abs_sobel_thresh(warped_img, orient='x', sobel_kernel=ksize, thresh=(10,150))
    #colorspaces
    threshold = [100,255]
    L_binary = colorspaces_binary(warped_img,cv2.COLOR_RGB2HLS,threshold,1)
    S_binary = colorspaces_binary(warped_img,cv2.COLOR_RGB2HLS,threshold,2)
    #combining colorspaces and sobel threshold gradients
    combined_binary = np.zeros_like(S_binary)
    combined_binary[((S_binary==1)&(L_binary==1))| (gradx == 1)]=1
    return combined_binary,Minv
```

#### Identifying lane-line pixels and fit their positions with a polynomial

Historgram for the bottom half of the image where the lane lines are likely to be mostly vertical as they are nearest to car is drawn so that we can distinguish the left lane pixels and right lane pixels.

```python
def Histogram(binary_image):
    bottom_half = binary_image[binary_image.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    return histogram
```
When plotted:

![alt text][image7]      ![alt text][image9]

The next step is to apply sliding windows function to left and right lane pixels obtained from histogram function.

1. The left and right base points are calculated from the histogram
2. I then calculated the position of all non zero x and non zero y pixels.
3. I then Started iterating over the windows where I identify window boundaries in x and y (and right and left)
4. Later we calculate four  boundaries of the window.
5. Later we Identify the nonzero pixels in x and y within the window.
6. We then collect all the indices in the list and decide the center of next window using these points
7. After above step, we seperate the points to left and right positions
8. We then fit a second degree polynomial using np.polyfit for left and right points.

```python
def sliding_windows(binary_image):
    histogram = Histogram(binary_image)
    out_img = np.dstack((binary_image, binary_image, binary_image))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    window_height = np.int(binary_image.shape[0]//nwindows)
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_image.shape[0] - (window+1)*window_height
        win_y_high = binary_image.shape[0] - window*window_height
        ###  four  boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ###Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                        & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                        & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
        
    return left_fit,right_fit,left_lane_inds,right_lane_inds,out_img,leftx, lefty, rightx, righty
```

![alt text][image10]     ![alt text][image11]



#### Determine the curvature of the lane and vehicle position with respect to center.

1. First we define values to convert pixels to meters
2. Plot the left and right lines
3. Calculate the curvature from left and right lanes seperately
4. Mean of the values obtained from step 3 is radius of curvature in meters. 
5. We take the mean of the left bottom most point of the left lane and right bottom most point of the right lane and then subtract it from the center of the car to get the deviation from the center wich gives us the vehicle position with respect to center.

```python
def measure_curvature(binary_image,left_fit,right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    car_position= binary_image.shape[1]/2
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    #####Implementing the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+((2*left_fit[0]*y_eval)+left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])  ## the calculation of the left line 
    right_curverad = ((1+((2*right_fit[0]*y_eval)+right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])  ## the calculation of the right line 
    
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    actual_position= (left_lane_bottom+ right_lane_bottom)/2
    
    distance= (car_position - actual_position)* xm_per_pix
    
    # Now our radius of curvature is in meters
    return (left_curverad + right_curverad)/2, distance
```   
    

####  Final result plotted back down onto the road such that the lane area is identified clearly.

1. Create wraped balnk image to draw the lines.
2. Recast the x an y points into usable format for cv2.fillPoly()
3. Lanes are drawn on wraped blank image.
4. Warp the blank back to original image space using inverse perspective matrix (Minv)
5. Combine the result with the original image.

```python
def draw_lines(orginal_image,binary_image,left_fit,right_fit,Minv):
    #image to draw lines 
    warp_zero = np.zeros_like(binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    #drawing lane on wraped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    h,w = binary_image.shape
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(orginal_image, 1, newwarp, 0.5, 0)
    return result
```

![alt text][image12]

---

### Pipeline (video)

Below is the complete pipeline to draw lines on the test images as well as the videos. 

```python
def pipeline(original_image):
    binary_image,Minv = preprocessing_image(original_image)
    left_fit,right_fit,left_lane_inds,right_lane_inds,out_img,leftx, lefty, rightx, righty=sliding_windows(binary_image)
    line_output = draw_lines(original_image,binary_image,left_fit,right_fit,Minv)
    radius,distance = measure_curvature(binary_image,left_fit,right_fit)
    cv2.putText(line_output,"Radius of Curvature is " + str(int(radius))+ "m", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0),2)
    cv2.putText(line_output,"Distance from center is {:2f}".format(distance)+ "m", (100,150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0),2)
    return line_output
```
Applying the pipeline on the video.

```python
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time video_clip.write_videofile(video_output, audio=False)
```
Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

The source and destination points for prespective trasfomation is hard coded after trial and error.

The combination of color channel and Sobel threshold gradients results did not work for few conditions. Again right combination is obtained by trial and error.

The pipeline fails on both the challenge video which i will work on further to make it work. The pipeline on mountain terrain condition will also fail.



