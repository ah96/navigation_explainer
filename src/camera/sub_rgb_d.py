#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import os
from PIL import Image as PImg
import pandas as pd

img_ctr = 0

# directory
dirCurr = os.getcwd()

create_dataset = True
dirMain = 'camera_feed_dataset'
try:
    os.mkdir(dirMain)
except FileExistsError:
    pass

# save the RGB image from the robot's camera as .png
def saveImage(image):
    global img_ctr
    # save image as .png
    cv2.imwrite(dirMain + '/image_' + str(img_ctr) + '.png', image)
    img_ctr += 1

# callback function
# get the RGB image from the robot's camera
def image_raw_callback(img):
    print('\n' + str(img_ctr))

    # in the case if the image must be undistorted
    #camera_info_K = np.array(camera_info.K).reshape([3, 3])
    #camera_info_D = np.array(camera_info.D)
    #rgb_undist = cv2.undistort(rgb_cv2, camera_info_K, camera_info_D)
    
    # get image from the topic and convert it to cv2 format
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)

    image = image[...,::-1].copy()

    # save image for dataset
    saveImage(image)

# callback function
# get the depth image from the robot's camera
def image_depth_callback(img):
    print('\nimage_depth_callback')

    #print('img.height = ', img.height)
    #print('img.width = ', img.width)
    #print('img.encoding = ', img.encoding)
    #print('img.is_bigendian = ', img.is_bigendian)
    #print('img.step = ', img.step)

    # convert depth image to np array
    depth_image = np.frombuffer(img.data, dtype=np.float32).reshape(img.height, img.width, -1)
    
    # fill missing values with negative distance
    depth_image = np.nan_to_num(depth_image, nan=-1.0)

    # print image shape
    #print('depth_image.shape: ', depth_image.shape)

    # save depth image as png
    #cv2.imwrite("depth_image.png", np.uint8(depth_image[:,:,0]))

    # save depth image as csv
    #pd.DataFrame(depth_image[:,:,0]).to_csv("depth_image.csv")

    # save image for dataset
    saveImage(depth_image)

if __name__ == '__main__':
    rospy.init_node('camera_feed', anonymous=True)
    image_raw_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, image_raw_callback)
    image_depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, image_depth_callback) #"32FC1"
    rospy.spin()