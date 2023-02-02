#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import os

img_ctr = 0

# directory
dirCurr = os.getcwd()

dirMain = 'camera_feed_dataset'
try:
    os.mkdir(dirMain)
except FileExistsError:
    pass

# callback function
def callback(img):
    print('\ncallback')

    global img_ctr

    # in the case if the image must be undistorted before sending it as an input to the yolo
    #camera_info_K = np.array(camera_info.K).reshape([3, 3])
    #camera_info_D = np.array(camera_info.D)
    #rgb_undist = cv2.undistort(rgb_cv2, camera_info_K, camera_info_D)
    
    # get image from the topic and convert it to cv2 format
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        
    # save image as .png
    cv2.imwrite(dirMain + '/image_' + str(img_ctr) + '.png', image)

    img_ctr += 1

if __name__ == '__main__':
    rospy.init_node('camera_feed', anonymous=True)
    image_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, callback)
    rospy.spin()