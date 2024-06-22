#!/usr/bin/env python3

import message_filters
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import os
import pandas as pd

img_ctr = 0
pos = []
orient = []

# directory
dirCurr = os.getcwd()

create_dataset = True
dirMain = 'camera_feed_dataset'
try:
    os.mkdir(dirMain)
except FileExistsError:
    pass

def camera_callback(rgb_msg, depth_msg):
    global img_ctr, pos, orient

    # RGB
    # get image from the topic and convert it to cv2 format
    rgb_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)

    # DEPTH
    # convert depth image to np array
    depth_image = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
    
    # fill missing values with negative distance
    depth_image = np.nan_to_num(depth_image, nan=-1.0)

    # CAMERA INFO
    #camera_info_K = np.array(camera_info.K).reshape([3, 3])
    #camera_info_D = np.array(camera_info.D)
    #rgb_undist = cv2.undistort(rgb_image, camera_info_K, camera_info_D)

    cv2.imwrite(dirMain + '/rgb_' + str(img_ctr) + '.png', rgb_image)
    # save depth image as png
    cv2.imwrite(dirMain + '/depth_' + str(img_ctr) + '.png', np.uint8(depth_image[:,:,0]))
    # save depth image as csv
    pd.DataFrame(depth_image[:,:,0]).to_csv(dirMain + '/depth_' + str(img_ctr) + '.csv')
    #cv2.imwrite(dirMain + '/rgb_undist_' + str(img_ctr) + '.png', rgb_undist)
    #camera_info_K.tofile(dirMain + '/K_' + str(img_ctr) + '.csv', sep=',',format='%10.5f')
    #camera_info_D.tofile(dirMain + '/D_' + str(img_ctr) + '.csv', sep=',',format='%10.5f')
    #np.array(depth_image).tofile(dirMain + '/depth_' + str(img_ctr) + '.csv', sep=',',format='%10.5f')
    #np.array(pos).tofile(dirMain + '/position_' + str(img_ctr) + '.csv', sep=',',format='%10.5f')
    #np.array(orient).tofile(dirMain + '/quaternion_' + str(img_ctr) + '.csv', sep=',',format='%10.5f')
    #print('\n' + str(img_ctr))

    img_ctr += 1

def pose_callback(pose_msg):
    global pos, orient
    print('...')
    pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
    orient = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]

if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)
    image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
    depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image)
    #info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 1.0)
    ts.registerCallback(camera_callback)

    #sub = rospy.Subscriber ('/amcl_pose', PoseWithCovarianceStamped, pose_callback) 
    rospy.spin()

# rosbag record rosout tf amcl_pose xtion/rgb/camera_info xtion/rgb/image_raw /xtion/depth_registered/image_raw