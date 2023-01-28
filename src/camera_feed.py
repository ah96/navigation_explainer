#!/usr/bin/env python2.7

import message_filters
import cv2
import rospy
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import darknet as dn

# callback function
def callback(rgb_msg, camera_info):
    print('usao u callback')
    # get image from the topic and convert it to cv2 format
    rgb_cv2 = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
    
    # in the case if the image must be undistorted before sending it as an input to the yolo
    #camera_info_K = np.array(camera_info.K).reshape([3, 3])
    #camera_info_D = np.array(camera_info.D)
    #rgb_undist = cv2.undistort(rgb_cv2, camera_info_K, camera_info_D)
    
    # save image as .png
    cv2.imwrite('camera_feed.png', rgb_cv2)

    dn.set_gpu(0)
    net = dn.load_net("darknet/cfg/yolov3-tiny.cfg", "darknet/yolov3-tiny.weights", 0)
    meta = dn.load_meta("darknet/cfg/coco.data")
    r = dn.detect(net, meta, "data/person.jpg")
    print(r)

    # convert cv2 image to numpy array - may be needed
    #rgb_np = np.asarray(rgb_cv2)
    #print(rgb_np.shape)

    # follow the next steps to detect objects on the image:
    # 1. convert rgb_cv2 to the right input format for the trained yolo model
    # 2. call the function to do a feedforward propagation through the trained model and to get predictions/labels
    # 3. collect labels and draw them as bounding boxes on the original input image (rgb_cv2 or its numpy version)
    # these two links should help:
    # 1. this procedure done with yolov3: https://medium.com/@mkadric/how-to-use-yolo-object-detection-model-1604cf9bbaed 
    # 2. yolov5 detect.py file from where you should be able to derive which input image format yolov5 needs, how to call feedforward/predict function, etc.: https://github.com/ultralytics/yolov5/blob/master/detect.py 

    # when you form a image with bounding boxes you should save it or publish it to some topic or show it on the separate screen during navigation
    # for the start you can find some random pictures of the doors online, download them, and use them as input for your yolo model

if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)
    image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
    info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], 1, 1.5)
    ts.registerCallback(callback)
    rospy.spin()