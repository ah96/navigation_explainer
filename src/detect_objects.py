#!/usr/bin/env python3

import message_filters
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import pandas as pd
import time
import torch
import os
import cv2
from std_msgs.msg import String
from matplotlib import pyplot as plt
from PIL import Image as PImage

dirCurr = os.getcwd()
path_prefix = dirCurr + '/yolo_data/'

print('torch.cuda.is_available() = ',torch.cuda.is_available())

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
#model = torch.hub.load(path_prefix + '/yolov5_master/', 'custom', path_prefix + '/models/yolov5s.pt', source='local')  # custom trained model

# some global variables
image = []
callback_ctr = 0
yolo_ctr = 0
results_ctr = 0

# callback function
def callback1(img, depth_img, info):
    # these two links should help:
    # 1. this procedure done with yolov3: https://medium.com/@mkadric/how-to-use-yolo-object-detection-model-1604cf9bbaed 
    # 2. yolov5 detect.py file from where you should be able to derive which input image format yolov5 needs, how to call feedforward/predict function, etc.: https://github.com/ultralytics/yolov5/blob/master/detect.py 

    global callback_ctr, yolo_ctr

    print('\ncallback_' + str(callback_ctr))

    # RGB IMAGE
    # image from robot's camera to np.array
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
    #image = np.array(cv2.imread(path_prefix + "/images/icml1.jpg"))
    # Get image dimensions
    #(height, width) = image.shape[:2]

    # DEPTH IMAGE
    # convert depth image to np array
    depth_image = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_img.height, depth_img.width, -1)    
    # fill missing values with negative distance
    depth_image = np.nan_to_num(depth_image, nan=-1.0)
    
    # get projection matrix
    P = info.P

    callback_ctr = callback_ctr + 1
    yolo_ctr = yolo_ctr + 1
    if yolo_ctr == 1:
        yolo_ctr = 0
        yolo(image, depth_image, P)

# callback function
def callback2(img):
    # these two links should help:
    # 1. this procedure done with yolov3: https://medium.com/@mkadric/how-to-use-yolo-object-detection-model-1604cf9bbaed 
    # 2. yolov5 detect.py file from where you should be able to derive which input image format yolov5 needs, how to call feedforward/predict function, etc.: https://github.com/ultralytics/yolov5/blob/master/detect.py 

    global callback_ctr, image

    print('\nimage_callback')

    # image from robot's camera to np.array
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
    #image = np.array(cv2.imread(path_prefix + "/images/icml1.jpg"))
    
    # Get image dimensions
    #(height, width) = image.shape[:2]

    callback_ctr = callback_ctr + 1
    if callback_ctr == 1:
        callback_ctr = 0
        yolo(image, np.empty((0,0)))

# callback function
def callback3(msg):
    global callback_ctr, image

    print('\ncamera_info_callback')

    '''
    std_msgs/Header header
    uint32 height
    uint32 width
    string distortion_model
    float64[] D
    float64[9] K
    float64[9] R
    float64[12] P
    uint32 binning_x
    uint32 binning_y
    sensor_msgs/RegionOfInterest roi
    '''

    print('msg.height = ', msg.height)
    print('msg.height = ', msg.width)
    print('msg.height = ', msg.distortion_model)
    print('msg.height = ', msg.D)
    print('msg.height = ', msg.K)
    print('msg.height = ', msg.R)
    print('msg.height = ', msg.P)
    print('msg.height = ', msg.binning_x)
    print('msg.height = ', msg.binning_y)
    print('msg.height = ', msg.roi)

# yolo object detection
def yolo(image, depth_image, P):
    global results_ctr

    start = time.time()
    
    # Inference
    results = model(image)
    end = time.time()
    print('yolov5 inference runtime: ' + str(round(1000*(end-start), 2)) + ' ms')
    
    # Results
    #results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #results.pandas() #[xmin ymin xmax ymax confidence class name]
    res = np.array(results.pandas().xyxy[0])
    #print(results.pandas().xyxy[0])

    # labels, confidences and bounding boxes
    labels = list(res[:,-1])
    #print('original_labels: ', labels)
    confidences = list((res[:,-3]))
    #print('confidences: ', confidences)
    x_y_min_max = np.array(res[:,0:4])
    
    labels_ = []
    for i in range(len(labels)):
        if confidences[i] < 0.0:
            np.delete(x_y_min_max, i, 0)
        else:
            labels_.append(labels[i])
    labels = labels_
    #print('filtered_labels: ', labels)

    # creating and plotting textual explanation
    get_3D = False
    coordinates_3d = []
    depths_bool = False
    if depth_image.size != 0:
        depths_bool = True
    explanation = ' '
    if depths_bool == False:
        if len(labels) > 2:
            explanation = 'I detect ' + ', '.join(labels[:-1]) + ', and ' + labels[-1] + '.'
            print(explanation)
            test_exp_pub.publish(explanation)
        elif len(labels) == 2:
            explanation = 'I detect ' + labels[-2] + ' and ' + labels[-1] + '.'
            print(explanation)
            test_exp_pub.publish(explanation)
        elif len(labels) == 1:
            explanation = 'I detect ' + labels[-1] + '.'
            print(explanation)
            test_exp_pub.publish(explanation)
    else:
        fx = P[0]
        cx = P[2]
        fy = P[5]
        cy = P[6]
        if len(labels) > 2:
            explanation += 'I detect '
            for i in range(0, len(labels)):
                u = int((x_y_min_max[i,0]+x_y_min_max[i,2])/2)
                v = int((x_y_min_max[i,1]+x_y_min_max[i,3])/2)
                depth = depth_image[v, u][0]
                if i != len(labels) - 1:
                    explanation += labels[i] + ' at ' + str(depth) + ' meters, '
                else:
                    explanation += 'and ' + labels[i] + ' at ' + str(depth) + ' meters' + '.'
                if get_3D == True:
                    # get the 3D positions
                    z = depth
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    centroid = [x,y,z]
                    u = x_y_min_max[0,0]
                    v = x_y_min_max[0,1]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    top_left = [x,y,z]
                    u = x_y_min_max[0,2]
                    v = x_y_min_max[0,3]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    bottom_right = [x,y,z]
                    coordinates_3d.append([top_left, centroid, bottom_right])
            print(explanation)
            test_exp_pub.publish(explanation)
        elif len(labels) == 2:
            depths = []
            for i in range(0, 2):
                u = int((x_y_min_max[i,0]+x_y_min_max[i,2])/2)
                v = int((x_y_min_max[i,1]+x_y_min_max[i,3])/2)
                depth = depth_image[v, u][0]
                depths.append(depth)
                if get_3D == True:
                    # get the 3D positions
                    z = depth
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    centroid = [x,y,z]
                    u = x_y_min_max[0,0]
                    v = x_y_min_max[0,1]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    top_left = [x,y,z]
                    u = x_y_min_max[0,2]
                    v = x_y_min_max[0,3]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    bottom_right = [x,y,z]
                    coordinates_3d.append([top_left, centroid, bottom_right])
            explanation = 'I detect ' + labels[-2] + ' at ' + str(depths[-2]) + ' meters and ' + labels[-1] + ' at ' + str(depths[-1]) + ' meters' + '.'
            print(explanation)
            test_exp_pub.publish(explanation)
        elif len(labels) == 1:
            u = int((x_y_min_max[0,0]+x_y_min_max[0,2])/2)
            v = int((x_y_min_max[0,1]+x_y_min_max[0,3])/2)
            depth = depth_image[v, u][0]
            explanation = 'I detect ' + labels[-1] + ' at ' + str(depth) + ' meters.'
            print(explanation)
            test_exp_pub.publish(explanation)
            if get_3D == True:
                # get the 3D positions
                z = depth
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                centroid = [x,y,z]
                u = x_y_min_max[0,0]
                v = x_y_min_max[0,1]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                top_left = [x,y,z]
                u = x_y_min_max[0,2]
                v = x_y_min_max[0,3]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                bottom_right = [x,y,z]
                coordinates_3d.append([top_left, centroid, bottom_right])


    # plot with detected objects and labels
    # with cv2
    for i in range(0, len(labels)):    
        p1, p2 = (int(x_y_min_max[i,2]), int(x_y_min_max[i,3])), (int(x_y_min_max[i,0]), int(x_y_min_max[i,1]))
        cv2.rectangle(image, p1, p2, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
        if p2[1] < 10:
            cv2.putText(image, labels[i], (p2[0], p2[1]+40), 0, 2, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.putText(image, labels[i], (p2[0], p2[1]-10), 0, 2, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
    #cv2.imwrite(dirCurr + '/' + 'results_' + str(results_ctr) + '.png', image)
    cv2.imshow("Image Window", image)
    cv2.waitKey(1)

    results_ctr += 1

    end = time.time()
    print('yolov5 total runtime: ' + str(round(1000*(end-start), 2)) + ' ms')    

if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)

    # image subscriber
    #rospy.Subscriber('/xtion/rgb/image_raw', Image, callback2)

    # camera_info subscriber
    #rospy.Subscriber('/xtion/rgb/camera_info', CameraInfo, callback3)
    
    # text explanation publisher
    test_exp_pub = rospy.Publisher('/text_exps', String, queue_size=1)
    
    image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
    depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image) #"32FC1"
    info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, info_sub], 10, 1.0)
    ts.registerCallback(callback1)
    
    rospy.spin()