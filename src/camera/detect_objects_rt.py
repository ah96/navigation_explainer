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
def callback(img):

    global callback_ctr, yolo_ctr

    print('\ncallback_' + str(callback_ctr))

    # RGB IMAGE
    # image from robot's camera to np.array
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
    #image = np.array(cv2.imread(path_prefix + "/images/icml1.jpg"))
    # Get image dimensions
    #(height, width) = image.shape[:2]

    callback_ctr = callback_ctr + 1
    yolo_ctr = yolo_ctr + 1
    yolo(image)

# yolo object detection
def yolo(image):
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

    # plot with detected objects and labels
    # with cv2
    res = np.array(image)

    for i in range(0, len(labels)):
        #print('i, label:', (i, labels[i]))    
    
        p2, p1 = (int(x_y_min_max[i,2]), int(x_y_min_max[i,3])), (int(x_y_min_max[i,0]), int(x_y_min_max[i,1]))
        res = cv2.rectangle(res, p1, p2, (0,255,0), thickness=6, lineType=cv2.LINE_AA)
    
        if p2[1] < 10:
            res = cv2.putText(res, labels[i], (p2[0], p2[1]+40), 0, 2, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
        else:
            res = cv2.putText(res, labels[i], (p2[0], p2[1]-10), 0, 2, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
    
    #cv2.imwrite(dirCurr + '/' + 'results_' + str(results_ctr) + '.png', image)
    cv2.imshow("Image Window", res)
    cv2.waitKey(1)

    results_ctr += 1

    end = time.time()
    print('yolov5 total runtime: ' + str(round(1000*(end-start), 2)) + ' ms')    


if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)

    image_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, callback)
    
    rospy.spin()