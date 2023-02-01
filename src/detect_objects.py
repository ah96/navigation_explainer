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
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
model = torch.hub.load('/home/robolab/.cache/torch/hub/ultralytics_yolov5_master/', 'custom', dirCurr + '/yolov5s.pt', source='local')  # custom trained model

# load the global list of labels
#labels_global = np.array(pd.read_csv('./yolo_data/labels_coco.csv'))

image = []
callback_ctr = 0

fig = plt.figure(frameon=False)
w = 1.6 * 3
h = 1.6 * 3
fig.set_size_inches(w, h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# callback function
def callback(img):
    # these two links should help:
    # 1. this procedure done with yolov3: https://medium.com/@mkadric/how-to-use-yolo-object-detection-model-1604cf9bbaed 
    # 2. yolov5 detect.py file from where you should be able to derive which input image format yolov5 needs, how to call feedforward/predict function, etc.: https://github.com/ultralytics/yolov5/blob/master/detect.py 

    global callback_ctr, image

    print('\ncallback')

    # image from robot's camera to np.array
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
    #depth_image = np.frombuffer(depth_img.data, dtype=np.uint8).reshape(depth_img.height, depth_img.width, -1)

    #image = cv2.imread("./yolo_data/slika1.jpg")

    # Get image dimensions
    #(height, width) = image.shape[:2]

    #depths_avg = [] #[0.0]*detections.shape[0]
    #print('depths_avg = ', depths_avg)

    callback_ctr = callback_ctr + 1
    if callback_ctr == 1:
        callback_ctr = 0
        yolo()

def yolo():
    start = time.time()
    # Inference
    results = model(image)
    end = time.time()
    print('yolov5 runtime: ', end-start)
    # Results
    #results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #print('type(results.show()) = ', type(results.show()))
    #results.save()
    #results.crop()
    results.show()
    #results.pandas() #[xmin ymin xmax ymax confidence class name]
    #print('type(results) = ', type(results))

    res = np.array(results.pandas().xyxy[0])
    #print(res)
    print(results.pandas().xyxy[0])

    labels = list((res[:,-1]))
    print('labels: ', labels)

    confidences = list((res[:,-3]))
    print('confidences: ', confidences)

    #data = results.ims[0]
    #data.swapaxes(0, 2)
    #cv2.imshow('Results', results.ims[0].astype(np.uint8))
    #print(results.ims[0].shape)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    #ax.imshow(data.astype('uint8'), aspect='auto')
    #plt.show()
    #fig.savefig(dirCurr + '/' + 'results' + '.png', transparent=False)

    labels_ = []
    for i in range(len(labels)):
        if confidences[i] >= 0.4:
            labels_.append(labels[i])

    labels=labels_ 
    
    explanation = ' '
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

if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)

    rospy.Subscriber('/xtion/rgb/image_raw', Image, callback)
    test_exp_pub = rospy.Publisher('/text_exps', String, queue_size=1)
    
    #image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
    #info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
    #depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image) #"32FC1"
    #ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], 10, 0.2)
    #ts.registerCallback(callback)
    
    rospy.spin()