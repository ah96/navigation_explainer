#!/usr/bin/env python3

import message_filters
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import pandas as pd
import time
import torch

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Load YOLO model
#net = cv2.dnn.readNet("./yolo_data/yolov3.weights", "./yolo_data/yolov3.cfg")
net = cv2.dnn.readNet("./yolo_data/yolov3-tiny.weights", "./yolo_data/yolov3-tiny.cfg")

# load the global list of labels
labels_global = np.array(pd.read_csv('./yolo_data/labels_coco.csv'))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# callback function
def callback(img, camera_info):
    # these two links should help:
    # 1. this procedure done with yolov3: https://medium.com/@mkadric/how-to-use-yolo-object-detection-model-1604cf9bbaed 
    # 2. yolov5 detect.py file from where you should be able to derive which input image format yolov5 needs, how to call feedforward/predict function, etc.: https://github.com/ultralytics/yolov5/blob/master/detect.py 

    print('\ncallback')

    # image from robot's camera to np.array
    image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)

    image = cv2.imread("./yolo_data/image.png")

    # Get image dimensions
    (height, width) = image.shape[:2]

    start = time.time()
    # Define the neural network input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward propagation
    output_layer_name = net.getUnconnectedOutLayersNames()
    output_layers = net.forward(output_layer_name)
    end = time.time()
    print('yolov3 runtime: ', end-start)

    start = time.time()
    # Inference
    results = model(image)
    end = time.time()
    print('yolov5 runtime: ', end-start)
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.save()

    # Initialize list of detected objects and their labels
    detections = []
    labels = []

    # Loop over the output layers
    for output in output_layers:
        # Loop over the detections
        for detection in output:
            # Extract the class ID and confidence, as well as label of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = labels_global[class_id][0]

            # Only keep detections with a high confidence
            if confidence >= 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Only add the detection to the list of detections, if it is not a duplicate detection
                duplicate = False
                
                ctr = 0
                for (xi, yi, wi, hi) in detections:
                    if label == labels[ctr]:
                        boxA = (x, y, x + w, y + h)
                        boxB = (xi, yi, xi + wi, yi + hi)
                        iou = bb_intersection_over_union(boxA, boxB)
                        if iou >= 0.4:
                            duplicate = True
                            break
                    ctr += 1
                
                if duplicate == False:
                    detections.append((x, y, w, h))
                    labels.append(label)

    # Draw bounding boxes around the detections
    ctr = 0
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, labels[ctr], (x, y + 30), 0, 2, (0, 255, 0), 3)
        ctr += 1

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite('./yolo_data/result.png', image)

if __name__ == '__main__':
    rospy.init_node('my_node', anonymous=True)
    image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
    info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], 10, 0.2)
    ts.registerCallback(callback)
    rospy.spin()