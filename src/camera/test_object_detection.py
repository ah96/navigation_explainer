#!/usr/bin/env python3

import message_filters
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
import pandas as pd
import time
import torch
import os

dirCurr = os.getcwd()
path_prefix = dirCurr + '/yolo_data/'

print('torch.cuda.is_available() = ',torch.cuda.is_available())

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


def yolov3(image):
    try:
        # Load YOLOv3 model
        #net = cv2.dnn.readNet(path_prefix + "yolov3.weights", path_prefix + "yolov3.cfg")
        net = cv2.dnn.readNet(path_prefix + "weights/yolov3-tiny.weights", path_prefix + "cfg/yolov3-tiny.cfg")

        # load the global list of labels
        labels_global = np.array(pd.read_csv(path_prefix + 'datasets/labels_coco.csv'))

        start = time.time()
        # Define the neural network input
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Perform forward propagation
        output_layer_name = net.getUnconnectedOutLayersNames()
        output_layers = net.forward(output_layer_name)
        end = time.time()
        print('yolov3 runtime: ', end-start)

        # Get image dimensions
        (height, width) = image.shape[:2]

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
        cv2.imwrite(path_prefix + 'result.png', image)

    except Exception as e:
        print(str(e))

def yolov5(image):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
    #model = torch.hub.load(path_prefix + 'yolov5_master/', 'custom', path_prefix + 'models/best.pt', source='local')  # custom trained model

    start = time.time()
    # Inference
    results = model(image)
    end = time.time()
    print('yolov5 runtime: ', end-start)
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #results.show()
    results.save()
    #results.crop()
    #results.pandas() #[xmin ymin xmax ymax confidence class name]
    #print('type(results) = ', type(results))

    res = np.array(results.pandas().xyxy[0])
    #print(res)
    print(results.pandas().xyxy[0])

    labels = list((res[:,-1]))
    print('labels: ', labels)

    # creating and plotting textual explanation
    explanation = ' '
    if len(labels) > 2:
        explanation = 'I detect ' + ', '.join(labels[:-1]) + ', and ' + labels[-1] + '.'
        print(explanation)
    elif len(labels) == 2:
        explanation = 'I detect ' + labels[-2] + ' and ' + labels[-1] + '.'
        print(explanation)
    elif len(labels) == 1:
        explanation = 'I detect ' + labels[-1] + '.'
        print(explanation)

def yolov7(image):
    # Load fine-tuned custom model
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_prefix + 'models/yolov7-d6.pt')
    
    start = time.time()
    # Inference
    results = model(image)
    end = time.time()
    print('yolov7 runtime: ', end-start)
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #results.show()
    results.save()
    #results.crop()
    #results.pandas() #[xmin ymin xmax ymax confidence class name]
    #print('type(results) = ', type(results))

    #res = np.array(results.pandas().xyxy[0])
    #print(res)
    print(results.pandas().xyxy[0])


#image = cv2.imread(path_prefix + "images/indo.jpg")
#image = cv2.imread(path_prefix + "images/icml1.jpg")
#image = cv2.imread(path_prefix + "images/icml2.jpg")
#image = cv2.imread(path_prefix + "images/ki.png")
image = cv2.imread(path_prefix + "images/door.png")

#yolov3(image)
yolov5(image)
#yolov7(image)
