#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import numpy as np

import tf2_ros
#from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#import cv2
import copy

from data.base_dataset import get_params, get_transform
import PIL.Image
#import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

import time


global odom_x, odom_y, local_plan_xs, local_plan_ys
global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image 
global br, pub_exp_image, pub_exp_pointcloud
global global_plan_xs, global_plan_ys
global_plan_xs = [] 
global_plan_ys = []

# GAN options
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
input_nc = 3
transform_params = get_params(opt, (160, 160)) #input.size)
input_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))
    
# plot options
w = 1.6 
h = 1.6
fig = plt.figure(frameon=False)    
    
# Define a callback for the local plan
def local_plan_callback(msg):
    #print('\nlocal_plan')

    global global_plan_xs, global_plan_ys, localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image, odom_x, odom_y, local_plan_xs, local_plan_ys
    local_plan_xs = [] 
    local_plan_ys = []

    #start = time.time()
    for i in range(0,len(msg.poses)):
        x_temp = round((msg.poses[i].pose.position.x - localCostmapOriginX) / localCostmapResolution) 
        y_temp = round((msg.poses[i].pose.position.y - localCostmapOriginY) / localCostmapResolution)
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            local_plan_xs.append(x_temp)
            local_plan_ys.append(y_temp)
    end = time.time()
    #print()
    #print('LOCAL PLAN RUNTIME = ', end-start)    

# Define a callback for the global plan
def global_plan_callback(msg):
    #print('\nglobal_plan')
    
    global global_plan_xs, global_plan_ys, localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image, odom_x, odom_y, local_plan_xs, local_plan_ys
    global_plan_xs = [] 
    global_plan_ys = []

    for i in range(0,len(msg.poses)):
        global_plan_xs.append(msg.poses[i].pose.position.x) 
        global_plan_ys.append(msg.poses[i].pose.position.y)

    # catch transform from /map to /odom
    transf = tfBuffer.lookup_transform('map', 'odom', rospy.Time())
    t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
    r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
    r_ = np.asarray(r.as_matrix())

    # transform global plan to from /map to /odom
    transformed_plan_xs = []
    transformed_plan_ys = []
    #global_plan_start = time.time()
    for i in range(0, len(global_plan_xs)):
        p = np.array([global_plan_xs[i], global_plan_ys[i], 0.0])
        pnew = p.dot(r_) + t
        x_temp = round((pnew[0] - localCostmapOriginX) / localCostmapResolution)
        y_temp = round((pnew[1] - localCostmapOriginY) / localCostmapResolution)
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            transformed_plan_xs.append(x_temp)
            transformed_plan_ys.append(y_temp)
    #global_plan_end = time.time()
    #print('global_plan_transform_time = ', global_plan_end - global_plan_start)
    
    # save indices of robot's odometry location in local costmap to class variables
    x_odom_index = round((odom_x - localCostmapOriginX) / localCostmapResolution)
    y_odom_index = round((odom_y - localCostmapOriginY) / localCostmapResolution)

    # plot costmap with plans
    #plot_start = time.time()
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax) 
    ax.imshow(image.astype(np.uint8), aspect='auto')
    ax.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='o')
    ax.scatter(local_plan_xs, local_plan_ys, c='yellow', marker='o')
    ax.scatter([x_odom_index], [y_odom_index], c='white', marker='o')
    fig.canvas.draw()
    fig.canvas.tostring_argb()
    #plot_end = time.time()
    #print('plot_time = ', plot_end - plot_start)

    # get GAN output
    input = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    fig.clf()
    input = input_transform(input)
    model.set_input_one(input)  # unpack data from data loader
    forward_start = time.time()
    model.forward()
    forward_end = time.time()
    output = tensor2im(model.fake_B)
    print('gan_output_time = ', forward_end - forward_start)

    # RGB to BGR
    #start_bgr = time.time()
    output = output[:, :, [2, 1, 0]]
    #end_bgr = time.time()
    #print('\nBGR time = ', end_bgr - start_bgr)

    # publish explanation layer
    #points_start = time.time()
    z = 0.0
    a = 255                    
    points = []
    for i in range(0, 160):
        for j in range(0, 160):
            x = localCostmapOriginX + i * localCostmapResolution
            y = localCostmapOriginY + j * localCostmapResolution
            r = output[j, i, 2]
            g = output[j, i, 1]
            b = output[j, i, 0]
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, rgb]
            points.append(pt)
    #points_end = time.time()
    #print('\npoints_time = ', points_end - points_start)

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]

    header = Header()
    header.frame_id = 'odom'
    pc2 = point_cloud2.create_cloud(header, fields, points)
    pc2.header.stamp = rospy.Time.now()
    pub_exp_pointcloud.publish(pc2)
    #rospy.sleep(1.0)

    # publish explanation image
    #flip_start = time.time()
    output[:,:,0] = np.flip(output[:,:,0], axis=1)
    output[:,:,1] = np.flip(output[:,:,1], axis=1)
    output[:,:,2] = np.flip(output[:,:,2], axis=1)
    #flip_end = time.time()
    #print('\nflip_time = ', flip_end - flip_start)
    output_cv = br.cv2_to_imgmsg(output) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3 - it seems so
    pub_exp_image.publish(output_cv)    

        
# Define a callback for the local plan
def odom_callback(msg):
    #print('\nodom')

    global odom_x, odom_y

    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y

# Define a callback for the local plan
def local_costmap_callback(msg):
    #print('\nlocal_costmap')

    global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image

    localCostmapOriginX = msg.info.origin.position.x
    localCostmapOriginY = msg.info.origin.position.y
    localCostmapResolution = msg.info.resolution

    image = np.asarray(msg.data)
    image.resize((msg.info.height,msg.info.width))

    free_space_shade = 180
    obstacle_shade = 255


    # Turn inflated area to free space and 100s to 99s
    image[image >= 99] = obstacle_shade
    image[image <= 98] = free_space_shade
    #image = gray2rgb(image)
    image = np.stack(3 * (image,), axis=-1)


# main part
# Initialize the ROS Node named 'model_with_links_state', allow multiple nodes to be run with this name
rospy.init_node('gan_local_explanation', anonymous=True)

tfBuffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tfBuffer)

pub_exp_image = rospy.Publisher('/local_explanation_image', Image, queue_size=10)
pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)
br = CvBridge()

# Initalize a subscriber to the TEB local plan
sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, local_plan_callback)

# Initalize a subscriber to the TEB global plan
#sub_global_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/global_plan", Path, global_plan_callback)
sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, global_plan_callback)

# Initalize a subscriber to the odometry
sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, odom_callback)

# Initalize a subscriber to the local costmap
sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, local_costmap_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()