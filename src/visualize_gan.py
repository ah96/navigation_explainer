#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import numpy as np

import tf2_ros
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy

from data.base_dataset import get_params, get_transform
import PIL.Image
import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct


global odom_x, odom_y, local_plan_xs, local_plan_ys, global_plan_xs, global_plan_ys, listener 
global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image 
global pub_exp_image, br, explanation_map, pub_explanation_map, costmap_original, my_awesome_pointcloud

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

# Define a callback for the local plan
def local_plan_callback(msg):
    #print('\nlocal_plan')

    global local_plan_xs, local_plan_ys
    local_plan_xs = [] 
    local_plan_ys = []

    for i in range(0,len(msg.poses)):
        local_plan_xs.append(msg.poses[i].pose.position.x) 
        local_plan_ys.append(msg.poses[i].pose.position.y)

# Define a callback for the local plan
def global_plan_callback(msg):
    #print('\nglobal_plan')

    #global global_plan_xs, global_plan_ys, localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image, odom_x, odom_y, local_plan_xs, local_plan_ys
    global_plan_xs = [] 
    global_plan_ys = []

    for i in range(0,len(msg.poses)):
        global_plan_xs.append(msg.poses[i].pose.position.x) 
        global_plan_ys.append(msg.poses[i].pose.position.y)

    transf_to_map = tfBuffer.lookup_transform('odom', 'map', rospy.Time())
    #print('\ntransf.transform.translation = ', transf.transform.translation)
    #print('\ntransf.transform.rotation = ', transf.transform.rotation)
    #(t,r) = listener.lookupTransform('/odom', '/map', rospy.Time(0))
    t_to_map=np.asarray([transf_to_map.transform.translation.x,transf_to_map.transform.translation.y,transf_to_map.transform.translation.z])
   
    #print(transf.transform.rotation)
    r_to_map = R.from_quat([transf_to_map.transform.rotation.x,transf_to_map.transform.rotation.y,transf_to_map.transform.rotation.z,transf_to_map.transform.rotation.w])
    #print(r)
    r__to_map = np.asarray(r_to_map.as_matrix())    

    transf = tfBuffer.lookup_transform('map', 'odom', rospy.Time())
    #print('\ntransf.transform.translation = ', transf.transform.translation)
    #print('\ntransf.transform.rotation = ', transf.transform.rotation)
    #(t,r) = listener.lookupTransform('/odom', '/map', rospy.Time(0))
    t=np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
   
    #print(transf.transform.rotation)
    r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
    #print(r)
    r_ = np.asarray(r.as_matrix())

    # save indices of robot's odometry location in local costmap to class variables
    localCostmapIndex_x_odom = round((odom_x - localCostmapOriginX) / localCostmapResolution)
    localCostmapIndex_y_odom = round((odom_y - localCostmapOriginY) / localCostmapResolution)

    # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
    x_odom_index = [localCostmapIndex_x_odom]
    y_odom_index = [localCostmapIndex_y_odom]

    local_plan_x_list = []
    local_plan_y_list = []
    for i in range(1, len(local_plan_xs)):
        x_temp = round((local_plan_xs[i] - localCostmapOriginX) / localCostmapResolution)
        y_temp = round((local_plan_ys[i] - localCostmapOriginY) / localCostmapResolution)
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            local_plan_x_list.append(x_temp)
            local_plan_y_list.append(y_temp)

    #print('\nglobal_plan_xs = ', global_plan_xs)

    transformed_plan_xs = []
    transformed_plan_ys = []

    for i in range(0, len(global_plan_xs)):
        z_dummy = 0.0
        p = np.array([global_plan_xs[i], global_plan_ys[i], z_dummy])
        pnew = p.dot(r_) + t
        #global_plan_xs[i] = pnew[0]
        #global_plan_ys[i] = pnew[1]
        transformed_plan_xs.append(pnew[0])
        transformed_plan_ys.append(pnew[1])

    global_plan_x_list = []
    global_plan_y_list = []
    for i in range(1, len(transformed_plan_xs)):
        x_temp = round((transformed_plan_xs[i] - localCostmapOriginX) / localCostmapResolution)
        y_temp = round((transformed_plan_ys[i] - localCostmapOriginY) / localCostmapResolution)
        #print('\n(x_temp, y_temp) = ', (x_temp, y_temp))
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            global_plan_x_list.append(x_temp)
            global_plan_y_list.append(y_temp)

    #print('\nglobal_plan_x_list = ', global_plan_x_list)        

    # plot costmap with plans
    fig = plt.figure(frameon=False)
    w = 1.6
    h = 1.6
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image.astype(np.uint8), aspect='auto')
    ax.scatter(global_plan_x_list, global_plan_y_list, c='blue', marker='o')
    ax.scatter(local_plan_x_list, local_plan_y_list, c='yellow', marker='o')
    ax.scatter(x_odom_index, y_odom_index, c='white', marker='o')
    fig.savefig('input.png', transparent=False)
    fig.clf()

    path = os.getcwd() + '/input.png'
    input = PIL.Image.open(path).convert('RGB')
    #print(type(input))
    #print(input.size)
    transform_params = get_params(opt, input.size)
    input_nc = 3
    input_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))
    input = input_transform(input)
    #print(type(input))
    #print(input.size)
    model.set_input_one(input)  # unpack data from data loader
    model.forward()
    output = tensor2im(model.fake_B)
    #print(type(output))
    #print(output.shape)
   
    '''
    fig = plt.figure(frameon=False)
    w = 1.6 #* 3
    h = 1.6 #* 3
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output, aspect='auto')
    fig.savefig('GAN.png')
    fig.clf()
    '''

    #import pandas as pd
    #pd.DataFrame(output[:,:,0]).to_csv('R.csv')
    #pd.DataFrame(output[:,:,1]).to_csv('G.csv')
    #pd.DataFrame(output[:,:,2]).to_csv('B.csv')

    #output = output.astype(int)

    R_ = copy.deepcopy(output[:,:,0])
    G_ = copy.deepcopy(output[:,:,1])
    B_ = copy.deepcopy(output[:,:,2])

    #import pandas as pd
    #pd.DataFrame(np.flip(output[:,:,0], axis=1)).to_csv('R.csv')
    #temp = output[:,:,0]
    #temp.resize(25600)
    #temp = temp.astype(np.int8).tolist()

    temp = [np.int8(40)]*25600 # gray
    for i in range(0, 160):
        for j in range(0, 160):
            if R_[i, j] > 250 and G_[i, j] > 250:
                temp[i*160+j] = np.int8(254) # yellow
            elif R_[i, j] < 5 and G_[i, j] > 200 and B_[i, j] < 5 and costmap_original[i, j] >= 99:
                temp[i*160+j] = np.int8(110) # green
            elif R_[i, j] > 200 and G_[i, j] < 5 and B_[i, j] < 5:
                temp[i*160+j] = np.int8(128) # red
            elif R_[i, j] > 250 and G_[i, j] > 250 and G_[i, j] > 250:         
                temp[i*160+j] = np.int8(0) # white

    explanation_map.data = tuple(temp)
    #print(type(explanation_map.data))
    #print(type(explanation_map.data[0]))
    #print(type(explanation_map.data[1]))
    #print(type(explanation_map.data[2]))
    #print(type(explanation_map.data[3]))

    pub_explanation_map.publish(explanation_map)

    # RGB to BGR and flip
    output[:,:,0] = B_
    output[:,:,1] = G_
    output[:,:,2] = R_

    #/image_publisher/topic

    #temp = copy.deepcopy(output[:,:,0])
    #output[:,:,0] = output[:,:,2]
    #output[:,:,2] = temp

    #output[:,:,0] = np.flip(output[:,:,0], axis=1)
    #output[:,:,1] = np.flip(output[:,:,1], axis=1)
    #output[:,:,2] = np.flip(output[:,:,2], axis=1)

    points = []
    for i in range(0, 160):
        for j in range(0, 160):
            p = np.array([localCostmapOriginY + j * localCostmapResolution, localCostmapOriginX + i * localCostmapResolution, 0.0])
            pnew = p.dot(r__to_map) + t_to_map
            y = pnew[0]
            x = pnew[1]
            z = 0.0
            r = output[j, i, 2]
            g = output[j, i, 1]
            b = output[j, i, 0]
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            #print hex(rgb)
            pt = [x, y, z, rgb]
            points.append(pt)

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]

    header = Header()
    #header = explanation_map.header
    header.frame_id = 'map'
    pc2 = point_cloud2.create_cloud(header, fields, points)

    pc2.header.stamp = rospy.Time.now()
    pointcloud_publisher.publish(pc2)
    #rospy.sleep(1.0)

    #if output is not None:
    output_cv = br.cv2_to_imgmsg(output)#,encoding="rgb8: CV_8UC3") - encoding not supported in Python3 - it seems so
    pub_exp_image.publish(output_cv)

    pub_exp_image_new.publish(output_cv)

        
# Define a callback for the local plan
def odom_callback(msg):
    #print('\nodom')

    global odom_x, odom_y

    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y

# Define a callback for the local plan
def local_costmap_callback(msg):
    print('\nlocal_costmap')

    global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image, explanation_map, costmap_original

    explanation_map = OccupancyGrid()
    explanation_map.header = msg.header
    explanation_map.data = msg.data
    explanation_map.info = msg.info
    print(msg.info)
    print(msg.header)

    localCostmapOriginX = msg.info.origin.position.x
    localCostmapOriginY = msg.info.origin.position.y
    localCostmapResolution = msg.info.resolution

    image = np.asarray(msg.data)
    image.resize((160,160))

    costmap_original = copy.deepcopy(image)

    #import pandas as pd
    #pd.DataFrame(image).to_csv('costmap.csv')

    # Turn inflated area to free space and 100s to 99s
    image[image == 100] = 99
    image[image != 99] = 0

    gray_shade = 180
    white_shade = 255
    image = gray2rgb(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = gray_shade
            elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = white_shade

# Define a callback for the local plan
#def map_callback(msg):
#    print('\nmap')


# Initialize the ROS Node named 'model_with_links_state', allow multiple nodes to be run with this name
rospy.init_node('visualize_gan', anonymous=True)

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

pub_exp_image = rospy.Publisher('explanation_image', Image, queue_size=10)
pub_exp_image_new = rospy.Publisher('image_publisher/image', Image, queue_size=10)
pointcloud_publisher = rospy.Publisher("/my_pointcloud_topic", PointCloud2)
pub_explanation_map = rospy.Publisher('explanation_map', OccupancyGrid, queue_size=10)
explanation_map = OccupancyGrid()
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

# Initalize a subscriber to the global map
#sub_map = rospy.Subscriber("/map", OccupancyGrid, map_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    print('spinning')
    rospy.spin()