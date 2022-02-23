#!/usr/bin/env python2.7

import rospy
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
#import tf2_ros
import tf2_ros
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

global odom_x, odom_y, local_plan_xs, local_plan_ys, global_plan_xs, global_plan_ys, listener 
global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image 

from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
from data.base_dataset import get_params, get_transform
transform_params = get_params(opt, input.size)
input_nc = 3
input_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))

import PIL.Image
import os

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
        global_plan_ys.append(msg.poses[i].pose.position.x)

    (t, r) = tfBuffer.lookup_transform('/odom', '/map', rospy.Time(0))
    #(t,r) = listener.lookupTransform('/odom', '/map', rospy.Time(0))
    t=np.asarray(t)
   
    r = R.from_quat(r)
    print(r)
    r = np.asarray(r.as_dcm())

    # save indices of robot's odometry location in local costmap to class variables
    localCostmapIndex_x_odom = int((odom_x - localCostmapOriginX) / localCostmapResolution)
    localCostmapIndex_y_odom = int((odom_y - localCostmapOriginY) / localCostmapResolution)

    # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
    x_odom_index = [localCostmapIndex_x_odom]
    y_odom_index = [localCostmapIndex_y_odom]

    local_plan_x_list = []
    local_plan_y_list = []
    for i in range(1, len(local_plan_xs)):
        x_temp = int((local_plan_xs[i] - localCostmapOriginX) / localCostmapResolution)
        y_temp = int((local_plan_ys[i] - localCostmapOriginY) / localCostmapResolution)
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            local_plan_x_list.append(x_temp)
            local_plan_y_list.append(y_temp)

    for i in range(0, len(global_plan_xs)):
        z_dummy = 0
        p = np.array([global_plan_xs[i], global_plan_ys[i], z_dummy])
        pnew = p.dot(r) + t
        global_plan_xs[i] = pnew[0]
        global_plan_ys[i] = pnew[1]

    global_plan_x_list = []
    global_plan_y_list = []
    for i in range(1, len(global_plan_xs)):
        x_temp = int((global_plan_xs[i] - localCostmapOriginX) / localCostmapResolution)
        y_temp = int((global_plan_ys[i] - localCostmapOriginY) / localCostmapResolution)
        if 0 <= x_temp < 160 and 0 <= y_temp < 160:
            global_plan_x_list.append(x_temp)
            global_plan_y_list.append(y_temp)

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
    input = input_transform(input)
    model.set_input_one(input)  # unpack data from data loader
    model.forward()
    output = tensor2im(model.fake_B)
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
        
# Define a callback for the local plan
def odom_callback(msg):
    print('\nodom')

    global odom_x, odom_y

    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y

# Define a callback for the local plan
def local_costmap_callback(msg):
    print('\nlocal_costmap')

    global localCostmapOriginX, localCostmapOriginY, localCostmapResolution, image

    localCostmapOriginX = msg.info.origin.position.x
    localCostmapOriginY = msg.info.origin.position.y
    localCostmapResolution = msg.info.resolution

    image = np.asarray(msg.data)
    image.resize((160,160))

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

# Initalize a subscriber to the TEB local plan
sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, local_plan_callback)

# Initalize a subscriber to the TEB global plan
sub_global_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/global_plan", Path, global_plan_callback)

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