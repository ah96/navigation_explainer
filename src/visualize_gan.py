#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path, OccupancyGrid

# Define a callback for the local plan
def local_plan_callback(msg):
    print('\nlocal_plan')
    print(type(msg))
    #print('IN1')
    #print(msg)

# Define a callback for the local plan
def global_plan_callback(msg):
    print('\nglobal_plan')
    print(type(msg))
    #print('IN')
    #print(msg)

# Define a callback for the local plan
def odom_callback(msg):
    print('\nodom')
    print(type(msg))
    #print('IN')
    #print(msg)

# Define a callback for the local plan
def local_costmap_callback(msg):
    print('\nlocal_costmap')
    print(type(msg))
    #print('IN')
    #print(msg)

# Define a callback for the local plan
def map_callback(msg):
    print('\nmap')
    print(type(msg))
    #print('IN')
    #print(msg)


# Initialize the ROS Node named 'model_with_links_state', allow multiple nodes to be run with this name
rospy.init_node('visualize_gan', anonymous=True)

# Initalize a subscriber to the TEB local plan
sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, local_plan_callback)

# Initalize a subscriber to the TEB global plan
sub_global_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/global_plan", Path, global_plan_callback)

# Initalize a subscriber to the odometry
sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, odom_callback)

# Initalize a subscriber to the local costmap
sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, local_costmap_callback)

# Initalize a subscriber to the global map
sub_map = rospy.Subscriber("/map", OccupancyGrid, map_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    print('spinning')
    rospy.spin()