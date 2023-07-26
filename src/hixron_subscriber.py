#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
import copy
import tf2_ros
import os
from gazebo_msgs.msg import ModelStates
import math
from skimage.measure import regionprops
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
import message_filters
import torch
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import cv2
from sensor_msgs import point_cloud2
import struct

# hixron_subscriber class
class hixron_subscriber(object):
    # constructor
    def __init__(self):
        # simulation or real-world experiment
        self.simulation = True

        # whether to plot
        self.plot_local_costmap_bool = False
        self.plot_global_costmap_bool = False
        self.plot_local_semantic_map_bool = False
        self.plot_global_semantic_map_bool = False

        # global counter for plotting
        self.counter_global = 0
        self.local_plan_counter = 0
        self.global_costmap_counter = 0

        # use local and/or global costmap
        self.use_local_costmap = False
        self.use_global_costmap = False

        # use local and/or global (semantic) map
        self.use_local_semantic_map = False
        self.use_global_semantic_map = True
        
        # use camera
        self.use_camera = False

        # inflation
        self.inflation_radius = 0.275

        # explanation layer
        self.semantic_layer_bool = True
        
        # data directories
        self.dirCurr = os.getcwd()
        self.yoloDir = self.dirCurr + '/yolo_data/'

        self.dirMain = 'hixron_data'
        try:
            os.mkdir(self.dirMain)
        except FileExistsError:
            pass

        self.dirData = self.dirMain + '/data'
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_local_semantic_map_bool == True:
            self.local_semantic_map_dir = self.dirMain + '/local_semantic_map_images'
            try:
                os.mkdir(self.local_semantic_map_dir)
            except FileExistsError:
                pass

        if self.plot_global_semantic_map_bool == True:
            self.global_semantic_map_dir = self.dirMain + '/global_semantic_map_images'
            try:
                os.mkdir(self.global_semantic_map_dir)
            except FileExistsError:
                pass

        if self.plot_local_costmap_bool == True and self.use_local_costmap == True:
            self.local_costmap_dir = self.dirMain + '/local_costmap_images'
            try:
                os.mkdir(self.local_costmap_dir)
            except FileExistsError:
                pass

        if self.plot_global_costmap_bool == True and self.use_global_costmap == True:
            self.global_costmap_dir = self.dirMain + '/global_costmap_images'
            try:
                os.mkdir(self.global_costmap_dir)
            except FileExistsError:
                pass

        # simulation variables
        if self.simulation:
            # gazebo vars
            self.gazebo_names = []
            self.gazebo_poses = []
            self.gazebo_labels = []

        # tf vars
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_odom_map = [] 
        self.tf_map_odom = []

        # robot variables
        self.robot_position_map = Point(0.0,0.0,0.0)        
        self.robot_orientation_map = Quaternion(0.0,0.0,0.0,1.0)
        self.robot_position_odom = Point(0.0,0.0,0.0)        
        self.robot_orientation_odom = Quaternion(0.0,0.0,0.0,1.0)
        self.footprint = []  

        # plans' variables
        self.local_plan = []
        self.global_plan = [] 
                
        # local semantic map vars
        self.local_semantic_map_origin_x = 0 
        self.local_semantic_map_origin_y = 0 
        self.local_semantic_map_resolution = 0.025
        self.local_semantic_map_size = 160
        self.local_semantic_map_info = []
        self.local_semantic_map = np.zeros((self.local_semantic_map_size, self.local_semantic_map_size))

        # ontology part
        self.scenario_name = 'library' #'scenario1', 'library'
        # load ontology
        self.ontology = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'ontology.csv')
        #cols = ['c_map_x', 'c_map_y', 'd_map_x', 'd_map_y']
        #self.ontology[cols] = self.ontology[cols].astype(float)
        self.ontology = np.array(self.ontology)
        #print(self.ontology)

        # load global semantic map info
        self.global_semantic_map_info = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'map_info.csv')) 
        # global semantic map vars
        self.global_semantic_map_origin_x = float(self.global_semantic_map_info[0,4]) 
        self.global_semantic_map_origin_y = float(self.global_semantic_map_info[0,5]) 
        self.global_map_resolution = float(self.global_semantic_map_info[0,1])
        self.global_semantic_map_size = [int(self.global_semantic_map_info[0,3]), int(self.global_semantic_map_info[0,2])]
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]), dtype=float)
        #print(self.global_semantic_map_origin_x, self.global_semantic_map_origin_y, self.global_map_resolution, self.global_semantic_map_size)

        # camera variables
        self.camera_image = np.array([])
        self.depth_image = np.array([])
        # camera projection matrix 
        self.P = np.array([])

    # declare subscribers
    def main_(self):
        # create the base plot structure
        if self.plot_local_costmap_bool == True or self.plot_global_costmap_bool == True or self.plot_local_semantic_map_bool == True or self.plot_global_semantic_map_bool == True:
            self.fig = plt.figure(frameon=False)
            #self.w = 1.6 * 3
            #self.h = 1.6 * 3
            #self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

        # subscribers
        # local plan subscriber
        #self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        # global plan subscriber 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        # robot footprint subscriber
        #self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        # odometry subscriber
        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        # global-amcl pose subscriber
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        # local costmap subscriber
        if self.use_local_costmap == True:
            self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        # global costmap subscriber
        if self.use_global_costmap == True:
            self.sub_global_costmap = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_callback)

        # explanation layer
        if self.semantic_layer_bool:
            self.pub_semantic_layer = rospy.Publisher("/semantic_layer", PointCloud2)
        
            # point_cloud variables
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

            # header
            self.header = Header()

        # CV part
        # robot camera subscribers
        if self.use_camera == True:
            self.camera_image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
            self.depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image) #"32FC1"
            self.camera_info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.camera_image_sub, self.depth_sub, self.camera_info_sub], 10, 1.0)
            self.ts.registerCallback(self.camera_feed_callback)
            # Load YOLO model
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
            self.model = torch.hub.load(self.path_prefix + '/yolov5_master/', 'custom', self.path_prefix + '/models/yolov5s.pt', source='local')  # custom trained model

        # gazebo vars
        if self.simulation:
            # gazebo model states subscriber
            self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

            # load gazebo tags
            self.gazebo_labels = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'gazebo_tags.csv')) 
            
    # camera feed callback
    def camera_feed_callback(self, img, depth_img, info):
        #print('\ncamera_feed_callback')

        # RGB IMAGE
        # convert rgb image from robot's camera to np.array
        self.camera_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # Get image dimensions
        #(height, width) = image.shape[:2]

        # DEPTH IMAGE
        # convert depth image to np array
        self.depth_image = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_img.height, depth_img.width, -1)    
        # fill missing values with negative distance (-1.0)
        self.depth_image = np.nan_to_num(self.depth_image, nan=-1.0)
        
        # get projection matrix
        self.P = info.P

        # potential place to make semantic map, if local costmap is not used
        if self.use_local_costmap == False:
            # create semantic data
            self.create_semantic_data()

            # increase the global counter
            self.counter_global += 1

    # odometry callback
    def odom_callback(self, msg):
        #print('odom_callback')
        
        self.robot_position_odom = msg.pose.pose.position
        self.robot_orientation_odom = msg.pose.pose.orientation
        self.robot_twist_linear = msg.twist.twist.linear
        self.robot_twist_angular = msg.twist.twist.angular

    # amcl (global) pose callback
    def amcl_callback(self, msg):
        #print('amcl_callback')

        self.robot_position_map = msg.pose.pose.position
        self.robot_orientation_map = msg.pose.pose.orientation

    # robot footprint callback
    def footprint_callback(self, msg):
        #print('local_plan_callback')  
        self.footprint = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        
    # Gazebo callback
    def model_state_callback(self, states_msg):
        #print('model_state_callback')  
        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

    # global plan callback
    def global_plan_callback(self, msg):
        print('\nglobal_plan_callback!')
        
        self.global_plan = []
        for i in range(0,len(msg.poses)):
            self.global_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
        #pd.DataFrame(self.global_plan).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan.csv', index=False)#, header=False)

        # potential place to make a local semantic map, if local costmap is not used
        if self.use_local_costmap == False and self.use_local_semantic_map:

            # update local_map params (origin cordinates)
            self.local_semantic_map_origin_x = self.robot_position_map.x - self.local_semantic_map_resolution * self.local_semantic_map_size * 0.5 
            self.local_semantic_map_origin_y = self.robot_position_map.y - self.local_semantic_map_resolution * self.local_semantic_map_size * 0.5
            self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y]
            
            # create semantic data
            self.create_semantic_data()

            # increase the global counter (needed for plotting numeration)
            self.counter_global += 1

        elif self.use_global_semantic_map:
            
            # create semantic data
            self.create_semantic_data()

            # increase the global counter (needed for plotting numeration)
            self.counter_global += 1        
        
    # local plan callback
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        
        try:
            self.odom_copy = copy.deepcopy(self.odom)
            self.amcl_pose_copy = copy.deepcopy(self.amcl_pose)
            self.footprint_copy = copy.deepcopy(self.footprint)

            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
            self.tf_map_odom = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]

            # get the local plan
            self.local_plan = []
            for i in range(0,len(msg.poses)):
                self.local_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

            # save the vars for the publisher/explainer
            pd.DataFrame(self.odom_copy).to_csv(self.dirCurr + '/' + self.dirData + '/odom.csv', index=False)#, header=False)
            pd.DataFrame(self.amcl_pose_copy).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose.csv', index=False)#, header=False)
            pd.DataFrame(self.footprint_copy).to_csv(self.dirCurr + '/' + self.dirData + '/footprint.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_odom_map).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom.csv', index=False)#, header=False)            
            pd.DataFrame(self.local_plan).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan.csv', index=False)#, header=False)

            # potential place to make semantic map, if local costmap is not used
            if self.use_local_costmap == False and self.simulation == True:
                # increase the local planner counter
                self.local_plan_counter += 1

                if self.local_plan_counter == 20:
                    # update local_map (costmap) data
                    self.local_semantic_map_origin_x = self.robot_position_map.x - 0.5 * self.local_semantic_map_size * self.local_semantic_map_resolution
                    self.local_semantic_map_origin_y = self.robot_position_map.y - 0.5 * self.local_semantic_map_size * self.local_semantic_map_resolution
                    #self.local_semantic_map_resolution = msg.info.resolution
                    #self.local_semantic_map_size = self.local_semantic_map_size
                    self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y]#, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

                    # create np.array local_map object
                    #self.local_semantic_map = np.zeros((self.local_semantic_map_size,self.local_semantic_map_size))

                    # create semantic data
                    self.create_semantic_data()

                    # increase the global counter
                    self.counter_global += 1
                    # reset the local planner counter
                    self.local_plan_counter = 0

        except:
            #print('exception = ', e) # local plan frame rate is too high to print possible exceptions
            return

    # local costmap callback
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')
        
        try:          
            # update local_map data
            self.local_semantic_map_origin_x = msg.info.origin.position.x
            self.local_semantic_map_origin_y = msg.info.origin.position.y
            #self.local_semantic_map_resolution = msg.info.resolution
            #self.local_semantic_map_size = self.local_semantic_map_size
            self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create np.array local_map object
            self.local_semantic_map = np.asarray(msg.data)
            self.local_semantic_map.resize((self.local_semantic_map_size,self.local_semantic_map_size))

            if self.plot_costmaps_bool == True:
                self.plot_costmaps()
                
            # Turn inflated area to free space and 100s to 99s
            self.local_semantic_map[self.local_semantic_map == 100] = 99
            self.local_semantic_map[self.local_semantic_map <= 98] = 0

            # create semantic map
            self.create_semantic_data()

            # increase the global counter
            self.counter_global += 1

        except Exception as e:
            print('exception = ', e)
            return

    # global costmap callback
    def global_costmap_callback(self, msg):
        print('\nglobal_costmap_callback')
        
        try:
            # create np.array global_map object
            self.global_costmap = np.asarray(msg.data)
            self.global_costmap.resize((msg.info.width,msg.info.height))

            if self.plot_global_costmap_bool == True:
                self.plot_global_costmap()

            # increase the global costmap counter
            self.counter_global_costmap += 1

        except Exception as e:
            print('exception = ', e)
            return

    # plot local costmap
    def plot_local_costmap(self):
        start = time.time()

        dirCurr = self.local_costmap_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass
        
        local_map_99s_100s = copy.deepcopy(self.local_semantic_map)
        local_map_99s_100s[local_map_99s_100s < 99] = 0        
        #self.fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        local_map_99s_100s = np.flip(local_map_99s_100s, 0)
        self.ax.imshow(local_map_99s_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s_100s.png', transparent=False)
        #self.fig.clf()
        
        local_map_original = copy.deepcopy(self.local_semantic_map)
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_map_original = np.flip(local_map_original, 0)
        self.ax.imshow(local_map_original.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_original.png', transparent=False)
        #self.fig.clf()
        
        local_map_100s = copy.deepcopy(self.local_semantic_map)
        local_map_100s[local_map_100s != 100] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_map_100s = np.flip(local_map_100s, 0)
        self.ax.imshow(local_map_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_100s.png', transparent=False)
        #self.fig.clf()
        
        local_map_99s = copy.deepcopy(self.local_semantic_map)
        local_map_99s[local_map_99s != 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_map_99s = np.flip(local_map_99s, 0)
        self.ax.imshow(local_map_99s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s.png', transparent=False)
        #self.fig.clf()
        
        local_map_less_than_99 = copy.deepcopy(self.local_semantic_map)
        local_map_less_than_99[local_map_less_than_99 >= 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_map_less_than_99 = np.flip(local_map_less_than_99, 0)
        self.ax.imshow(local_map_less_than_99.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_less_than_99.png', transparent=False)
        self.fig.clf()
        
        end = time.time()
        print('costmaps plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # plot global costmap
    def plot_global_costmap(self):
        start = time.time()

        dirCurr = self.global_costmap_dir + '/' + str(self.counter_global_costmap)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        self.ax.imshow(self.global_costmap, aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'global_costmap_original.png', transparent=False)
        self.fig.clf()
        
        end = time.time()
        print('global costmap plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # create semantic data
    def create_semantic_data(self):
        # update ontology
        self.update_ontology()

        # create semantic map
        if self.use_local_semantic_map == True:
            self.create_local_semantic_map()
        if self.use_global_semantic_map ==  True:
            self.create_global_semantic_map()

        # plot semantic_map
        if self.plot_local_semantic_map_bool == True:
            self.plot_local_semantic_map()
        if self.plot_global_semantic_map_bool == True:
            self.plot_global_semantic_map()

        # create interpretable features
        self.create_interpretable_features()

        if self.semantic_layer_bool:
            self.publish_semantic_layer()
        
    # update ontology
    def update_ontology(self):
        # check if any object changed its position from simulation or from object detection (and tracking)

        # simulation relying on Gazebo
        if self.simulation:
            respect_mass_centre = True

            for i in range(0, self.ontology.shape[0]):
                ## if the object has some affordance (etc. movability, openability), then it may have changed its position 
                #if self.ontology[i][7] == 1 or self.ontology[i][8] == 1:
                # get the object's new position from Gazebo
                obj_gazebo_name = self.ontology[i][1]
                obj_gazebo_name_idx = self.gazebo_names.index(obj_gazebo_name)
                obj_x_new = self.gazebo_poses[obj_gazebo_name_idx].position.x
                obj_y_new = self.gazebo_poses[obj_gazebo_name_idx].position.y

                obj_x_size = copy.deepcopy(self.ontology[i][5])
                obj_y_size = copy.deepcopy(self.ontology[i][6])

                obj_x_current = self.ontology[i][3]
                obj_y_current = self.ontology[i][4]

                if respect_mass_centre == False:
                    # check whether the (centroid) coordinates of the object are changed (enough)
                    diff_x = abs(obj_x_new - obj_x_current)
                    diff_y = abs(obj_y_new - obj_y_current)
                    if diff_x > obj_x_size or diff_y > obj_y_size:
                        #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                        self.ontology[i][3] = obj_x_new
                        self.ontology[i][4] = obj_y_new
                
                else:
                    # update ontology
                    # almost every object type in Gazebo has a different center of mass
                    if 'chair' in obj_gazebo_name:
                        # top-right is the mass centre
                        obj_x_new -= 0.5*obj_x_size
                        obj_y_new -= 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > 0.5*obj_x_size or diff_y > 0.5*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new

                    elif 'bookshelf' in obj_gazebo_name:
                        # top-right is the mass centre
                        obj_x_new += 0.5*obj_x_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > obj_x_size or diff_y > obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new

                    else:
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > 0.5*obj_x_size or diff_y > 0.5*obj_y_size:
                            #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new 

        # real world or simulation relying on object detection
        elif self.simulation == False:
            # in object detection the center of mass is always the object's centroid
            start = time.time()
    
            # Inference
            results = self.model(self.camera_image)
            end = time.time()
            print('yolov5 inference runtime: ' + str(round(1000*(end-start), 2)) + ' ms')
            
            # Results
            res = np.array(results.pandas().xyxy[0])
            
            # labels, confidences and bounding boxes
            labels = list(res[:,-1])
            #print('original_labels: ', labels)
            confidences = list((res[:,-3]))
            #print('confidences: ', confidences)
            # boundix boxes coordinates
            x_y_min_max = np.array(res[:,0:4])
            
            confidence_threshold = 0.0
            if confidence_threshold > 0.0:
                labels_ = []
                for i in range(len(labels)):
                    if confidences[i] < confidence_threshold:
                        np.delete(x_y_min_max, i, 0)
                    else:
                        labels_.append(labels[i])
                labels = labels_
                #print('filtered_labels: ', labels)

            # get the 3D coordinates of the detected objects in the /map dataframe 
            objects_coordinates_3d = []

            # camera intrinsic parameters
            fx = self.P[0]
            cx = self.P[2]
            #fy = self.P[5]
            #cy = self.P[6]
            for i in range(0, len(labels)):
                # get the depth of the detected object's centroid
                u = int((x_y_min_max[i,0]+x_y_min_max[i,2])/2)
                v = int((x_y_min_max[i,1]+x_y_min_max[i,3])/2)
                depth = self.depth_image[v, u][0]
                # get the 3D positions relative to the robot
                x = depth
                y = (u - cx) * z / fx
                z = 0 #(v - cy) * z / fy
                t_ro_R = [x,y,z]
                
                t_wr_W = [self.robot_position_map.x, self.robot_position_map.y, self.robot_position_map.z]
                r_RW = R.from_quat([self.robot_orientation_map.x, self.robot_orientation_map.y, self.robot_orientation_map.z, self.robot_orientation_map.w]).inv

                t_ro_W = r_RW * t_ro_R
                t_wo_W = t_wr_W + t_ro_W
                
                x_o_new = t_wo_W[0]
                y_o_new = t_wo_W[1]

                for j in range(0, self.ontology.shape[0]):
                    if labels[i] == self.ontology[j][1]:

                        x_o_curr = self.ontology[i][2]
                        y_o_curr = self.ontology[i][3]

                        if abs(x_o_new - x_o_curr) > 0.1 and abs(y_o_new - y_o_curr) > 0.1:
                            self.ontology[i][2] = x_o_new
                            self.ontology[i][3] = y_o_new
                            print('\nThe object ' + labels[i] + ' has changed its position!')
                            print('\nOld position: x = ' + str(x_o_curr) + ', y = ' + str(y_o_curr))
                            print('\nNew position: x = ' + str(x_o_new) + ', y = ' + str(y_o_new))                                

        # save the updated ontology for the publisher
        pd.DataFrame(self.ontology).to_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv', index=False)#, header=False)
    
    # create local semantic map
    def create_local_semantic_map(self):
        # GET transformations between coordinate frames
        # tf from map to odom
        t_mo = np.asarray([self.tf_map_odom[0],self.tf_map_odom[1],self.tf_map_odom[2]])
        r_mo = R.from_quat([self.tf_map_odom[3],self.tf_map_odom[4],self.tf_map_odom[5],self.tf_map_odom[6]])
        r_mo = np.asarray(r_mo.as_matrix())
        #print('r_mo = ', r_mo)
        #print('t_mo = ', t_mo)

        # tf from odom to map
        t_om = np.asarray([self.tf_odom_map[0],self.tf_odom_map[1],self.tf_odom_map[2]])
        r_om = R.from_quat([self.tf_odom_map[3],self.tf_odom_map[4],self.tf_odom_map[5],self.tf_odom_map[6]])
        r_om = np.asarray(r_om.as_matrix())
        #print('r_om = ', r_om)
        #print('t_om = ', t_om)

        # convert LC points from /odom to /map
        # LC origin is a bottom-left point
        lc_bl_odom_x = self.local_semantic_map_origin_x
        lc_bl_odom_y = self.local_semantic_map_origin_y
        lc_p_odom = np.array([lc_bl_odom_x, lc_bl_odom_y, 0.0])#
        lc_p_map = lc_p_odom.dot(r_om) + t_om#
        lc_map_bl_x = lc_p_map[0]
        lc_map_bl_y = lc_p_map[1]
        
        # LC's top-right point
        lc_tr_odom_x = self.local_semantic_map_origin_x + self.local_semantic_map_size * self.local_semantic_map_resolution
        lc_tr_odom_y = self.local_semantic_map_origin_y + self.local_semantic_map_size * self.local_semantic_map_resolution
        lc_p_odom = np.array([lc_tr_odom_x, lc_tr_odom_y, 0.0])#
        lc_p_map = lc_p_odom.dot(r_om) + t_om#
        lc_map_tr_x = lc_p_map[0]
        lc_map_tr_y = lc_p_map[1]
        
        # LC sides coordinates in the /map frame
        lc_map_left = lc_map_bl_x
        lc_map_right = lc_map_tr_x
        lc_map_bottom = lc_map_bl_y
        lc_map_top = lc_map_tr_y
        #print('(lc_map_left, lc_map_right, lc_map_bottom, lc_map_top) = ', (lc_map_left, lc_map_right, lc_map_bottom, lc_map_top))

        start = time.time()
        self.semantic_map = np.zeros(self.local_semantic_map.shape)
        self.semantic_map_inflated = np.zeros(self.local_semantic_map.shape)
        inflation_factor = 0
        for i in range(0, self.ontology.shape[0]):
            # object's vertices from /map to /odom and /lc
            # top left vertex
            x_size = self.ontology[i][4]
            y_size = self.ontology[i][5]
            c_map_x = self.ontology[i][2]
            c_map_y = self.ontology[i][3]

            # top left vertex
            tl_map_x = c_map_x - 0.5*x_size
            tl_map_y = c_map_y + 0.5*y_size

            # bottom right vertex
            br_map_x = c_map_x + 0.5*x_size
            br_map_y = c_map_y - 0.5*y_size
            
            # top right vertex
            tr_map_x = c_map_x + 0.5*x_size
            tr_map_y = c_map_y + 0.5*y_size
            p_map = np.array([tr_map_x, tr_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo#
            tr_odom_x = p_odom[0]#
            tr_odom_y = p_odom[1]#
            tr_pixel_x = int((tr_odom_x - self.local_semantic_map_origin_x) / self.local_semantic_map_resolution)
            tr_pixel_y = int((tr_odom_y - self.local_semantic_map_origin_y) / self.local_semantic_map_resolution)

            # bottom left vertex
            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y - 0.5*y_size
            p_map = np.array([bl_map_x, bl_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo#
            bl_odom_x = p_odom[0]#
            bl_odom_y = p_odom[1]#
            bl_pixel_x = int((bl_odom_x - self.local_semantic_map_origin_x) / self.local_semantic_map_resolution)
            bl_pixel_y = int((bl_odom_y - self.local_semantic_map_origin_y) / self.local_semantic_map_resolution)

            # object's sides coordinates
            object_left = bl_pixel_x
            object_top = tr_pixel_y
            object_right = tr_pixel_x
            object_bottom = bl_pixel_y

            obstacle_in_neighborhood = False 
            x_1 = 0
            x_2 = 0
            y_1 = 0
            y_2 = 0

            # centroid in LC
            if lc_map_left < c_map_x < lc_map_right and lc_map_bottom < c_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            # top-left(tl) in LC
            elif lc_map_left < tl_map_x < lc_map_right and lc_map_bottom < tl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            # bottom-left(bl) in LC
            elif lc_map_left < bl_map_x < lc_map_right and lc_map_bottom < bl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = object_bottom
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True
                
            # bottom-right(br) in LC
            elif lc_map_left < br_map_x < lc_map_right and lc_map_bottom < br_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = object_bottom
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True
                
            # top-right(tr) in LC
            elif lc_map_left < tr_map_x < lc_map_right and lc_map_bottom < tr_map_y < lc_map_top:
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            if obstacle_in_neighborhood == True:
                # semantic map
                self.semantic_map[max(0, y_1-inflation_factor):min(self.local_semantic_map_size-1, y_2+inflation_factor), max(0,x_1-inflation_factor):min(self.local_semantic_map_size-1, x_2+inflation_factor)] = i+1
                
                # inflate semantic map using heuristics
                inflation_x = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                inflation_y = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2)                 
                self.semantic_map_inflated[max(0, y_1-inflation_y):min(self.local_semantic_map_size-1, y_2+inflation_y), max(0,x_1-inflation_x):min(self.local_semantic_map_size-1, x_2+inflation_x)] = i+1
       
        end = time.time()
        print('semantic map creation runtime = ' + str(round(end-start,3)) + ' seconds!')


        # find centroids of the objects in the semantic map
        lc_regions = regionprops(self.semantic_map.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        self.centroids_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            self.centroids_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        # inflate using the remaining obstacle points of the local costmap            
        if self.use_local_costmap == True:
            for i in range(self.semantic_map.shape[0]):
                for j in range(0, self.semantic_map.shape[1]):
                    if self.local_semantic_map[i, j] > 98 and self.semantic_map_inflated[i, j] == 0:
                        distances_to_centroids = []
                        distances_indices = []
                        for k in range(0, len(self.centroids_semantic_map)):
                            dx = abs(j - self.centroids_semantic_map[k][1])
                            dy = abs(i - self.centroids_semantic_map[k][2])
                            distances_to_centroids.append(dx + dy) # L1
                            #distances_to_centroids.append(math.sqrt(dx**2 + dy**2)) # L2
                            distances_indices.append(k)
                        idx = distances_to_centroids.index(min(distances_to_centroids))
                        self.semantic_map_inflated[i, j] = self.centroids_semantic_map[idx][0]

            # turn pixels in the inflated semantic_map, which are zero in the local costmap, to zero
            self.semantic_map_inflated[self.local_semantic_map == 0] = 0

        # save local and semantic maps data
        pd.DataFrame(self.local_semantic_map_info).to_csv(self.dirCurr + '/' + self.dirData + '/local_map_info.csv', index=False)#, header=False)
        pd.DataFrame(self.local_semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/local_map.csv', index=False) #, header=False)
        pd.DataFrame(self.semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map.csv', index=False)#, header=False)
        pd.DataFrame(self.semantic_map_inflated).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map_inflated.csv', index=False)#, header=False)

    # create global semantic map
    def create_global_semantic_map(self):
        start = time.time()
        
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        self.global_semantic_map_inflated = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #print('(self.global_semantic_map_size[0],self.global_semantic_map_size[1]) = ', (self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        
        for i in range(0, self.ontology.shape[0]):
            #if self.ontology[i][2] != 'chair':
            #    continue
            # IMPORTANT OBJECT'S POINTS
            # centroid and size
            c_map_x = float(self.ontology[i][3])
            c_map_y = float(self.ontology[i][4])
            #print('(i, name) = ', (i, self.ontology[i][2]))
            x_size = float(self.ontology[i][5])
            y_size = float(self.ontology[i][6])
            
            # top left vertex
            #tl_map_x = c_map_x - 0.5*x_size
            #tl_map_y = c_map_y - 0.5*y_size
            #tl_pixel_x = int((tl_map_x - self.global_semantic_map_origin_x) / self.global_map_resolution)
            #tl_pixel_y = int((tl_map_y - self.global_semantic_map_origin_y) / self.global_map_resolution)

            # bottom right vertex
            #br_map_x = c_map_x + 0.5*x_size
            #br_map_y = c_map_y + 0.5*y_size
            #br_pixel_x = int((br_map_x - self.global_semantic_map_origin_x) / self.global_map_resolution)
            #br_pixel_y = int((br_map_y - self.global_semantic_map_origin_y) / self.global_map_resolution)
            
            # top right vertex
            tr_map_x = c_map_x + 0.5*x_size
            tr_map_y = c_map_y - 0.5*y_size
            tr_pixel_x = int((tr_map_x - self.global_semantic_map_origin_x) / self.global_map_resolution)
            tr_pixel_y = int((tr_map_y - self.global_semantic_map_origin_y) / self.global_map_resolution)

            # bottom left vertex
            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y + 0.5*y_size
            bl_pixel_x = int((bl_map_x - self.global_semantic_map_origin_x) / self.global_map_resolution)
            bl_pixel_y = int((bl_map_y - self.global_semantic_map_origin_y) / self.global_map_resolution)

            # object's sides coordinates
            object_left = bl_pixel_x
            object_top = tr_pixel_y
            object_right = tr_pixel_x
            object_bottom = bl_pixel_y
            #if self.ontology[i][1] == 'kitchen_chair_clone_4_clone_5':
            #print('\n(i, name) = ', (i, self.ontology[i][1]))
            #print('(object_left,object_right,object_top,object_bottom) = ', (object_left,object_right,object_top,object_bottom))
            #print('(c_map_x,c_map_y,x_size,y_size) = ', (c_map_x,c_map_y,x_size,y_size))

            # global semantic map
            self.global_semantic_map[max(0, object_top):min(self.global_semantic_map_size[0], object_bottom), max(0, object_left):min(self.global_semantic_map_size[1], object_right)] = i+1

            # inflate global semantic map
            inflation_x = int(self.inflation_radius / self.global_map_resolution) 
            inflation_y = int(self.inflation_radius / self.global_map_resolution)
            self.global_semantic_map_inflated[max(0, object_top-inflation_y):min(self.global_semantic_map_size[0], object_bottom+inflation_y), max(0, object_left-inflation_x):min(self.global_semantic_map_size[1], object_right+inflation_x)] = i+1

        end = time.time()
        print('global semantic map creation runtime = ' + str(round(end-start,3)) + ' seconds!')

        #self.global_semantic_map[232:270,63:71] = 40

        # find centroids of the objects in the semantic map
        lc_regions = regionprops(self.global_semantic_map.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        self.centroids_global_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            self.centroids_global_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        # save local and semantic maps data
        pd.DataFrame(self.global_semantic_map_info).to_csv(self.dirCurr + '/' + self.dirData + '/global_semantic_map_info.csv', index=False)#, header=False)
        pd.DataFrame(self.global_semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/global_semantic_map.csv', index=False) #, header=False)
        pd.DataFrame(self.global_semantic_map_inflated).to_csv(self.dirCurr + '/' + self.dirData + '/global_semantic_map_inflated.csv', index=False)#, header=False)

    # plot local semantic_map
    def plot_local_semantic_map(self):
        start = time.time()

        dirCurr = self.local_semantic_map_dir + '/' + str(self.counter_global)            
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/semantic_map.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_inflated_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/semantic_map_inflated.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_semantic_map.append([v,cx,cy,self.ontology[v-1][2]])

        for i in range(0, len(centroids_semantic_map)):
            self.ax.scatter(centroids_semantic_map[i][1], centroids_semantic_map[i][2], c='white', marker='o')   
            self.ax.text(centroids_semantic_map[i][1], centroids_semantic_map[i][2], centroids_semantic_map[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_semantic_map).to_csv(dirCurr + '/centroids_semantic_map.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_semantic_map_inflated = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_semantic_map_inflated.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_semantic_map_inflated)):
            self.ax.scatter(centroids_semantic_map_inflated[i][1], centroids_semantic_map_inflated[i][2], c='white', marker='o')   
            self.ax.text(centroids_semantic_map_inflated[i][1], centroids_semantic_map_inflated[i][2], centroids_semantic_map_inflated[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_inflated_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_semantic_map_inflated).to_csv(dirCurr + '/centroids_semantic_map_inflated.csv', index=False)#, header=False)

        end = time.time()
        print('semantic map plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # plot global semantic_map
    def plot_global_semantic_map(self):
        start = time.time()

        dirCurr = self.global_semantic_map_dir + '/' + str(self.counter_global)            
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/global_semantic_map.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_inflated_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/global_semantic_map_inflated.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_global_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_global_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_global_semantic_map)):
            self.ax.scatter(centroids_global_semantic_map[i][1], centroids_global_semantic_map[i][2], c='white', marker='o')   
            self.ax.text(centroids_global_semantic_map[i][1], centroids_global_semantic_map[i][2], centroids_global_semantic_map[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_global_semantic_map).to_csv(dirCurr + '/global_centroids_semantic_map.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_global_semantic_map_inflated = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_global_semantic_map_inflated.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_global_semantic_map_inflated)):
            self.ax.scatter(centroids_global_semantic_map_inflated[i][1], centroids_global_semantic_map_inflated[i][2], c='white', marker='o')   
            self.ax.text(centroids_global_semantic_map_inflated[i][1], centroids_global_semantic_map_inflated[i][2], centroids_global_semantic_map_inflated[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_inflated_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_global_semantic_map_inflated).to_csv(dirCurr + '/global_centroids_semantic_map_inflated.csv', index=False)#, header=False)

        end = time.time()
        print('semantic map plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # create interpretable features
    def create_interpretable_features(self):
        if self.use_global_semantic_map:
            # list of labels of objects in global semantic map
            labels = np.unique(self.global_semantic_map)
            object_affordance_pairs_global = [] # [label, object, affordance]
            # get object-affordance pairs in the current global semantic map
            for i in range(0, self.ontology.shape[0]):
                if self.ontology[i][0] in labels:
                    if self.ontology[i][7] == 1:
                        object_affordance_pairs_global.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                    if self.ontology[i][8] == 1:
                        object_affordance_pairs_global.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

            # save object-affordance pairs for publisher
            pd.DataFrame(object_affordance_pairs_global).to_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs_global.csv', index=False)#, header=False)

        if self.use_local_semantic_map:
            # list of labels of objects in local semantic map
            labels = np.unique(self.local_semantic_map)
            object_affordance_pairs_local = [] # [label, object, affordance]
            # get object-affordance pairs in the current local semantic map
            for i in range(0, self.ontology.shape[0]):
                if self.ontology[i][0] in labels:
                    if self.ontology[i][6] == 1:
                        object_affordance_pairs_local.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                    if self.ontology[i][7] == 1:
                        object_affordance_pairs_local.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

            # save object-affordance pairs for publisher
            pd.DataFrame(object_affordance_pairs_global).to_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs_global.csv', index=False)#, header=False)

    # publish semantic layer
    def publish_semantic_layer(self):
        if self.use_global_semantic_map:
            points_start = time.time()
            z = 0.0
            a = 255                    
            points = []

            # define output
            #output = self.global_semantic_map * 255.0
            output = self.global_semantic_map_inflated * 255.0
            #output = output[:, :, [2, 1, 0]] * 255.0
            output = output.astype(np.uint8)

            # draw layer
            for i in range(0, int(self.global_semantic_map_size[1])):
                for j in range(0, int(self.global_semantic_map_size[0])):
                    x = self.global_semantic_map_origin_x + i * self.global_map_resolution
                    y = self.global_semantic_map_origin_y + j * self.global_map_resolution
                    r = int(output[j, i])
                    g = int(output[j, i])
                    b = int(output[j, i])
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    pt = [x, y, z, rgb]
                    points.append(pt)

            points_end = time.time()
            print('Semantic layer runtime = ', round(points_end - points_start,3))
            
            # publish
            self.header.frame_id = 'map'
            pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
            pc2.header.stamp = rospy.Time.now()
            self.pub_semantic_layer.publish(pc2)

def main():
    # ----------main-----------
    rospy.init_node('hixron_subscriber', anonymous=False)

    # define hixron_subscriber object
    hixron_subscriber_obj = hixron_subscriber()
    # call main to initialize subscribers
    hixron_subscriber_obj.main_()

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        #print('spinning')
        rospy.spin()

main()