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
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
import message_filters
import torch

# lc -- local costmap

def rotationMatrixToQuaternion(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    q = Quaternion(q[0],q[1],q[2],q[3])
    return q

# lime subscriber class
class lime_rt_sub(object):
     # Constructor
    def _init_(self):
        # simulation or real-world experiment
        self.simulation = True

        # use local costmap
        self.use_local_costmap = False

        # whether to plot
        self.plot_costmaps = False
        self.plot_segments = False
        # global counter for plotting
        self.counter_global = 0
        
        # data directories
        self.dirCurr = os.getcwd()

        self.path_prefix = self.dirCurr + '/yolo_data/'
        
        self.dirMain = 'explanation_data'
        try:
            os.mkdir(self.dirMain)
        except FileExistsError:
            pass

        self.dirData = self.dirMain + '/lime_rt_data'
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_segments == True:
            self.segmentation_dir = self.dirMain + '/segmentation_images'
            try:
                os.mkdir(self.segmentation_dir)
            except FileExistsError:
                pass

        if self.plot_costmaps == True and self.use_local_costmap == True:
            self.costmap_dir = self.dirMain + '/costmap_images'
            try:
                os.mkdir(self.costmap_dir)
            except FileExistsError:
                pass

        # semantic variables
        self.ontology = []
        if self.simulation:
            # gazebo vars
            self.gazebo_names = []
            self.gazebo_poses = []
            self.gazebo_tags = []

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

        # plans' variables
        # local plan
        self.local_plan_xs = [] 
        self.local_plan_ys = []
        self.local_plan = []
        # global plan 
        self.global_plan = [] 
                
        # poses' variables
        self.footprint = []  
        self.amcl_pose = [] 
        self.odom = []
        self.odom_x = 0
        self.odom_y = 0 

        # local map vars
        self.local_map = np.array([])
        self.local_map_origin_x = 0 
        self.local_map_origin_y = 0 
        self.local_map_resolution = 0.025
        self.local_map_size = 160
        self.local_map_info = []

        # camera and semantic map variables
        self.semantic_map = np.array([])
        self.camera_image = np.array([])
        self.depth_image = np.array([])
        # camera projection matrix 
        self.P = np.array([])

    # Declare subscribers
    def main_(self):
        # if plotting==True create the base plot structure
        if (self.plot_costmaps == True and self.use_local_costmap == True) or self.plot_segments == True:
            self.fig = plt.figure(frameon=False)
            self.w = 1.6 * 3
            self.h = 1.6 * 3
            self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

        # subscribers
        # local plan subscriber
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)
        # global plan subscriber 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        # robot footprint subscriber
        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)
        # odometry subscriber
        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)
        # global-amcl pose subscriber
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)
        # local costmap subscriber
        if self.use_local_costmap == True:
            self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)
        # robot camera subscribers
        self.local_map_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image) #"32FC1"
        self.camera_info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.local_map_sub, self.depth_sub, self.camera_info_sub], 10, 1.0)
        self.ts.registerCallback(self.camera_feed_callback)

        if self.simulation == False:
            # Load YOLO model
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
            self.model = torch.hub.load(self.path_prefix + '/yolov5_master/', 'custom', self.path_prefix + '/models/yolov5s.pt', source='local')  # custom trained model

        # semantic part
        ontology_name = 'ont1' #'ont1-4'

        if self.simulation:
            # gazebo model states subscriber
            self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

            # load gazebo tags
            self.gazebo_tags = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + ontology_name + '/' + 'gazebo_tags.csv')) 
    
        # load ontology
        self.ontology = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + ontology_name + '/' + 'ontology.csv'))
        
    # camera feed callback
    def camera_feed_callback(self, img, depth_img, info):
        #print('\ncamera_feed_callback')

        # RGB IMAGE
        # image from robot's camera to np.array
        self.camera_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # Get image dimensions
        #(height, width) = image.shape[:2]

        # DEPTH IMAGE
        # convert depth image to np array
        self.depth_image = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_img.height, depth_img.width, -1)    
        # fill missing values with negative distance
        self.depth_image = np.nan_to_num(self.depth_image, nan=-1.0)
        
        # get projection matrix
        self.P = info.P

        # potential place to make semantic map, if local costmap is not used
        if self.use_local_costmap == False:
            pass

    # odometry callback
    def odom_callback(self, msg):
        #print('odom_callback')
        
        self.odom = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        
        self.robot_position_odom = msg.pose.pose.position
        self.robot_orientation_odom = msg.pose.pose.orientation
        
    # global plan callback
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')
        
        self.global_plan = []

        for i in range(0,len(msg.poses)):
            self.global_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        pd.DataFrame(self.global_plan).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan.csv', index=False)#, header=False)

        # potential place to make semantic map, if local costmap is not used
        if self.use_local_costmap == False:
            # update local_map params (origin cordinates)
            self.local_map_origin_x = self.robot_position_map.x - self.local_map_resolution * self.local_map_size * 0.5 
            self.local_map_origin_y = self.robot_position_map.y - self.local_map_resolution * self.local_map_size * 0.5
            self.local_map_info = [self.local_map_size, self.local_map_resolution, self.local_map_origin_x, self.local_map_origin_y]

            # create semantic map
            self.create_semantic_map()

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
            self.local_plan_xs = [] 
            self.local_plan_ys = [] 
            for i in range(0,len(msg.poses)):
                self.local_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.local_map_origin_x) / self.local_map_resolution)
                y_temp = int((msg.poses[i].pose.position.y - self.local_map_origin_y) / self.local_map_resolution)
                if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                    self.local_plan_xs.append(x_temp)
                    self.local_plan_ys.append(self.local_map_size - y_temp)

            # save the vars for the publisher/explainer
            pd.DataFrame(self.odom_copy).to_csv(self.dirCurr + '/' + self.dirData + '/odom.csv', index=False)#, header=False)
            pd.DataFrame(self.amcl_pose_copy).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose.csv', index=False)#, header=False)
            pd.DataFrame(self.footprint_copy).to_csv(self.dirCurr + '/' + self.dirData + '/footprint.csv', index=False)#, header=False)
            
            pd.DataFrame(self.tf_odom_map).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom.csv', index=False)#, header=False)
            
            pd.DataFrame(self.local_plan_xs).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_xs.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_ys).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_ys.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan.csv', index=False)#, header=False)

        except:
            #print('exception = ', e) # local plan frame rate is too high to print possible exceptions
            return
    
    # robot footprint callback
    def footprint_callback(self, msg):
        self.footprint = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
    
    # amcl (global) pose callback
    def amcl_callback(self, msg):
        #print('amcl_callback')
        
        self.amcl_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

        self.robot_position_map = msg.pose.pose.position
        self.robot_orientation_map = msg.pose.pose.orientation

    # Gazebo callback
    def model_state_callback(self, states_msg):
        #print('model_state_callback')  

        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

    # local costmap callback
    def local_costmap_callback(self, msg):
        #print('\nlocal_costmap_callback')
        
        try:          
            # update local_map (costmap) data
            self.local_map_origin_x = msg.info.origin.position.x
            self.local_map_origin_y = msg.info.origin.position.y
            #self.local_map_resolution = msg.info.resolution
            self.local_map_info = [self.local_map_size, self.local_map_resolution, self.local_map_origin_x, self.local_map_origin_y]

            # create np.array local_map object
            self.local_map = np.asarray(msg.data)
            self.local_map.resize((self.local_map_size,self.local_map_size))

            if self.plot_costmaps == True:
                self.plot_costmaps()
                
            # Turn inflated area to free space and 100s to 99s
            self.local_map[self.local_map == 100] = 99
            self.local_map[self.local_map <= 98] = 0

            # create semantic map
            self.create_semantic_data()

            # increase the global counter
            self.counter_global += 1

        except Exception as e:
            print('exception = ', e)
            return

    # plot costmaps
    def plot_costmaps(self):
        start = time.time()

        dirCurr = self.costmap_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass
        
        local_map_99s_100s = copy.deepcopy(self.local_map)
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
        
        local_map_original = copy.deepcopy(self.local_map)
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
        
        local_map_100s = copy.deepcopy(self.local_map)
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
        
        local_map_99s = copy.deepcopy(self.local_map)
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
        
        local_map_less_than_99 = copy.deepcopy(self.local_map)
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
        print('COSTMAPS PLOTTING TIME = ' + str(end-start) + ' seconds')

    # update ontology
    def update_ontology(self):
        # check if any object changed its position from simulation or from object detection (and tracking)

        # simulation relying on Gazebo
        if self.simulation:
            for i in range(0, self.ontology.shape[0]):
                # if the object has some affordance (etc. movability, openability), then it may have changed its position 
                if self.ontology[i][6] == 1 or self.ontology[i][7] == 1:
                    obj_gazebo_name = self.gazebo_tags[i][0]
                    obj_idx = self.gazebo_names.index(obj_gazebo_name)

                    obj_x_size = copy.deepcopy(self.ontology[i][4])
                    obj_y_size = copy.deepcopy(self.ontology[i][5])

                    obj_x = self.ontology[i][2]
                    obj_y = self.ontology[i][3]
                    
                    # update ontology
                    # almost every object type in Gazebo has a different center of mass
                    if 'table' in obj_gazebo_name:
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(self.gazebo_poses[obj_idx].position.x + 0.5*obj_x_size - obj_x)
                        diff_y = abs(self.gazebo_poses[obj_idx].position.y - 0.5*obj_y_size - obj_y)
                        if diff_x > 0.1 or diff_y > 0.1:
                            #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                            self.ontology[i][2] = self.gazebo_poses[obj_idx].position.x + 0.5*obj_x_size
                            self.ontology[i][3] = self.gazebo_poses[obj_idx].position.y - 0.5*obj_y_size 
                    elif 'wardrobe' in obj_gazebo_name:
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(self.gazebo_poses[obj_idx].position.x - 0.5*obj_x_size - obj_x)
                        diff_y = abs(self.gazebo_poses[obj_idx].position.y - 0.5*obj_y_size - obj_y)
                        if diff_x > 0.1 or diff_y > 0.1:
                            #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                            self.ontology[i][2] = self.gazebo_poses[obj_idx].position.x - 0.5*obj_x_size
                            self.ontology[i][3] = self.gazebo_poses[obj_idx].position.y - 0.5*obj_y_size 
                    elif 'door' in obj_gazebo_name:
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(self.gazebo_poses[obj_idx].position.x - obj_x)
                        diff_y = abs(self.gazebo_poses[obj_idx].position.y - obj_y)
                        if diff_x > 0.1 or diff_y > 0.1:
                            # if the doors are opened/closed they are shifted for 90 degrees
                            #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                            self.ontology[i][4] = obj_y_size
                            self.ontology[i][5] = obj_x_size

                            self.ontology[i][2] = self.gazebo_poses[obj_idx].position.x
                            self.ontology[i][3] = self.gazebo_poses[obj_idx].position.y 
                    else:
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(self.gazebo_poses[obj_idx].position.x - obj_x)
                        diff_y = abs(self.gazebo_poses[obj_idx].position.y - obj_y)
                        if diff_x > 0.1 or diff_y > 0.1:
                            #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                            self.ontology[i][2] = self.gazebo_poses[obj_idx].position.x
                            self.ontology[i][3] = self.gazebo_poses[obj_idx].position.y 

        # real world or simulation relying on object detection
        else:
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

        # save ontology for publisher
        pd.DataFrame(self.ontology).to_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv', index=False)#, header=False)
        
    # create semantic data
    def create_semantic_data(self):
        # update ontology
        self.update_ontology()

        self.create_semantic_map()

        # plot segments
        if self.plot_segments == True:
            self.plot_segments()

        # create interpretable features
        self.create_interpretable_features()

    # create semantic map
    def create_semantic_map(self):
        # GET transformations between coordinate frames
        # tf from map to odom
        t_mo = np.asarray([self.tf_map_odom[0],self.tf_map_odom[1],self.tf_map_odom[2]])
        r_mo = R.from_quat([self.tf_map_odom[3],self.tf_map_odom[4],self.tf_map_odom[5],self.tf_map_odom[6]])
        r_mo = np.asarray(r_mo.as_matrix())

        # tf from odom to map
        t_om = np.asarray([self.tf_odom_map[0],self.tf_odom_map[1],self.tf_odom_map[2]])
        r_om = R.from_quat([self.tf_odom_map[3],self.tf_odom_map[4],self.tf_odom_map[5],self.tf_odom_map[6]])
        r_om = np.asarray(r_om.as_matrix())

        # convert LC points from /odom to /map
        # LC origin is a bottom-left point
        lc_bl_odom_x = self.local_map_origin_x
        lc_bl_odom_y = self.local_map_origin_y
        lc_p_odom = np.array([lc_bl_odom_x, lc_bl_odom_y, 0.0])
        lc_p_map = lc_p_odom.dot(r_om) + t_om
        lc_map_bl_x = lc_p_map[0]
        lc_map_bl_y = lc_p_map[1]
        # LC's top-right point
        lc_tr_odom_x = self.local_map_origin_x + self.local_map_size * self.local_map_resolution
        lc_tr_odom_y = self.local_map_origin_y + self.local_map_size * self.local_map_resolution
        lc_p_odom = np.array([lc_tr_odom_x, lc_tr_odom_y, 0.0])
        lc_p_map = lc_p_odom.dot(r_om) + t_om
        lc_map_tr_x = lc_p_map[0]
        lc_map_tr_y = lc_p_map[1]
        # LC sides coordinates in the /map frame
        lc_map_left = lc_map_bl_x
        lc_map_right = lc_map_tr_x
        lc_map_bottom = lc_map_bl_y
        lc_map_top = lc_map_tr_y
        #print('(lc_map_left, lc_map_right, lc_map_bottom, lc_map_top) = ', (lc_map_left, lc_map_right, lc_map_bottom, lc_map_top))

        start = time.time()
        self.semantic_map = np.zeros(self.local_map.shape)
        self.semantic_map_inflated = np.zeros(self.local_map.shape)
        widening_factor = 0
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
            p_odom = p_map.dot(r_mo) + t_mo
            tr_odom_x = p_odom[0]
            tr_odom_y = p_odom[1]
            tr_pixel_x = int((tr_odom_x - self.local_map_origin_x) / self.local_map_resolution)
            tr_pixel_y = int((tr_odom_y - self.local_map_origin_y) / self.local_map_resolution)

            # bottom left vertex
            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y - 0.5*y_size
            p_map = np.array([bl_map_x, bl_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo
            bl_odom_x = p_odom[0]
            bl_odom_y = p_odom[1]
            bl_pixel_x = int((bl_odom_x - self.local_map_origin_x) / self.local_map_resolution)
            bl_pixel_y = int((bl_odom_y - self.local_map_origin_y) / self.local_map_resolution)

            # object's sides coordinates
            object_left = bl_pixel_x
            object_top = tr_pixel_y
            object_right = tr_pixel_x
            object_bottom = bl_pixel_y

            obstacle_in_lc = False 
            x_1 = 0
            x_2 = 0
            y_1 = 0
            y_2 = 0

            # centroid in LC
            if lc_map_left < c_map_x < lc_map_right and lc_map_bottom < c_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = min(self.local_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = min(self.local_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True

            # top-left(tl) in LC
            elif lc_map_left < tl_map_x < lc_map_right and lc_map_bottom < tl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True

            # bottom-left(bl) in LC
            elif lc_map_left < bl_map_x < lc_map_right and lc_map_bottom < bl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_map_size-1,object_right)
                y_1 = object_bottom
                y_2 = min(self.local_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True
                
            # bottom-right(br) in LC
            elif lc_map_left < br_map_x < lc_map_right and lc_map_bottom < br_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = object_bottom
                y_2 = min(self.local_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True
                
            # top-right(tr) in LC
            elif lc_map_left < tr_map_x < lc_map_right and lc_map_bottom < tr_map_y < lc_map_top:
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True

            if obstacle_in_lc == True:
                # semantic map
                self.semantic_map[max(0, y_1-widening_factor):min(self.local_map_size-1, y_2+widening_factor), max(0,x_1-widening_factor):min(self.local_map_size-1, x_2+widening_factor)] = i+1
                
                # inflate semantic map using heuristics
                inflation_x = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                inflation_y = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2)                 
                self.semantic_map_inflated[max(0, y_1-inflation_y):min(self.local_map_size-1, y_2+inflation_y), max(0,x_1-inflation_x):min(self.local_map_size-1, x_2+inflation_x)] = i+1
       
        end = time.time()
        print('semantic_segmentation_time = ' + str(end-start) + ' seconds!')


        # find centroids of the objects in the semantic map
        lc_regions = regionprops(self.semantic_map.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        self.centroids_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            self.centroids_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        if self.use_local_costmap == True:
            # inflate using the remaining obstacle points of the local costmap
            for i in range(self.semantic_map.shape[0]):
                for j in range(0, self.semantic_map.shape[1]):
                    if self.local_map[i, j] > 98 and self.semantic_map_inflated[i, j] == 0:
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

            # turn pixels in the inflated segments, which are zero in the local costmap, to zero
            self.semantic_map_inflated[self.local_map == 0] = 0

        # save local and semantic maps data
        pd.DataFrame(self.local_map_info).to_csv(self.dirCurr + '/' + self.dirData + '/local_map_info.csv', index=False)#, header=False)
        pd.DataFrame(self.local_map).to_csv(self.dirCurr + '/' + self.dirData + '/local_map.csv', index=False) #, header=False)
        pd.DataFrame(self.semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map.csv', index=False)#, header=False)
        pd.DataFrame(self.semantic_map_inflated).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map_inflated.csv', index=False)#, header=False)

    # plot segments
    def plot_segments(self):
        start = time.time()

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(self.semantic_map_inflated.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_semantic_map_inflated = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_semantic_map_inflated.append([v,cx,cy,self.ontology[v-1][1]])

        dirCurr = self.segmentation_dir + '/' + str(self.counter_global)            
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

        self.fig.savefig(dirCurr + '/' + 'segments_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/segments.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'segments_inflated_without_labels' + '.png', transparent=False)
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

        for i in range(0, len(self.centroids_semantic_map)):
            self.ax.scatter(self.centroids_semantic_map[i][1], self.local_map_size - self.centroids_semantic_map[i][2], c='white', marker='o')   
            self.ax.text(self.centroids_semantic_map[i][1], self.local_map_size - self.centroids_semantic_map[i][2], self.centroids_semantic_map[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'segments_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(self.centroids_semantic_map).to_csv(dirCurr + '/centroids_semantic_map.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        for i in range(0, len(centroids_semantic_map_inflated)):
            self.ax.scatter(centroids_semantic_map_inflated[i][1], self.local_map_size - centroids_semantic_map_inflated[i][2], c='white', marker='o')   
            self.ax.text(centroids_semantic_map_inflated[i][1], self.local_map_size - centroids_semantic_map_inflated[i][2], centroids_semantic_map_inflated[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'segments_inflated_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_semantic_map_inflated).to_csv(dirCurr + '/centroids_semantic_map_inflated.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        img = np.flip(self.local_map, axis=0)
        self.ax.imshow(img.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'local_costmap' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(img).to_csv(dirCurr + '/local_costmap.csv', index=False)#, header=False)

        end = time.time()
        print('SEGMENTS PLOTTING TIME = ' + str(end-start) + ' seconds')

    # create interpretable features
    def create_interpretable_features(self):
        # list of labels of objects in local costmap
        labels_in_lc = np.unique(self.semantic_map)
        object_affordance_pairs = [] # [label, object, affordance]
        # get object-affordance pairs in the current lc
        for i in range(0, self.ontology.shape[0]):
            if self.ontology[i][0] in labels_in_lc:
                if self.ontology[i][6] == 1:
                    object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                if self.ontology[i][7] == 1:
                    object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

        # save object-affordance pairs for publisher
        pd.DataFrame(object_affordance_pairs).to_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv', index=False)#, header=False)

    # inflate the local costmap
    def inflate_local_costmap(self):
        start = time.time()

        inflated_local_map = copy.deepcopy(self.local_map)
        inflated_local_map[inflated_local_map == 100] = 99

        local_map_100s = copy.deepcopy(self.local_map)
        local_map_100s[local_map_100s != 100] = 0
        
        inscribed_radius = 11.1
        r,c = np.nonzero(copy.deepcopy(local_map_100s)) # r - vertical, c - horizontal
        for j in range(0, self.local_map_size):
            for i in range(0, self.local_map_size):
                if self.inflated_local_map[j, i] < 99:
                    distance_from_obstacle = math.sqrt(((r - j)**2 + (c - i)**2).min())
                    if distance_from_obstacle <= inscribed_radius:
                        self.inflated_local_map[j, i] = 99        

        end = time.time()
        print('COSTMAP INFLATION RUNTIME = ', end-start)

        dirCurr = self.costmap_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(inflated_local_map.astype('float64'), aspect='auto')
        fig.savefig(dirCurr + '/' + 'inflated_local_costmap.png', transparent=False)
        fig.clf()
                


# ----------main-----------
# main function
# Initialize the ROS Node named 'lime_rt_semantic_sub', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_semantic_sub', anonymous=True)

# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()