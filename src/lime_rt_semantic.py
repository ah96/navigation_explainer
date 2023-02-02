#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
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
from skimage.color import gray2rgb
import sklearn
import shlex
from psutil import Popen
from functools import partial
from sklearn.utils import check_random_state
import sklearn.metrics
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
import scipy as sp
from skimage.measure import regionprops
from scipy.spatial.transform import Rotation as R

# global variables
PI = math.pi

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
    def __init__(self):
        self.lime_rt_pub = lime_rt_pub()

        # semantic variables
        self.ontology = []
        #self.semantic_global_map = []
        self.gazebo_tags = []

        # plotting
        self.plot_costmaps = False
        self.plot_segments = False

        # declare transformation buffer
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        
        # gazebo
        self.gazebo_names = []
        self.gazebo_poses = []

        # directory
        self.dirCurr = os.getcwd()
        
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

        if self.plot_costmaps == True:
            self.costmap_dir = self.dirMain + '/costmap_images'
            try:
                os.mkdir(self.costmap_dir)
            except FileExistsError:
                pass

        # global counter for plotting
        self.counter_global = 0

        # robot variables
        self.robot_position_map = Point(0.0,0.0,0.0)        
        self.robot_orientation_map = Quaternion(0.0,0.0,0.0,1.0)
        self.robot_position_odom = Point(0.0,0.0,0.0)        
        self.robot_orientation_odom = Quaternion(0.0,0.0,0.0,1.0)

        # plans' variables
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.local_plan_tmp = [] 
        #self.plan_tmp = [] 
        self.global_plan_tmp = [] 
        self.global_plan_empty = True
        self.local_plan_empty = True
        
        # poses' variables
        self.footprint_tmp = []  
        self.amcl_pose_tmp = [] 
        self.odom_tmp = []
        self.odom_x = 0
        self.odom_y = 0

        # tf variables
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 

        # costmap variables
        self.segments = np.array([])
        self.data = np.array([]) 
        self.image = np.array([])
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.costmap_size = 160
        self.costmap_info_tmp = []

        # deviation
        self.local_costmap_empty = True

        # samples        
        self.num_samples = 0
        self.n_features = 0

    # Declare subscribers
    def main_(self):
        lime_rt_pub().main_()

        if self.plot_costmaps == True or self.plot_segments == True:
            self.fig = plt.figure(frameon=False)
            self.w = 1.6 * 3
            self.h = 1.6 * 3
            self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

        # subscribers
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)
        
        # semantic part
        world_name = 'ont1' #ont1-4

        # load semantic_global_info
        #self.semantic_global_map_info = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + world_name + '/' + 'info.csv', index_col=None, header=None)
        #self.semantic_global_map_info = self.semantic_global_map_info.transpose()
        #print('self.semantic_global_map_info = ', self.semantic_global_map_info)
    
        # load gazebo tags
        self.gazebo_tags = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + world_name + '/' + 'gazebo_tags.csv'))
        #print('self.gazebo_tags = ', self.gazebo_tags)
 
        # load ontology
        self.ontology = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + world_name + '/' + 'ontology.csv'))
        #print('self.ontology = ', self.ontology)
        self.openability_state_changed_objs_ont_indices = []
         
    # Define a callback for the local plan
    def odom_callback(self, msg):
        #print('odom_callback!!!')
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.robot_position_odom = msg.pose.pose.position
        self.robot_orientation_odom = msg.pose.pose.orientation
        
    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')
        
        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        #self.plan_tmp = []
        #self.transformed_plan_xs = []
        #self.transformed_plan_ys = []

        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            #self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        #pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv', index=False)#, header=False)
        #pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/plan_tmp.csv', index=False)#, header=False)
        
        # it is not so important in which iteration this variable is set to False
        self.global_plan_empty = False
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        try:
            self.local_plan_tmp = []
            self.local_plan_x_list = [] 
            self.local_plan_y_list = [] 

            for i in range(0,len(msg.poses)):
                self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.local_plan_x_list.append(x_temp)
                    self.local_plan_y_list.append(self.costmap_size - y_temp)

            # it is not so important in which iteration this variable is set to False
            self.local_plan_empty = False

        except:
            pass
    
    # Define a callback for the footprint
    def footprint_callback(self, msg):
        #print('\nfootprint_callback') 
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
    
    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        #print('amcl_callback!!!')
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

    # Gazebo callback
    def model_state_callback(self, states_msg):
        #print('gazebo_callback!!!')
        # update robot coordinates from Gazebo/Simulation
        #robot_idx = states_msg.name.index('tiago')
        #self.robot_position_map = states_msg.pose[robot_idx].position
        #self.robot_orientation_map = states_msg.pose[robot_idx].orientation

        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')

        self.global_plan_tmp_copy = copy.deepcopy(self.global_plan_tmp)
        self.local_plan_x_list_copy = copy.deepcopy(self.local_plan_x_list)
        self.local_plan_y_list_copy = copy.deepcopy(self.local_plan_y_list)
        self.local_plan_tmp_copy = copy.deepcopy(self.local_plan_tmp)
        self.odom_tmp_copy = copy.deepcopy(self.odom_tmp)
        self.amcl_pose_tmp_copy = copy.deepcopy(self.amcl_pose_tmp)
        self.footprint_tmp_copy = copy.deepcopy(self.footprint_tmp)
        self.robot_position_map_copy = copy.deepcopy(self.robot_position_map)
        self.robot_orientation_map_copy = copy.deepcopy(self.robot_orientation_map)
        
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            #t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
            #r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
            #r_ = np.asarray(r.as_matrix())

            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())

            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]

            # convert robot positions from /odom to /map
            t_o_m = np.asarray([transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z])
            r_o_m = R.from_quat([transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w])
            r_o_m = np.asarray(r_o_m.as_matrix())
            r_o = R.from_quat([self.robot_orientation_odom.x,self.robot_orientation_odom.y,self.robot_orientation_odom.z,self.robot_orientation_odom.w])
            r_o = np.asarray(r_o.as_matrix())

            # position transformation
            p = np.array([self.robot_position_odom.x, self.robot_position_odom.y, self.robot_position_odom.z])
            pnew = p.dot(r_o_m) + t_o_m
            self.robot_position_map.x = pnew[0]
            self.robot_position_map.y = pnew[1]
            self.robot_position_map.z = pnew[2]
            
            # orientation transformation
            r_m = r_o * r_o_m
            self.robot_orientation_map = rotationMatrixToQuaternion(r_m) #tr.quaternion_from_matrix(r_m)
                
            # save costmap data
            self.localCostmapOriginX = msg.info.origin.position.x
            self.localCostmapOriginY = msg.info.origin.position.y
            self.localCostmapResolution = msg.info.resolution
            self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create np.array image object
            self.image = np.asarray(msg.data)
            self.image.resize((msg.info.height,msg.info.width))

            if self.plot_costmaps == True:
                start = time.time()

                dirCurr = self.costmap_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass
                
                self.image_99s_100s = copy.deepcopy(self.image)
                self.image_99s_100s[self.image_99s_100s < 99] = 0        
                #self.fig = plt.figure(frameon=False)
                #w = 1.6 * 3
                #h = 1.6 * 3
                #self.fig.set_size_inches(w, h)
                self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                self.ax.set_axis_off()
                self.fig.add_axes(self.ax)
                self.image_99s_100s = np.flip(self.image_99s_100s, 0)
                self.ax.imshow(self.image_99s_100s.astype('float64'), aspect='auto')
                self.fig.savefig(dirCurr + '/' + 'local_costmap_99s_100s.png', transparent=False)
                #self.fig.clf()
                
                self.image_original = copy.deepcopy(self.image)
                #fig = plt.figure(frameon=False)
                #w = 1.6 * 3
                #h = 1.6 * 3
                #self.fig.set_size_inches(w, h)
                #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                #ax.set_axis_off()
                #self.fig.add_axes(ax)
                self.image_original = np.flip(self.image_original, 0)
                self.ax.imshow(self.image_original.astype('float64'), aspect='auto')
                self.fig.savefig(dirCurr + '/' + 'local_costmap_original.png', transparent=False)
                #self.fig.clf()
                
                self.image_100s = copy.deepcopy(self.image)
                self.image_100s[self.image_100s != 100] = 0
                #fig = plt.figure(frameon=False)
                #w = 1.6 * 3
                #h = 1.6 * 3
                #self.fig.set_size_inches(w, h)
                #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                #ax.set_axis_off()
                #self.fig.add_axes(ax)
                self.image_100s = np.flip(self.image_100s, 0)
                self.ax.imshow(self.image_100s.astype('float64'), aspect='auto')
                self.fig.savefig(dirCurr + '/' + 'local_costmap_100s.png', transparent=False)
                #self.fig.clf()
                
                self.image_99s = copy.deepcopy(self.image)
                self.image_99s[self.image_99s != 99] = 0
                #fig = plt.figure(frameon=False)
                #w = 1.6 * 3
                #h = 1.6 * 3
                #self.fig.set_size_inches(w, h)
                #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                #ax.set_axis_off()
                #self.fig.add_axes(self.ax)
                self.image_99s = np.flip(self.image_99s, 0)
                self.ax.imshow(self.image_99s.astype('float64'), aspect='auto')
                self.fig.savefig(dirCurr + '/' + 'local_costmap_99s.png', transparent=False)
                #self.fig.clf()
                
                self.image_less_than_99 = copy.deepcopy(self.image)
                self.image_less_than_99[self.image_less_than_99 >= 99] = 0
                #fig = plt.figure(frameon=False)
                #w = 1.6 * 3
                #h = 1.6 * 3
                #self.fig.set_size_inches(w, h)
                #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                #ax.set_axis_off()
                #self.fig.add_axes(self.ax)
                self.image_less_than_99 = np.flip(self.image_less_than_99, 0)
                self.ax.imshow(self.image_less_than_99.astype('float64'), aspect='auto')
                self.fig.savefig(dirCurr + '/' + 'local_costmap_less_than_99.png', transparent=False)
                self.fig.clf()
                
                end = time.time()
                print('COSTMAPS PLOTTING TIME = ' + str(end-start) + ' seconds!!!')
                
            # Turn inflated area to free space and 100s to 99s
            self.image[self.image == 100] = 99
            self.image[self.image <= 98] = 0

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            #self.image = self.image * 1.0

            # find lc coordinates of robot's odometry coordinates 
            self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

            # find and save egments
            self.segments = self.find_segments()

            # create and save encoded perturbations
            self.create_data()

            # set the local_costmap_empty var to False
            self.local_costmap_empty = False

            # to explain from another callback function, track copies (faster changing vars), and current callback vars
            start = time.time()
            self.lime_rt_pub.explain(self.global_plan_tmp_copy, self.local_plan_x_list_copy, self.local_plan_y_list_copy, self.local_plan_tmp_copy, 
            self.odom_tmp_copy, self.amcl_pose_tmp_copy, self.footprint_tmp_copy, self.robot_position_map_copy, self.robot_orientation_map_copy, self.costmap_info_tmp, 
            self.segments, self.segments_inflated, self.data, self.tf_map_odom_tmp, self.tf_odom_map_tmp, self.object_affordance_pairs, self.ontology, self.image)
            end = time.time()
            print('\n\nEXPLANATION TIME = ', end-start)

            # increase the global counter
            self.counter_global += 1

        except Exception as e:
            print('exception = ', e)
            return

    # find segments -- do segmentation
    def find_segments(self):
        #print('self.gazebo_names: ', self.gazebo_names)
        #print('self.gazebo_poses: ', self.gazebo_poses)

        self.openability_state_changed_objs_ont_indices = -1

        # update ontology
        # check if any object changed its position (influenced by the affordance state flip) -- currently from simulation, later ideally by object detection, recognition and tracking
        for i in range(0, self.ontology.shape[0]):
            # if the object is movable (or openable), then it may have changed its state 
            if self.ontology[i][6] == 1 or self.ontology[i][7] == 1:
                obj_gazebo_name = self.gazebo_tags[i][0]
                obj_idx = self.gazebo_names.index(obj_gazebo_name)

                obj_x_size = copy.deepcopy(self.ontology[i][4])
                obj_y_size = copy.deepcopy(self.ontology[i][5])

                obj_x = self.ontology[i][2]
                obj_y = self.ontology[i][3]
                
                # update ontology
                # almost every object type in Gazebo has different center of mass
                # in object detection, recognition and tracking algorithm it should be the same point for every object
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

                        if i not in self.openability_state_changed_objs_ont_indices:
                            self.openability_state_changed_objs_ont_indices.append(i)
                else:
                    # check whether the (centroid) coordinates of the object are changed (enough)
                    diff_x = abs(self.gazebo_poses[obj_idx].position.x - obj_x)
                    diff_y = abs(self.gazebo_poses[obj_idx].position.y - obj_y)
                    if diff_x > 0.1 or diff_y > 0.1:
                        #print('Object ' + self.ontology[i][1] + ' (' + obj_gazebo_name + ') changed its position')
                        self.ontology[i][2] = self.gazebo_poses[obj_idx].position.x
                        self.ontology[i][3] = self.gazebo_poses[obj_idx].position.y 

        if self.openability_state_changed_objs_ont_indices >= 0:
            print('openability_state_changed_objs_ont_indices_name: ', self.ontology[self.openability_state_changed_objs_ont_indices][1])

        # GET transformations between coordinate frames
        # tf from map to odom
        t_mo = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r_mo = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_mo = np.asarray(r_mo.as_matrix())

        # tf from odom to map
        t_om = np.asarray([self.tf_odom_map_tmp[0],self.tf_odom_map_tmp[1],self.tf_odom_map_tmp[2]])
        r_om = R.from_quat([self.tf_odom_map_tmp[3],self.tf_odom_map_tmp[4],self.tf_odom_map_tmp[5],self.tf_odom_map_tmp[6]])
        r_om = np.asarray(r_om.as_matrix())


        # GET local costmap points in the map frame
        # convert LC points from /odom to /map
        #'''
        #lc_c_odom_x = self.localCostmapOriginX + 0.5 * self.costmap_size * self.localCostmapResolution
        #lc_c_odom_y = self.localCostmapOriginY + 0.5 * self.costmap_size * self.localCostmapResolution
        #lc_p_odom = np.array([lc_c_odom_x, lc_c_odom_y, 0.0])
        #lc_p_map = lc_p_odom.dot(r_om) + t_om
        #lc_c_map_x = lc_p_map[0]
        #lc_c_map_y = lc_p_map[1]
        #print('(lc_c_map_x, lc_c_map_y) = ', (lc_c_map_x, lc_c_map_y))
        #'''
        # LC origin is a bottom-left point
        lc_bl_odom_x = self.localCostmapOriginX
        lc_bl_odom_y = self.localCostmapOriginY
        lc_p_odom = np.array([lc_bl_odom_x, lc_bl_odom_y, 0.0])
        lc_p_map = lc_p_odom.dot(r_om) + t_om
        lc_map_bl_x = lc_p_map[0]
        lc_map_bl_y = lc_p_map[1]
        # LC's top-right point
        lc_tr_odom_x = self.localCostmapOriginX + self.costmap_size * self.localCostmapResolution
        lc_tr_odom_y = self.localCostmapOriginY + self.costmap_size * self.localCostmapResolution
        lc_p_odom = np.array([lc_tr_odom_x, lc_tr_odom_y, 0.0])
        lc_p_map = lc_p_odom.dot(r_om) + t_om
        lc_map_tr_x = lc_p_map[0]
        lc_map_tr_y = lc_p_map[1]
        # LC sides in the map frame
        lc_map_left = lc_map_bl_x
        lc_map_right = lc_map_tr_x
        lc_map_bottom = lc_map_bl_y
        lc_map_top = lc_map_tr_y
        #print('(lc_map_left, lc_map_right, lc_map_bottom, lc_map_top) = ', (lc_map_left, lc_map_right, lc_map_bottom, lc_map_top))

        ###### SEGMENTATION PART ######
        start = time.time()
        self.segments = np.zeros(self.image.shape)
        self.segments_inflated = np.zeros(self.image.shape)
        widening_factor = 0
        for i in range(0, self.ontology.shape[0]):
            x_size = self.ontology[i][4]
            y_size = self.ontology[i][5]

            c_map_x = self.ontology[i][2]
            c_map_y = self.ontology[i][3]
            # transform centroids from /map to /odom frame
            #p_map = np.array([c_map_x, c_map_y, 0.0])
            #p_odom = p_map.dot(r_mo) + t_mo
            # centroids of a known object in /lc frame
            #c_pixel_x = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            #c_pixel_y = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)

            # object"s vertices from /map to /odom and /lc
            tl_map_x = c_map_x - 0.5*x_size
            tl_map_y = c_map_y + 0.5*y_size
            #p_map = np.array([tl_map_x, tl_map_y, 0.0])
            #p_odom = p_map.dot(r_mo) + t_mo
            #tl_odom_x = p_odom[0]
            #tl_odom_y = p_odom[1]
            #tl_pixel_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            #tl_pixel_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
            
            tr_map_x = c_map_x + 0.5*x_size
            tr_map_y = c_map_y + 0.5*y_size
            p_map = np.array([tr_map_x, tr_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo
            tr_odom_x = p_odom[0]
            tr_odom_y = p_odom[1]
            tr_pixel_x = int((tr_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            tr_pixel_y = int((tr_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y - 0.5*y_size
            p_map = np.array([bl_map_x, bl_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo
            bl_odom_x = p_odom[0]
            bl_odom_y = p_odom[1]
            bl_pixel_x = int((bl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            bl_pixel_y = int((bl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

            br_map_x = c_map_x + 0.5*x_size
            br_map_y = c_map_y - 0.5*y_size
            #p_map = np.array([br_map_x, br_map_y, 0.0])
            #p_odom = p_map.dot(r_mo) + t_mo
            #br_odom_x = p_odom[0]
            #br_odom_y = p_odom[1]
            #br_pixel_x = int((br_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            #br_pixel_y = int((br_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        
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
                x_2 = min(self.costmap_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = min(self.costmap_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True

            # top-left(tl) in LC
            elif lc_map_left < tl_map_x < lc_map_right and lc_map_bottom < tl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.costmap_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True

            # bottom-left(bl) in LC
            elif lc_map_left < bl_map_x < lc_map_right and lc_map_bottom < bl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.costmap_size-1,object_right)
                y_1 = object_bottom
                y_2 = min(self.costmap_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_lc = True
                
            # bottom-right(br) in LC
            elif lc_map_left < br_map_x < lc_map_right and lc_map_bottom < br_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = object_bottom
                y_2 = min(self.costmap_size-1,object_top)
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
                #print('at least 1 obstacle_in_lc')    
                self.segments[max(0, y_1-widening_factor):min(self.costmap_size-1, y_2+widening_factor), max(0,x_1-widening_factor):min(self.costmap_size-1, x_2+widening_factor)] = i+1
                
                # inflate using heuristics
                inflation_x = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                inflation_y = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2)                 
                self.segments_inflated[max(0, y_1-inflation_y):min(self.costmap_size-1, y_2+inflation_y), max(0,x_1-inflation_x):min(self.costmap_size-1, x_2+inflation_x)] = i+1
       
        end = time.time()
        print('semantic_segmentation_time = ' + str(end-start) + ' seconds!')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(self.segments.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_segments = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_segments.append([v,cx,cy,self.ontology[v-1][1]])

        # inflate using the remaining obstacle points of the local costmap
        for i in range(self.segments.shape[0]):
            for j in range(0, self.segments.shape[1]):
                if self.image[i, j] > 98 and self.segments_inflated[i, j] == 0:
                    distances_to_centroids = []
                    distances_indices = []
                    for k in range(0, len(centroids_segments)):
                        dx = abs(j - centroids_segments[k][1])
                        dy = abs(i - centroids_segments[k][2])
                        distances_to_centroids.append(dx + dy) # L1
                        #distances_to_centroids.append(math.sqrt(dx**2 + dy**2)) # L2
                        distances_indices.append(k)
                    idx = distances_to_centroids.index(min(distances_to_centroids))
                    self.segments_inflated[i, j] = centroids_segments[idx][0]

        # turn pixels in the inflated segments, which are zero in the local costmap, to zero
        #self.segments_inflated[self.image == 0] = 0

        # plot segments
        if self.plot_segments == True:
            start = time.time()

            # find centroids_in_LC of the objects' areas
            lc_regions = regionprops(self.segments_inflated.astype(int))
            #print('\nlen(lc_regions) = ', len(lc_regions))
            centroids_segments_inflated = []
            for lc_region in lc_regions:
                v = lc_region.label
                cy, cx = lc_region.centroid
                centroids_segments_inflated.append([v,cx,cy,self.ontology[v-1][1]])

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
            segs = np.flip(self.segments, axis=0)
            self.ax.imshow(segs.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'segments_without_labels_' + str(self.counter_global) + '.png', transparent=False)
            #self.fig.clf()


            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            #self.ax.set_axis_off()
            #self.fig.add_axes(self.ax)
            segs = np.flip(self.segments_inflated, axis=0)
            self.ax.imshow(segs.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'segments_inflated_without_labels_' + str(self.counter_global) + '.png', transparent=False)
            #self.fig.clf()


            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            #self.ax.set_axis_off()
            #self.fig.add_axes(self.ax)
            segs = np.flip(self.segments, axis=0)
            self.ax.imshow(segs.astype('float64'), aspect='auto')

            for i in range(0, len(centroids_segments)):
                self.ax.scatter(centroids_segments[i][1], self.costmap_size - centroids_segments[i][2], c='white', marker='o')   
                self.ax.text(centroids_segments[i][1], self.costmap_size - centroids_segments[i][2], centroids_segments[i][3], c='white')

            self.fig.savefig(dirCurr + '/' + 'segments_with_labels_' + str(self.counter_global) + '.png', transparent=False)
            self.fig.clf()

            
            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)
            segs = np.flip(self.segments_inflated, axis=0)
            self.ax.imshow(segs.astype('float64'), aspect='auto')

            for i in range(0, len(centroids_segments_inflated)):
                self.ax.scatter(centroids_segments_inflated[i][1], self.costmap_size - centroids_segments_inflated[i][2], c='white', marker='o')   
                self.ax.text(centroids_segments_inflated[i][1], self.costmap_size - centroids_segments_inflated[i][2], centroids_segments_inflated[i][3], c='white')

            self.fig.savefig(dirCurr + '/' + 'segments_inflated_with_labels_' + str(self.counter_global) + '.png', transparent=False)
            self.fig.clf()

            
            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)
            img = np.flip(self.image, axis=0)
            self.ax.imshow(img.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'local_costmap_' + str(self.counter_global) + '.png', transparent=False)
            self.fig.clf()

            end = time.time()
            print('SEGMENTS PLOTTING TIME = ' + str(end-start) + ' seconds!!!')

        return self.segments

    # create encoded perturbations
    def create_data(self):
        # N+1 perturbations (N - num of object-affordance pairs)

        # save ontology for publisher
        #pd.DataFrame(self.ontology).to_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv', index=False)#, header=False)

        # list of labels of objects in local costmap
        labels_in_lc = np.unique(self.segments)
        self.object_affordance_pairs = [] # [label, object, affordance]
        # get object-affordance pairs in the current lc
        for i in range(0, self.ontology.shape[0]):
            if self.ontology[i][0] in labels_in_lc:
                if self.ontology[i][6] == 1:
                    self.object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                if self.ontology[i][7] == 1:
                    self.object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'openability'])


        self.n_features = len(self.object_affordance_pairs)
        # save object-affordance pairs for publisher
        #pd.DataFrame(self.object_affordance_pairs).to_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv', index=False)#, header=False)

        self.num_samples = self.n_features + 1
        lst = [[1]*self.n_features]
        for i in range(1, self.num_samples):
            lst.append([1]*self.n_features)
            lst[i][i-1] = 0    
        self.data = np.array(lst).reshape((self.num_samples, self.n_features))

    # Inflate the costmap
    def inflateCostmap(self):
        start = time.time()

        self.inflatedCostmap = copy.deepcopy(self.image)
        self.inflatedCostmap[self.inflatedCostmap == 100] = 99

        inscribed_radius = 11.1
        r,c = np.nonzero(copy.deepcopy(self.image_100s)) # r - vertical, c - horizontal
        for j in range(0, self.costmap_size):
            for i in range(0, self.costmap_size):
                if self.inflatedCostmap[j, i] < 99:
                    distance_from_obstacle = math.sqrt(((r - j)**2 + (c - i)**2).min())
                    if distance_from_obstacle <= inscribed_radius:
                        self.inflatedCostmap[j, i] = 99        

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
        #self.image_99s_100s = np.flip(self.image_99s_100s, 0)
        ax.imshow(self.inflatedCostmap.astype('float64'), aspect='auto')
        fig.savefig(dirCurr + '/' + 'inflated_costmap.png', transparent=False)
        fig.clf()


# convert orientation quaternion to euler angles
def quaternion_to_euler(x, y, z, w):
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return [yaw, pitch, roll]
# convert euler angles to orientation quaternion
def euler_to_quaternion(roll, pitch, yaw):
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  return Quaternion(qx, qy, qz, qw)

class QSR(object):
    # constructor
    def __init__(self):
        self.binary_qsr_choice = 1
        self.ternary_qsr_choice = 3
        self.PI = PI
        self.R = 1    

    # define binary QSR calculus
    def defineBinaryQsrCalculus(self):
        if self.binary_qsr_choice == 0:
            self.binary_qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.binary_qsr_choice == 1:
            self.binary_qsr_dict = {
                'left': 0,
                'right': 1,
                'front': 2,
                'back': 3
            }

        elif self.binary_qsr_choice == 2:
            self.binary_qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        # used for deriving NLP annotations
        self.binary_qsr_dict_inv = {v: k for k, v in self.binary_qsr_dict.items()}

    # get binary QSR value
    def getBinaryQsrValue(self, angle):
        value = ''    

        if self.binary_qsr_choice == 0:
            if -self.PI/2 <= angle < self.PI/2:
                value += 'right'
            elif self.PI/2 <= angle < self.PI or -self.PI <= angle < -self.PI/2:
                value += 'left'

        elif self.binary_qsr_choice == 1:
            if 3*self.PI/4 <= angle < self.PI or -self.PI <= angle < -3*self.PI/4:
                value += 'left'
            elif -self.PI/4 <= angle < self.PI/4:
                value += 'right'
            elif self.PI/4 <= angle < 3*self.PI/4:
                value += 'front'
            elif -3*self.PI/4 <= angle < -self.PI/4:
                value += 'back'

        elif self.binary_qsr_choice == 2:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        return value

    # define ternary QSR calculus
    def defineTernaryQsrCalculus(self):
        if self.ternary_qsr_choice == 0:
            self.ternary_qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.ternary_qsr_choice == 1:
            self.ternary_qsr_dict = {
                'left': 0,
                'right': 1,
                'front': 2,
                'back': 3
            }

        elif self.ternary_qsr_choice == 2:
            self.ternary_qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        elif self.ternary_qsr_choice == 3:
            self.ternary_qsr_dict = {
                'cl': 0,
                'dl': 1,
                'cr': 2,
                'dr': 3,
                'cb': 4,
                'db': 5,
                'cf': 6,
                'df': 7
                }

        # used for deriving NLP annotations
        self.ternary_qsr_dict_inv = {v: k for k, v in self.ternary_qsr_dict.items()}

    # get ternary QSR value
    def getTernaryQsrValue(self, r, angle, R):
        value = ''    

        if self.ternary_qsr_choice == 0:
            if -self.PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif self.ternary_qsr_choice == 1:
            if -self.PI/4 <= angle <= self.PI/4:
                value += 'front'
            elif self.PI/4 < angle < 3*self.PI/4:
                value += 'left'
            elif 3*self.PI/4 <= angle or angle <= -3*self.PI/4:
                value += 'back'
            elif -3*self.PI/4 < angle < -self.PI/4:
                value += 'right'     

        elif self.ternary_qsr_choice == 2:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        elif self.ternary_qsr_choice == 3:
            if r<=R:
                value += 'c'
            else:
                value += 'd'

            if -self.PI/4 <= angle <= self.PI/4:
                value += 'front'
            elif self.PI/4 < angle < 3*self.PI/4:
                value += 'left'
            elif 3*self.PI/4 <= angle or angle <= -3*self.PI/4:
                value += 'back'
            elif -3*self.PI/4 < angle < -self.PI/4:
                value += 'right' 

        return value

class LimeBase(object):
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        #if self.verbose:
        #    print('Intercept', easy_model.intercept_)
        #    print('Prediction_local', local_pred,)
        #    print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
    
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

class ImageExplanation(object):
    def __init__(self, image, segments, object_affordance_pairs, ontology):
        self.image = image
        self.segments = segments
        self.object_affordance_pairs = object_affordance_pairs
        self.ontology = ontology
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

        self.color_free_space = False
        self.use_maximum_weight = True
        self.all_weights_zero = False


        self.val_low = 0.0
        self.val_high = 1.0
        self.free_space_shade = 0.7

    def get_image_and_mask(self, label):

        print('GET IMAGE AND MASK STARTING!!!!')
        #print('self.segments.shape: ', self.segments.shape)
        #print('self.local_exp: ', self.local_exp)
        #print('label: ', label)
        #print('self.ontology: ', self.ontology)
        print('self.object_affordance_pairs: ', self.object_affordance_pairs)

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        exp = self.local_exp[label]

        temp = np.zeros(self.image.shape)

        w_sum = 0.0
        w_s_abs = []
        for f, w in exp:
            w_sum += abs(w)
            w_s_abs.append(abs(w))
        max_w = max(w_s_abs)
        if max_w == 0:
            self.all_weights_zero = True
            #temp[self.image == 0] = self.free_space_shade
            #temp[self.image != 0] = 0.0
            #return temp, exp

        segments_labels = np.unique(self.segments)
        print('segments_labels: ', segments_labels)

        w_s = [0]*len(exp)
        f_s = [0]*len(exp)
        imgs = [np.zeros(self.image.shape)]*len(exp)
        for f, w in exp:
            print('(f, w): ', (f, w))
            print(self.object_affordance_pairs[f][1] + '_' + self.object_affordance_pairs[f][2] + ' has weight ' + str(w))
            w_s[f] = w

            temp = np.zeros(self.image.shape)
            v = self.object_affordance_pairs[f][0]

            # color free space with gray
            temp[self.segments == 0, 0] = self.free_space_shade
            temp[self.segments == 0, 1] = self.free_space_shade
            temp[self.segments == 0, 2] = self.free_space_shade

            # color the obstacle-affordance pair with green or white
            if w > 0:
                temp[self.segments == v, 1] = self.val_high
            elif w < 0:
                temp[self.segments == v, 0] = self.val_high

            imgs[f] = temp    

        print('weights = ', w_s)

        """
        for f, w in exp:
            #print('(f, w): ', (f, w))
            f = segments_labels[f]
            #print('segments_labels[f] = ', f)

            if w < 0.0:
                c = -1
            elif w > 0.0:
                c = 1
            else:
                c = 0
            #print('c = ', c)
            
            # free space
            if f == 0:
                #print('free_space, (f, w) = ', (f, w))
                if self.color_free_space == False:
                    temp[segments == f, 0] = self.free_space_shade
                    temp[segments == f, 1] = self.free_space_shade
                    temp[segments == f, 2] = self.free_space_shade
            # obstacle
            else:
                if self.color_free_space == False:
                    if c == 1:
                        temp[segments == f, 0] = 0.0
                        if self.use_maximum_weight == True:
                            temp[segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[segments == f, 2] = 0.0
                    elif c == 0:
                        temp[segments == f, 0] = 0.0
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
        """

        print('GET IMAGE AND MASK ENDING!!!!')

        return imgs, exp, w_s

class lime_rt_pub(object):
    # Constructor
    def __init__(self):
        self.counter_global = 0

        self.publish_explanation_coeffs = False  
        self.publish_explanation_image = False

        self.plot_explanation = False
        self.plot_perturbations = False 
        self.plot_classification = False

        self.hard_obstacle = 99
        
        self.qsr = QSR()
        
        self.dirCurr = os.getcwd()

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

        if self.plot_explanation == True:
            self.explanation_dir = self.dirMain + '/explanation_images'
            try:
                os.mkdir(self.explanation_dir)
            except FileExistsError:
                pass

        if self.plot_perturbations == True: 
            self.perturbation_dir = self.dirMain + '/perturbation_images'
            try:
                os.mkdir(self.perturbation_dir)
            except FileExistsError:
                pass
        
        if self.plot_classification == True:
            self.classifier_dir = self.dirMain + '/classifier_images'
            try:
                os.mkdir(self.classifier_dir)
            except FileExistsError:
                pass

        # plans' variables
        self.global_plan_empty = True
        self.local_plan_empty = True
        self.local_plan_counter = 0

        # costmap variables
        self.labels = np.array([]) 
        self.distances = np.array([])
        self.costmap_size = 160
        self.pd_image_size = (self.costmap_size,self.costmap_size) 
        self.local_costmap_empty = True
   
        # deviation
        self.original_deviation = 0

        # samples variables
        self.num_samples = 0
        self.n_features = 0

        # LIME variables
        kernel_width=.25
        kernel=None
        feature_selection='auto'
        random_state=None
        verbose=True
        kernel_width = float(kernel_width)
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        random_state = check_random_state(random_state)    
        feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, random_state=random_state)
        
        # header
        self.header = Header()

    # initialize publishers
    def main_(self):
        if self.publish_explanation_image:
            self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
            self.br = CvBridge()

        if self.publish_explanation_coeffs:
            # N_segments * (label, coefficient) + (original_deviation)
            self.pub_lime = rospy.Publisher("/lime_rt_exp", Float32MultiArray, queue_size=10)

    # flip matrix horizontally or vertically
    def matrixflip(self,m,d):
        if d=='h':
            tempm = np.fliplr(m)
            return(tempm)
        elif d=='v':
            tempm = np.flipud(m)
            return(tempm)

    # save data for local planner
    def saveImageDataForLocalPlanner(self):
        # Saving data to .csv files for C++ node - local navigation planner
        
        try:
            # Save footprint instance to a file
            self.footprint_tmp_pd = pd.DataFrame(self.footprint_tmp)
            self.footprint_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_original_pd = pd.DataFrame(self.local_plan_original)
            self.local_plan_original_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp_pd = pd.DataFrame(self.plan_tmp)
            self.plan_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp_pd = pd.DataFrame(self.global_plan_tmp)
            self.global_plan_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)

            # Save costmap_info instance to file
            self.costmap_info_tmp_pd = pd.DataFrame(self.costmap_info_tmp).transpose()
            self.costmap_info_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_tmp_pd = pd.DataFrame(self.amcl_pose_tmp).transpose()
            self.amcl_pose_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_tmp_pd = pd.DataFrame(self.tf_odom_map_tmp).transpose()
            self.tf_odom_map_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_tmp_pd = pd.DataFrame(self.tf_map_odom_tmp).transpose()
            self.tf_map_odom_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

            # Save odometry instance to file
            self.odom_tmp_pd = pd.DataFrame(self.odom_tmp).transpose()
            self.odom_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        except Exception as e:
            print('exception = ', e)
            return False

        return True

    # saving important data to class variables
    def saveImportantData2ClassVars(self):
        #print(self.local_plan_original.shape)
        #print(self.local_plan_original)
        
        #print('\nself.costmap_info_tmp = ', self.costmap_info_tmp)
        #print('\nself.odom_tmp = ', self.odom_tmp)
        #print('\nself.global_plan_tmp = ', self.global_plan_tmp)

        try:
            self.t_om = np.asarray([self.tf_odom_map_tmp[0],self.tf_odom_map_tmp[1],self.tf_odom_map_tmp[2]])
            self.r_om = R.from_quat([self.tf_odom_map_tmp[3],self.tf_odom_map_tmp[4],self.tf_odom_map_tmp[5],self.tf_odom_map_tmp[6]])
            self.r_om = np.asarray(self.r_om.as_matrix())

            self.t_mo = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
            self.r_mo = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
            self.r_mo = np.asarray(self.r_mo.as_matrix())    

            # save costmap info to class variables
            self.localCostmapOriginX = self.costmap_info_tmp[3]
            #print('self.localCostmapOriginX: ', self.localCostmapOriginX)
            self.localCostmapOriginY = self.costmap_info_tmp[4]
            #print('self.localCostmapOriginY: ', self.localCostmapOriginY)
            self.localCostmapResolution = self.costmap_info_tmp[0]
            #print('self.localCostmapResolution: ', self.localCostmapResolution)
            self.localCostmapHeight = self.costmap_info_tmp[2]
            #print('self.localCostmapHeight: ', self.localCostmapHeight)
            self.localCostmapWidth = self.costmap_info_tmp[1]
            #print('self.localCostmapWidth: ', self.localCostmapWidth)

            # save robot odometry location to class variables
            self.odom_x = self.odom_tmp[0]
            # print('self.odom_x: ', self.odom_x)
            self.odom_y = self.odom_tmp[1]
            # print('self.odom_y: ', self.odom_y)

            # save indices of robot's odometry location in local costmap to class variables
            self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
            self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
            # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

            # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
            self.x_odom_index = [self.localCostmapIndex_x_odom]
            # print('self.x_odom_index: ', self.x_odom_index)
            self.y_odom_index = [self.localCostmapIndex_y_odom]
            # print('self.y_odom_index: ', self.y_odom_index)

            #'''
            # save robot odometry orientation to class variables
            self.odom_z = self.odom_tmp[2]
            self.odom_w = self.odom_tmp[2]
            # calculate Euler angles based on orientation quaternion
            [self.yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
            
            # find yaw angles projections on x and y axes and save them to class variables
            self.yaw_odom_x = math.cos(self.yaw_odom)
            self.yaw_odom_y = math.sin(self.yaw_odom)
            #'''

            self.transformed_plan_xs = []
            self.transformed_plan_ys = []
            for i in range(0, len(self.global_plan_tmp)):
                x_temp = int((self.global_plan_tmp[i][0] - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((self.global_plan_tmp[i][1] - self.localCostmapOriginY) / self.localCostmapResolution)

                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.transformed_plan_xs.append(x_temp)
                    self.transformed_plan_ys.append(y_temp)
        
        except Exception as e:
            print('exception = ', e)
            return False

        return True
            
    # call local planner
    def create_labels(self, classifier_fn):
        try:
            if self.plot_perturbations == True:
                dirCurr = self.perturbation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass

            #self.ontology = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv'))
            
            #self.object_affordance_pairs = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv'))
            #print('self.object_affordance_pairs = ', self.object_affordance_pairs)
            
            #self.segments_inflated = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/segments_inflated.csv'))
            #print('self.segments_inflated.shape = ', self.segments_inflated.shape)

            self.labels = []
            imgs = [copy.deepcopy(self.segments_inflated)]
                
            # plot 'full' perturbation    
            if self.plot_perturbations:
                fig = plt.figure(frameon=False)
                w = 1.6 * 3
                h = 1.6 * 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(np.flipud(self.segments_inflated).astype('float64'), aspect='auto')
                fig.savefig(dirCurr + '/perturbation_full.png', transparent=False)
                fig.clf()
                
            for i in range(0, len(self.object_affordance_pairs)):
                
                # movability affordance
                if self.object_affordance_pairs[i][2] == 'movability':
                    #print('create_labels_12')
                    temp = copy.deepcopy(self.segments_inflated)
                    temp[self.segments_inflated == self.object_affordance_pairs[i][0]] = 0
                    #print('temp.shape = ', temp.shape)
                    
                    imgs.append(temp)

                # openability affordance                
                if self.object_affordance_pairs[i][2] == 'openability':
                    
                    if self.object_affordance_pairs[i][1] == 'door':
                        temp = copy.deepcopy(self.segments_inflated)

                        label = self.object_affordance_pairs[i][0]
                        #print('label = ', label)
    
                        temp[self.segments_inflated == label] = 0

                        #print('temp.shape = ', temp.shape)
                        #print('self.segments_inflated.shape = ', self.segments_inflated.shape)
                        #print('self.object_affordance_pairs = ', self.object_affordance_pairs)
                        
                        # object"s vertices from /map frame to /odom and /lc frames
                        door_tl_map_x = self.ontology[-1][2] + 0.5*self.ontology[-1][4]
                        door_tl_map_y = self.ontology[-1][3] + 0.5*self.ontology[-1][5]
                        p_map = np.array([door_tl_map_x, door_tl_map_y, 0.0])
                        p_odom = p_map.dot(self.r_mo) + self.t_mo
                        tl_odom_x = p_odom[0]
                        tl_odom_y = p_odom[1]
                        door_tl_pixel_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                        door_tl_pixel_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                        
                        x_size = int(self.ontology[label - 1][4] / self.localCostmapResolution)
                        y_size = int(self.ontology[label - 1][5] / self.localCostmapResolution)
                        #print('(x_size, y_size) = ', (x_size, y_size))

                        #temp[door_tl_pixel_y:door_tl_pixel_y + x_size, door_tl_pixel_x:door_tl_pixel_x + y_size] = label
                        inflation_x = int((max(23, x_size) - x_size) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                        inflation_y = int((max(23, y_size) - y_size) / 2)                 
                        temp[max(0, door_tl_pixel_y-inflation_x):min(self.costmap_size-1, door_tl_pixel_y+x_size+inflation_x), max(0,door_tl_pixel_x-inflation_y):min(self.costmap_size-1, door_tl_pixel_x+y_size+inflation_y)] = label
                        
                        imgs.append(temp)

                    if self.object_affordance_pairs[i][1] == 'cabinet':
                        
                        temp = copy.deepcopy(self.segments_inflated)
                        
                        label = self.object_affordance_pairs[i][0]
                        #print('label = ', label)
                        
                        # tl
                        cab_tl_map_x = self.ontology[label-1][2] - 0.5*self.ontology[label-1][4]
                        cab_tl_map_y = self.ontology[label-1][3] + 0.5*self.ontology[label-1][5]
                        p_map = np.array([cab_tl_map_x, cab_tl_map_y, 0.0])
                        p_odom = p_map.dot(self.r_mo) + self.t_mo
                        tl_odom_x = p_odom[0]
                        tl_odom_y = p_odom[1]
                        cab_tl_pixel_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                        cab_tl_pixel_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                        #print('(cab_tl_pixel_x, cab_tl_pixel_y) = ', (cab_tl_pixel_x, cab_tl_pixel_y))
                        
                        # tr
                        cab_tr_map_x = self.ontology[label-1][2] + 0.5*self.ontology[label-1][4]
                        cab_tr_map_y = self.ontology[label-1][3] + 0.5*self.ontology[label-1][5]
                        p_map = np.array([cab_tr_map_x, cab_tr_map_y, 0.0])
                        p_odom = p_map.dot(self.r_mo) + self.t_mo
                        tr_odom_x = p_odom[0]
                        tr_odom_y = p_odom[1]
                        cab_tr_pixel_x = int((tr_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                        cab_tr_pixel_y = int((tr_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                        #print('(cab_tr_pixel_x, cab_tr_pixel_y) = ', (cab_tr_pixel_x, cab_tr_pixel_y))

                        # width and height
                        x_size = int(0.2 * self.ontology[label - 1][4] / self.localCostmapResolution)
                        y_size = int(1.5 * self.ontology[label - 1][5] / self.localCostmapResolution)
                        #print('(x_size, y_size) = ', (x_size, y_size))


                        if (0 <= cab_tl_pixel_x < self.costmap_size and 0 <= cab_tl_pixel_y < self.costmap_size):                            
                            temp[cab_tl_pixel_y:min(self.costmap_size-1, cab_tl_pixel_y+y_size), max(0, cab_tl_pixel_x-x_size):cab_tl_pixel_x] = label

                        if (0 <= cab_tr_pixel_x < self.costmap_size and 0 <= cab_tl_pixel_y < self.costmap_size):
                            temp[cab_tr_pixel_y:min(self.costmap_size-1, cab_tr_pixel_y+y_size), max(0, cab_tr_pixel_x-x_size):cab_tr_pixel_x] = label
                            
                        imgs.append(temp)

                # plot perturbation
                if self.plot_perturbations:
                    fig = plt.figure(frameon=False)
                    w = 1.6 * 3
                    h = 1.6 * 3
                    fig.set_size_inches(w, h)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(np.flipud(imgs[-1]).astype('float64'), aspect='auto')
                    fig.savefig(dirCurr + '/perturbation_' + self.object_affordance_pairs[i][1] + '_' + self.object_affordance_pairs[i][2] + '.png', transparent=False)
                    fig.clf()
            
            # call predictor and store labels
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                self.labels.extend(preds)
            self.labels = np.array(self.labels)
        
        except Exception as e:
            print('e: ', e)
            return False

        return True   

    # plot local planner outputs for every perturbation
    def classifier_fn_plot(self, transformed_plan, local_plans, sampled_instance, sample_size):
        dirCurr = self.classifier_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        transformed_plan_xs = []
        transformed_plan_ys = []
        for i in range(0, transformed_plan.shape[0]):
            x_temp = int((transformed_plan[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((transformed_plan[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)

        for ctr in range(0, sample_size):
            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []

            # find if there is local plan
            local_plans_local = local_plans.loc[local_plans['ID'] == ctr]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)

            fig = plt.figure(frameon=True)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(np.flipud(sampled_instance[ctr]).astype(np.uint8))
            plt.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='x')
            plt.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='x')
            ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')
            fig.savefig(dirCurr + '/perturbation_' + str(ctr) + '.png')
            fig.clf()

    # classifier function for the explanation algorithm (LIME)
    def classifier_fn(self, sampled_instance):
        # save perturbations for the local planner
        start = time.time()
        sampled_instance_shape_len = len(sampled_instance.shape)
        #print('sampled_instance.shape = ', sampled_instance.shape)
        sample_size = 1 if sampled_instance_shape_len == 2 else sampled_instance.shape[0]
        #print('sample_size = ', sample_size)

        if sampled_instance_shape_len > 3:
            temp = np.delete(sampled_instance,2,3)
            temp = np.delete(temp,1,3)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 2:
            np.savetxt(self.dirCurr + 'src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")
        end = time.time()
        print('DATA PREP TIME = ', end-start)

        # calling ROS C++ node
        #print('\nC++ node started')

        start = time.time()
        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        #rospy.wait_for_service("/perturb_node_image/finished")

        # kill ROS node
        #Popen(shlex.split('rosnode kill /perturb_node_image'))
        
        time.sleep(0.30)
        
        end = time.time()
        print('TEB CALL TIME = ', end-start)

        #print('\nC++ node ended')

       
        start = time.time()
        
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('local_plans = ', local_plans)
        # save original local plan for qsr_rt
        #self.local_plan_full_perturbation = local_plans.loc[local_plans['ID'] == 0]
        #print('self.local_plan_full_perturbation.shape = ', self.local_plan_full_perturbation.shape)
        #local_plan_full_perturbation.to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_full_perturbation.csv', index=False)#, header=False)

        # load transformed global plan to /odom frame
        transformed_plan = np.array(pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        # save transformed plan for the qsr_rt
        #transformed_plan.to_csv(self.dirCurr + '/' + self.dirData + '/transformed_plan.csv', index=False)#, header=False)
       
        end = time.time()
        print('RESULTS LOADING TIME = ', end-start)


        local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sample_size), columns=['deviate'])

        if self.plot_classification == True:
            self.classifier_fn_plot(transformed_plan, local_plans, sampled_instance, sample_size)


        start = time.time()
        #transformed_plan = np.array(transformed_plan)

        # fill in deviation dataframe
        # transform transformed_plan to list
        #transformed_plan_xs = []
        #transformed_plan_ys = []
        #for i in range(0, transformed_plan.shape[0]):
        #    transformed_plan_xs.append(transformed_plan.iloc[i, 0])
        #    transformed_plan_ys.append(transformed_plan.iloc[i, 1])
        
        for i in range(0, sample_size):
            #print('i = ', i)
            
            #local_plan_xs = []
            #local_plan_ys = []
            
            # transform local_plan to list
            local_plan_local = np.array(local_plans.loc[local_plans['ID'] == i])
            #local_plans_local = np.array(local_plans_local)
            #for j in range(0, local_plans_local.shape[0]):
            #        local_plan_xs.append(local_plans_local.iloc[j, 0])
            #        local_plan_ys.append(local_plans_local.iloc[j, 1])
            
            # find deviation as a sum of minimal point-to-point differences
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, local_plan_local.shape[0]):
                local_diffs = []
                for k in range(0, transformed_plan.shape[0]):
                    #diff_x = (local_plans_local[j, 0] - transformed_plan[k, 0]) ** 2
                    #diff_y = (local_plans_local[j, 1] - transformed_plan[k, 1]) ** 2
                    diff_x = (local_plan_local[j][0] - transformed_plan[k][0]) ** 2
                    diff_y = (local_plan_local[j][1] - transformed_plan[k][0]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))   

            local_plan_deviation.iloc[i, 0] = sum(devs)
        end = time.time()
        print('TARGET CALC TIME = ', end-start)
        
        self.original_deviation = local_plan_deviation.iloc[0, 0]
        #print('\noriginal_deviation = ', self.original_deviation)

        cmd_vel_perturb['deviate'] = local_plan_deviation
        #return local_plan_deviation
        return np.array(cmd_vel_perturb.iloc[:, 3:])

    # explain function
    def explain(self, global_plan_tmp_copy, local_plan_x_list_copy, local_plan_y_list_copy, local_plan_tmp_copy, odom_tmp_copy, 
    amcl_pose_tmp_copy, footprint_tmp_copy, robot_position_map_copy, robot_orientation_map_copy, costmap_info_tmp, segments, segments_inflated, data,
    tf_map_odom_tmp, tf_odom_map_tmp, object_affordance_pairs, ontology, image):
    
        explain_time_start = time.time()

        self.local_plan_x_list_fixed = local_plan_x_list_copy
        self.local_plan_y_list_fixed = local_plan_y_list_copy

        self.robot_position_map = robot_position_map_copy
        self.robot_orientation_map = robot_orientation_map_copy

        self.image = image
        self.global_plan_tmp = global_plan_tmp_copy
        self.plan_tmp = global_plan_tmp_copy
        self.footprint_tmp = footprint_tmp_copy
        self.local_plan_original = local_plan_tmp_copy
        self.costmap_info_tmp = costmap_info_tmp 
        self.segments = segments, 
        self.segments_inflated = segments_inflated
        self.data = data
        self.amcl_pose_tmp = amcl_pose_tmp_copy
        self.odom_tmp = odom_tmp_copy
        self.costmap_info_tmp = costmap_info_tmp
        self.segments = segments
        self.segments_inflated = segments_inflated
        self.data = data
        self.tf_map_odom_tmp = tf_map_odom_tmp
        self.tf_odom_map_tmp = tf_odom_map_tmp
        self.object_affordance_pairs = object_affordance_pairs
        self.ontology = ontology

        # turn grayscale image to rgb image
        self.image_rgb = gray2rgb(self.image * 1.0)
        
        # saving important data to class variables
        if self.saveImportantData2ClassVars() == False:
            print('\nData not saved into variables correctly!')
            return

        # save data for teb
        if self.saveImageDataForLocalPlanner() == False:
            print('\nData not saved correctly!')
            return
        
        start = time.time()
        # call local planner
        self.labels=(0,)
        self.top = self.labels
        if self.create_labels(self.classifier_fn) == False:
            print('error while creating labels')
            return
        end = time.time()
        print('\nCALLING TEB TIME = ', end-start)
        print('labels: ', self.labels)

        # find distances
        # distance_metric = 'jaccard' - alternative distance metric
        distance_metric='cosine'
        self.distances = sklearn.metrics.pairwise_distances(
            self.data,
            self.data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        #print('self.distances: ', self.distances)

        
        # Explanation variables
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'
                
        try:
            start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.image_rgb, self.segments, self.object_affordance_pairs, self.ontology)
            if top_labels:
                top = np.argsort(self.labels[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()
            for label in top:
                (ret_exp.intercept[label],
                    ret_exp.local_exp[label],
                    ret_exp.score[label],
                    ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                    self.data, self.labels, self.distances, label, num_features,
                    model_regressor=model_regressor,
                    feature_selection=feature_selection)
            end = time.time()
            print('\nMODEL FITTING TIME = ', end-start)

            start = time.time()
            # get explanation image
            outputs, exp, weights = ret_exp.get_image_and_mask(label=0)
            end = time.time()
            print('\nGET EXP PIC TIME = ', end-start)
            print('exp: ', exp)

            centroids_for_plot = []
            lc_regions = regionprops(self.segments.astype(int))
            for lc_region in lc_regions:
                v = lc_region.label
                cy, cx = lc_region.centroid
                centroids_for_plot.append([v,cx,cy,self.ontology[v-1][1]])


            if self.plot_explanation == True:
                dirCurr = self.explanation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass

            for k in range(0, len(outputs)):
                if self.plot_explanation:
                    output = outputs[k]
                    output[:,:,0] = np.flip(output[:,:,0], axis=0)
                    output[:,:,1] = np.flip(output[:,:,1], axis=0)
                    output[:,:,2] = np.flip(output[:,:,2], axis=0)

                    fig = plt.figure(frameon=False)
                    w = 1.6 * 3
                    h = 1.6 * 3
                    fig.set_size_inches(w, h)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    #print('1')
                    ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='x')
                    #print('2')
                    ax.scatter(self.local_plan_x_list_fixed, self.local_plan_y_list_fixed, c='yellow', marker='o')
                    #print('3')
                    ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
                    #print('4')
                    ax.text(self.x_odom_index[0], self.y_odom_index[0], 'robot', c='white')
                    #print('6')
                    ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')
                    #print('7')
                    ax.imshow(output.astype('float64'), aspect='auto')
                    #print('8')
                    #print('centroids_for_plot: ', centroids_for_plot)
                    for i in range(0, len(centroids_for_plot)):
                        ax.scatter(centroids_for_plot[i][1], self.costmap_size - centroids_for_plot[i][2], c='white', marker='o')   
                        ax.text(centroids_for_plot[i][1], self.costmap_size - centroids_for_plot[i][2], centroids_for_plot[i][3], c='white')
                    #print('9')
                    #print('self.object_affordance_pairs[k][1] = ', self.object_affordance_pairs[k][1])
                    #print('type(self.object_affordance_pairs[k][1]) = ', type(self.object_affordance_pairs[k][1]))
                    #print('self.object_affordance_pairs[k][2] = ', self.object_affordance_pairs[k][2])
                    #print('type(self.object_affordance_pairs[k][2]) = ', type(self.object_affordance_pairs[k][2]))
                    #print('dirrCurr = ', dirCurr)
                    #print('type(dirrCurr) = ', type(dirCurr + '/explanation_' + self.object_affordance_pairs[k][1] + '_' + self.object_affordance_pairs[k][2] + '.png'))
                    fig.savefig(dirCurr + '/explanation_' + self.object_affordance_pairs[k][1] + '_' + self.object_affordance_pairs[k][2] + '.png', transparent=False)
                    #print('9.5')
                    fig.clf()
                    #print('10')

                # QSR part
                # robot coordinates
                #print('self.robot_position_map = ', self.robot_position_map)
                robot_x = self.robot_position_map.x
                robot_y = self.robot_position_map.y
                #self.robot_orientation_map

                # binary relation between object and robot
                label = self.object_affordance_pairs[k][0]
                object_x = self.ontology[k-1][2]
                object_y = self.ontology[k-1][3]
                d_x = object_x - robot_x 
                d_y = object_y - robot_y
                angle = np.arctan2(d_y, d_x)
                [angle_ref,pitch,roll] = quaternion_to_euler(self.robot_orientation_map.x,self.robot_orientation_map.y,self.robot_orientation_map.z,self.robot_orientation_map.w)
                angle = angle - angle_ref
                if angle >= PI:
                    angle -= 2*PI
                elif angle < -PI:
                    angle += 2*PI
                qsr_value = self.qsr.getBinaryQsrValue(angle)
                if self.object_affordance_pairs[k][2] == 'movability':
                    if weights[k] > 0.0:
                        print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and increases my deviation ' + '\n')
                    elif weights[k] < 0.0:
                        print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and decreases my deviation ' + '\n')
                if self.object_affordance_pairs[k][2] == 'openability':
                    if 'cabinet' in self.object_affordance_pairs[k][1]:
                        if weights[k] > 0.0:
                            print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and when closed increases my deviation ' + '\n')
                        elif weights[k] < 0.0:
                            print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and when open increases my deviation ' + '\n')
                    if 'door' in self.object_affordance_pairs[k][1]:
                        if weights[k] > 0.0:
                            print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and when open increases my deviation ' + '\n')
                        elif weights[k] < 0.0:
                            print(self.object_affordance_pairs[k][1] + ' is to my ' + qsr_value + ' and when closed increases my deviation ' + '\n')
            #print('\nexp = ', exp)

            #pd.DataFrame(output[:,:,0]).to_csv(self.dirCurr + '/' + self.dirData + '/output_B.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,1]).to_csv(self.dirCurr + '/' + self.dirData + '/output_G.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,2]).to_csv(self.dirCurr + '/' + self.dirData + '/output_R.csv', index=False) #, header=False)

            if self.publish_explanation_coeffs:
                # publish explanation coefficients
                exp_with_centroids = Float32MultiArray()
                segs_unique = np.unique(self.segments)
                for k in range(0, len(exp)):
                    exp_with_centroids.data.append(segs_unique[exp[k][0]])
                    exp_with_centroids.data.append(exp[k][1]) 
                exp_with_centroids.data.append(self.original_deviation) # append original deviation as the last element
                self.pub_lime.publish(exp_with_centroids) # N_segments * (label, coefficient) + (original_deviation)

            if self.publish_explanation_image:
                fig = plt.figure(frameon=False)
                w = 1.6 * 3
                h = 1.6 * 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(output_semantic.astype('float64'), aspect='auto')
                fig.savefig('explanation.png', transparent=False)
                fig.clf()

                exp_img_start = time.time()
                # publish explanation image
                #output = output[:, :, [2, 1, 0]].astype(np.uint8)
                #output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                output_semantic = output_semantic[:, :, [2, 1, 0]]#.astype(np.uint8)
                output_cv = self.br.cv2_to_imgmsg(output_semantic.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                self.pub_exp_image.publish(output_cv)

                exp_img_end = time.time()
                print('\nexp_img_time = ', exp_img_end - exp_img_start)
            
            explain_time_end = time.time()
            with open(self.dirMain + '/explanation_time.csv','a') as file:
                file.write(str(explain_time_end-explain_time_start))
                file.write('\n')
            
            self.counter_global+=1
            
        except Exception as e:
            print('Exception: ', e)
            #print('Exception - explanation is skipped!!!')
            return
        


# ----------main-----------
# main function
# Initialize the ROS Node named 'lime_rt_semantic', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_semantic', anonymous=True)

# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()