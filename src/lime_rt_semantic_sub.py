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
        # simulation or real-world experiment
        self.simulation = True

        # semantic variables
        self.ontology = []
        #self.semantic_global_map = []
        if self.simulation:
            # gazebo
            self.gazebo_names = []
            self.gazebo_poses = []
            self.gazebo_tags = []

        # plotting
        self.plot_costmaps = False
        self.plot_segments = False

        # declare transformation buffer
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        
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
        #self.global_plan_xs = []
        #self.global_plan_ys = []
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

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        # semantic part
        world_name = 'ont1' #'ont1-4'

        if self.simulation:
            self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

            # load gazebo tags
            self.gazebo_tags = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + world_name + '/' + 'gazebo_tags.csv')) 

        # load semantic_global_info
        #self.semantic_global_map_info = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/ontologies/' + world_name + '/' + 'info.csv', index_col=None, header=None)
        #self.semantic_global_map_info = self.semantic_global_map_info.transpose()
        #print('self.semantic_global_map_info = ', self.semantic_global_map_info)
    
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
        #self.global_plan_xs = [] 
        #self.global_plan_ys = []
        self.global_plan_tmp = []
        #self.plan_tmp = []
        #self.transformed_plan_xs = []
        #self.transformed_plan_ys = []

        for i in range(0,len(msg.poses)):
            #self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            #self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            #self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        #pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv', index=False)#, header=False)
        #pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/plan_tmp.csv', index=False)#, header=False)
        
        # it is not so important in which iteration this variable is set to False
        self.global_plan_empty = False

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv', index=False)#, header=False)
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        try:
            self.odom_tmp_copy = copy.deepcopy(self.odom_tmp)
            self.amcl_pose_tmp_copy = copy.deepcopy(self.amcl_pose_tmp)
            self.footprint_tmp_copy = copy.deepcopy(self.footprint_tmp)
            #self.robot_position_map_copy = copy.deepcopy(self.robot_position_map)
            #self.robot_orientation_map_copy = copy.deepcopy(self.robot_orientation_map)

            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            #t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
            #r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
            #r_ = np.asarray(r.as_matrix())

            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())

            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]

            # convert robot positions from /odom to /map
            #t_o_m = np.asarray([transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z])
            #r_o_m = R.from_quat([transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w])
            #r_o_m = np.asarray(r_o_m.as_matrix())
            #r_o = R.from_quat([self.robot_orientation_odom.x,self.robot_orientation_odom.y,self.robot_orientation_odom.z,self.robot_orientation_odom.w])
            #r_o = np.asarray(r_o.as_matrix())

            # position transformation
            #p = np.array([self.robot_position_odom.x, self.robot_position_odom.y, self.robot_position_odom.z])
            #pnew = p.dot(r_o_m) + t_o_m
            #self.robot_position_map.x = pnew[0]
            #self.robot_position_map.y = pnew[1]
            #self.robot_position_map.z = pnew[2]
            
            # orientation transformation
            #r_m = r_o * r_o_m
            #self.robot_orientation_map = rotationMatrixToQuaternion(r_m) #tr.quaternion_from_matrix(r_m)

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

            pd.DataFrame(self.odom_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/odom_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.amcl_pose_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.footprint_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/footprint_tmp.csv', index=False)#, header=False)
            
            pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom_tmp.csv', index=False)#, header=False)
            
            pd.DataFrame(self.local_plan_x_list).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_x_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_y_list).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_y_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_tmp.csv', index=False)#, header=False)
            
            #pd.DataFrame([self.robot_position_map.x,self.robot_position_map_copy.y,self.robot_position_map_copy.z]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_position_map.csv', index=False) #, header=False)
            #pd.DataFrame([self.robot_orientation_map.x,self.robot_orientation_map_copy.y,self.robot_orientation_map_copy.z,self.robot_orientation_map_copy.w]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_orientation_map.csv', index=False) #, header=False)

        except:
            pass
    
    # Define a callback for the footprint
    def footprint_callback(self, msg):
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
    
    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

    # Gazebo callback
    def model_state_callback(self, states_msg):
        # update robot coordinates from Gazebo/Simulation
        #robot_idx = states_msg.name.index('tiago')
        #self.robot_position_map = states_msg.pose[robot_idx].position
        #self.robot_orientation_map = states_msg.pose[robot_idx].orientation

        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')

        # robot and planning data
        #self.global_plan_tmp_copy = copy.deepcopy(self.global_plan_tmp)
        #self.local_plan_x_list_copy = copy.deepcopy(self.local_plan_x_list)
        #self.local_plan_y_list_copy = copy.deepcopy(self.local_plan_y_list)
        #self.local_plan_tmp_copy = copy.deepcopy(self.local_plan_tmp)
        #self.odom_tmp_copy = copy.deepcopy(self.odom_tmp)
        #self.amcl_pose_tmp_copy = copy.deepcopy(self.amcl_pose_tmp)
        #self.footprint_tmp_copy = copy.deepcopy(self.footprint_tmp)
        #self.robot_position_map_copy = copy.deepcopy(self.robot_position_map)
        #self.robot_orientation_map_copy = copy.deepcopy(self.robot_orientation_map)
        
        try:          
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
            self.create_perturbation_data()

            # set the local_costmap_empty var to False
            self.local_costmap_empty = False

            # increase the global counter
            self.counter_global += 1

            # save robot and planning data (could be more often, i.e. with every new local plan)
            #pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.global_plan_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.local_plan_x_list_copy).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_x_list.csv', index=False)#, header=False)
            #pd.DataFrame(self.local_plan_y_list_copy).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_y_list.csv', index=False)#, header=False)
            #pd.DataFrame(self.local_plan_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.odom_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/odom_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.amcl_pose_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose_tmp.csv', index=False)#, header=False)
            #pd.DataFrame(self.footprint_tmp_copy).to_csv(self.dirCurr + '/' + self.dirData + '/footprint_tmp.csv', index=False)#, header=False)
            #pd.DataFrame([self.robot_position_map_copy.x,self.robot_position_map_copy.y,self.robot_position_map_copy.z]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_position_map.csv', index=False) #, header=False)
            #pd.DataFrame([self.robot_orientation_map_copy.x,self.robot_orientation_map_copy.y,self.robot_orientation_map_copy.z,self.robot_orientation_map_copy.w]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_orientation_map.csv', index=False) #, header=False)
            #pd.DataFrame([self.robot_position_odom.x,self.robot_position_odom.y,self.robot_position_odom.z]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_position_map.csv', index=False) #, header=False)
            #pd.DataFrame([self.robot_orientation_odom.x,self.robot_orientation_odom.y,self.robot_orientation_odom.z,self.robot_orientation_odom.w]).to_csv(self.dirCurr + '/' + self.dirData + '/robot_orientation_map.csv', index=False) #, header=False)
            
            # save map data
            pd.DataFrame(self.costmap_info_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/costmap_info_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.image).to_csv(self.dirCurr + '/' + self.dirData + '/image.csv', index=False) #, header=False)
            pd.DataFrame(self.segments).to_csv(self.dirCurr + '/' + self.dirData + '/segments.csv', index=False)#, header=False)
            pd.DataFrame(self.segments_inflated).to_csv(self.dirCurr + '/' + self.dirData + '/segments_inflated.csv', index=False)#, header=False)
            #pd.DataFrame(self.data).to_csv(self.dirCurr + '/' + self.dirData + '/data.csv', index=False)#, header=False)

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
        self.segments_inflated[self.image == 0] = 0

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
            segs = np.flip(self.segments_inflated, axis=0)
            self.ax.imshow(segs.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'segments_inflated_without_labels' + '.png', transparent=False)
            #self.fig.clf()
            pd.DataFrame(segs).to_csv(dirCurr + '/segments_inflated.csv', index=False)#, header=False)
            
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

            self.fig.savefig(dirCurr + '/' + 'segments_with_labels' + '.png', transparent=False)
            self.fig.clf()

            pd.DataFrame(centroids_segments).to_csv(dirCurr + '/centroids_segments.csv', index=False)#, header=False)
            
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

            self.fig.savefig(dirCurr + '/' + 'segments_inflated_with_labels' + '.png', transparent=False)
            self.fig.clf()

            pd.DataFrame(centroids_segments_inflated).to_csv(dirCurr + '/centroids_segments_inflated.csv', index=False)#, header=False)
            
            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)
            img = np.flip(self.image, axis=0)
            self.ax.imshow(img.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'local_costmap' + '.png', transparent=False)
            self.fig.clf()

            pd.DataFrame(img).to_csv(dirCurr + '/local_costmap.csv', index=False)#, header=False)

            end = time.time()
            print('SEGMENTS PLOTTING TIME = ' + str(end-start) + ' seconds!!!')

        return self.segments

    # create encoded perturbations
    def create_perturbation_data(self):
        # save ontology for publisher
        pd.DataFrame(self.ontology).to_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv', index=False)#, header=False)

        # list of labels of objects in local costmap
        labels_in_lc = np.unique(self.segments)
        object_affordance_pairs = [] # [label, object, affordance]
        # get object-affordance pairs in the current lc
        for i in range(0, self.ontology.shape[0]):
            if self.ontology[i][0] in labels_in_lc:
                if self.ontology[i][6] == 1:
                    object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                if self.ontology[i][7] == 1:
                    object_affordance_pairs.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

        #self.num_samples = self.n_features + 1
        #lst = [[1]*self.n_features]
        #for i in range(1, self.num_samples):
        #    lst.append([1]*self.n_features)
        #    lst[i][i-1] = 0    
        #self.data = np.array(lst).reshape((self.num_samples, self.n_features))

        #self.n_features = len(object_affordance_pairs)
        # save object-affordance pairs for publisher
        pd.DataFrame(object_affordance_pairs).to_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv', index=False)#, header=False)

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