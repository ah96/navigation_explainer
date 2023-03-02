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
import math
from skimage.measure import regionprops
from geometry_msgs.msg import Point, Quaternion
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
from skimage.segmentation import slic
from skimage.color import gray2rgb

# global variables
PI = math.pi

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


# lime subscriber class
class lime_rt_sub(object):
    # constructor
    def __init__(self):
        self.lime_rt_pub = lime_rt_pub()

        # plotting
        self.plot_costmaps_bool = True
        self.plot_segments_bool = True

        # declare transformation buffer
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        
        # gazebo
        self.gazebo_names = []
        self.gazebo_poses = []

        # directories
        self.dir_curr = os.getcwd()
        
        self.dir_main = 'explanation_data'
        try:
            os.mkdir(self.dir_main)
        except FileExistsError:
            pass

        self.dirData = self.dir_main + '/lime_rt_data'
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_segments_bool == True:
            self.segmentation_dir = self.dir_main + '/segmentation_images'
            try:
                os.mkdir(self.segmentation_dir)
            except FileExistsError:
                pass

        if self.plot_costmaps_bool == True:
            self.costmap_dir = self.dir_main + '/costmap_images'
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
        self.footprint = []  
        self.amcl_pose = [] 
        self.odom = []
        self.odom_x = 0
        self.odom_y = 0

        # plans' variables
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_xs = [] 
        self.local_plan_ys = []
        self.local_plan = [] 
        self.global_plan = [] 

        # tf variables
        self.tf_odom_map = [] 
        self.tf_map_odom = [] 

        # costmap variables
        self.local_costmap = np.array([])
        self.local_costmap_origin_x = 0 
        self.local_costmap_origin_y = 0 
        self.local_costmap_resolution = 0
        self.local_costmap_size = 160
        self.local_costmap_info = []
        self.segments = np.array([])
 
        # perturbation variables
        self.data = np.array([])

        # samples        
        self.n_samples = 0
        self.n_features = 0

    # declare subscribers
    def main_(self):
        lime_rt_pub().main_()

        if self.plot_costmaps_bool == True or self.plot_segments_bool == True:
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
        # or /move_base/TebLocalPlannerROS/local_plan

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)
              
    # odom callback
    def odom_callback(self, msg):
        #print('\nodom_callback')
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.robot_position_odom = msg.pose.pose.position
        self.robot_orientation_odom = msg.pose.pose.orientation
        
    # global plan callback
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback')
        
        self.global_plan = []

        for i in range(0,len(msg.poses)):
            self.global_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

    # local plan callback
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        try:
            self.local_plan = []
            self.local_plan_xs = [] 
            self.local_plan_ys = [] 

            for i in range(0,len(msg.poses)):
                self.local_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.local_costmap_origin_x) / self.local_costmap_resolution)
                y_temp = int((msg.poses[i].pose.position.y - self.local_costmap_origin_y) / self.local_costmap_resolution)
                if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                    self.local_plan_xs.append(x_temp)
                    self.local_plan_ys.append(self.local_costmap_size - y_temp)

        except Exception as e:
            print('exception = ', e)
            return
    
    # footprint callback
    def footprint_callback(self, msg):
        #print('\nfootprint_callback') 
        self.footprint = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
    
    # amcl pose callback
    def amcl_callback(self, msg):
        #print('\namcl_callback')
        self.amcl_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.robot_position_map = msg.pose.pose.position
        self.robot_orientation_map = msg.pose.pose.orientation

    # local costmap callback
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')

        self.global_plan_copy = copy.deepcopy(self.global_plan)
        self.local_plan_xs_copy = copy.deepcopy(self.local_plan_xs)
        self.local_plan_ys_copy = copy.deepcopy(self.local_plan_ys)
        self.local_plan_copy = copy.deepcopy(self.local_plan)
        self.odom_copy = copy.deepcopy(self.odom)
        self.amcl_pose_copy = copy.deepcopy(self.amcl_pose)
        self.footprint_copy = copy.deepcopy(self.footprint)
        self.robot_position_map_copy = copy.deepcopy(self.robot_position_map)
        self.robot_orientation_map_copy = copy.deepcopy(self.robot_orientation_map)
        
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())

            self.tf_map_odom = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]

            # convert robot positions from /odom to /map
            #t_o_m = np.asarray([transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z])
            r_o_m = R.from_quat([transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w])
            r_o_m = np.asarray(r_o_m.as_matrix())
            r_o = R.from_quat([self.robot_orientation_odom.x,self.robot_orientation_odom.y,self.robot_orientation_odom.z,self.robot_orientation_odom.w])
            r_o = np.asarray(r_o.as_matrix())

            # position transformation
            #p = np.array([self.robot_position_odom.x, self.robot_position_odom.y, self.robot_position_odom.z])
            #pnew = p.dot(r_o_m) + t_o_m
            #self.robot_position_map.x = pnew[0]
            #self.robot_position_map.y = pnew[1]
            #self.robot_position_map.z = pnew[2]
            # orientation transformation
            #r_m = r_o * r_o_m
            #self.robot_orientation_map = rotationMatrixToQuaternion(r_m) #tr.quaternion_from_matrix(r_m)
                
            # save costmap data
            self.local_costmap_origin_x = msg.info.origin.position.x
            self.local_costmap_origin_y = msg.info.origin.position.y
            self.local_costmap_resolution = msg.info.resolution
            self.local_costmap_info = [self.local_costmap_resolution, msg.info.width, msg.info.height, self.local_costmap_origin_x, self.local_costmap_origin_y, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create np.array image object
            self.local_costmap = np.asarray(msg.data)
            self.local_costmap.resize((msg.info.height,msg.info.width))

            if self.plot_costmaps_bool == True:
                self.plot_costmaps()
                
            # Turn inflated area to free space and 100s to 99s
            self.local_costmap[self.local_costmap == 100] = 99
            self.local_costmap[self.local_costmap <= 98] = 0

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.local_costmap = self.local_costmap * 1.0

            # my change - to return grayscale to classifier_fn
            self.fudged_image = self.local_costmap.copy()
            self.fudged_image[:] = 0 #hide_color = 0

            # find lc coordinates of robot's odometry coordinates 
            self.x_odom_index = round((self.odom_x - self.local_costmap_origin_x) / self.local_costmap_resolution)
            self.y_odom_index = round((self.odom_y - self.local_costmap_origin_y) / self.local_costmap_resolution)

            # find segments
            self.find_segments()

            # create and save encoded perturbations
            self.create_data()

            # to explain from another callback function, track copies (faster changing vars), and current callback vars
            start = time.time()
            self.lime_rt_pub.explain(self.global_plan_copy, self.local_plan_xs_copy, self.local_plan_ys_copy, self.local_plan_copy, 
            self.odom_copy, self.amcl_pose_copy, self.footprint_copy, self.robot_position_map_copy, self.robot_orientation_map_copy, self.local_costmap_info, 
            self.segments, self.data, self.tf_map_odom, self.tf_odom_map, self.local_costmap, self.fudged_image)
            end = time.time()
            print('explanation time = ', round(end-start,3))

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
        
        local_costmap_99s_100s = copy.deepcopy(self.local_costmap)
        local_costmap_99s_100s[local_costmap_99s_100s < 99] = 0        
        #self.fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        local_costmap_99s_100s = np.flip(local_costmap_99s_100s, 0)
        self.ax.imshow(local_costmap_99s_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s_100s.png', transparent=False)
        #self.fig.clf()
        
        local_costmap_original = copy.deepcopy(self.local_costmap)
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_costmap_original = np.flip(local_costmap_original, 0)
        self.ax.imshow(local_costmap_original.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_original.png', transparent=False)
        #self.fig.clf()
        
        local_costmap_100s = copy.deepcopy(self.local_costmap)
        local_costmap_100s[local_costmap_100s != 100] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_costmap_100s = np.flip(local_costmap_100s, 0)
        self.ax.imshow(local_costmap_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_100s.png', transparent=False)
        #self.fig.clf()
        
        local_costmap_99s = copy.deepcopy(self.local_costmap)
        local_costmap_99s[local_costmap_99s != 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_costmap_99s = np.flip(local_costmap_99s, 0)
        self.ax.imshow(local_costmap_99s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s.png', transparent=False)
        #self.fig.clf()
        
        local_costmap_less_than_99 = copy.deepcopy(self.local_costmap)
        local_costmap_less_than_99[local_costmap_less_than_99 >= 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_costmap_less_than_99 = np.flip(local_costmap_less_than_99, 0)
        self.ax.imshow(local_costmap_less_than_99.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_less_than_99.png', transparent=False)
        self.fig.clf()
        
        end = time.time()
        print('costmaps plot runtime = ', round(end-start, 3))

    # find segments -- do segmentation
    def find_segments(self):
        ###### SEGMENTATION PART ######
        start = time.time()

        # find image_rgb
        image_rgb = gray2rgb(self.local_costmap)

        segment_start = time.time()

        # Find segments_slic
        segments_slic = slic(image_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=False, enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False)#, start_label=1, mask=None)
                        #slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False) #0.14.2

        segment_end = time.time()        
        print('SLIC runtime = ', round(segment_end - segment_start,3))
        
        self.segments = np.zeros(self.local_costmap.shape, np.uint8)
        
        # make one free space segment
        ctr = 0
        self.segments[:, :] = ctr
        ctr = ctr + 1

        # add obstacle segments
        num_of_obstacles = 0        
        for i in np.unique(segments_slic):
            temp = self.local_costmap[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp != 0) #np.count_nonzero(temp == 99)
            if count_of_99_s > 0.95 * temp.shape[0]: #or np.all(image[segments_slic == i] == 99) 
                self.segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1
        #print('num_of_obstacles: ', num_of_obstacles)

        end = time.time()
        print('segmentation runtime = ', (round(end-start,3)))

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(self.segments.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        self.centroids_segments = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            self.centroids_segments.append([v,cx,cy])

        # plot segments
        if self.plot_segments_bool == True:
            self.plot_segments()

        return self.segments

    # plot segments
    def plot_segments(self):
        start = time.time()

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

        self.fig.savefig(dirCurr + '/' + 'segments_without_labels.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/segments_without_labels.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.segments, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        if len(self.centroids_segments) > 0:
            for i in range(0, len(self.centroids_segments)):
                self.ax.scatter(self.centroids_segments[i][1], self.local_costmap_size - self.centroids_segments[i][2], c='white', marker='o')   
                self.ax.text(self.centroids_segments[i][1], self.local_costmap_size - self.centroids_segments[i][2], self.centroids_segments[i][0], c='white')
            pd.DataFrame(self.centroids_segments).to_csv(dirCurr + '/centroids_segments.csv', index=False)#, header=False)

        self.fig.savefig(dirCurr + '/' + 'segments_with_labels.png', transparent=False)
        self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/segments_with_labels.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        img = np.flip(self.local_costmap, axis=0)
        self.ax.imshow(img.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'local_costmap.png', transparent=False)
        self.fig.clf()

        end = time.time()
        print('segments plot runtime = ' + str(round(end-start,3)))

    # create encoded perturbations
    def create_data(self):
        # create N+1 perturbations for N segments--superpixels
        self.n_features = np.unique(self.segments).shape[0]
        self.n_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.n_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.n_samples, self.n_features))


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
    def __init__(self, image, segments):
        self.local_costmap = image
        self.segments = segments
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

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        exp = self.local_exp[label]

        temp = np.zeros(self.local_costmap.shape)

        w_sum_abs = 0.0
        w_s_abs = []
        for f, w in exp:
            w_sum_abs += abs(w)
            w_s_abs.append(abs(w))
        max_w_abs = max(w_s_abs)
        if max_w_abs == 0:
            self.all_weights_zero = True
            temp[self.local_costmap == 0] = self.free_space_shade
            temp[self.local_costmap != 0] = 0.0
            return temp, exp

        segments_labels = np.unique(self.segments)
        
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
                    temp[self.segments == f, 0] = self.free_space_shade
                    temp[self.segments == f, 1] = self.free_space_shade
                    temp[self.segments == f, 2] = self.free_space_shade
            # obstacle
            else:
                if self.color_free_space == False:
                    if c == 1:
                        temp[self.segments == f, 0] = 0.0
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                        else:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs 
                        temp[self.segments == f, 2] = 0.0
                    elif c == 0:
                        temp[self.segments == f, 0] = 0.0
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                        else:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs 
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                                        
        return temp, exp

class lime_rt_pub(object):
    # Constructor
    def __init__(self):
        # global counter
        self.counter_global = 0

        # publishers
        self.publish_explanation_coeffs = False  
        self.publish_explanation_image = True

        # plotting
        self.plot_explanation = False
        self.plot_perturbations = False 
        self.plot_classification = False

        # directories        
        self.dir_curr = os.getcwd()

        self.dir_main = 'explanation_data'
        try:
            os.mkdir(self.dir_main)
        except FileExistsError:
            pass
        
        self.dirData = self.dir_main + '/lime_rt_data'
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_explanation == True:
            self.explanation_dir = self.dir_main + '/explanation_images'
            try:
                os.mkdir(self.explanation_dir)
            except FileExistsError:
                pass

        if self.plot_perturbations == True: 
            self.perturbation_dir = self.dir_main + '/perturbation_images'
            try:
                os.mkdir(self.perturbation_dir)
            except FileExistsError:
                pass
        
        if self.plot_classification == True:
            self.classifier_dir = self.dir_main + '/classifier_images'
            try:
                os.mkdir(self.classifier_dir)
            except FileExistsError:
                pass

        # local costmap variables
        self.labels = np.array([]) 
        self.distances = np.array([])
        self.local_costmap_size = 160
   
        # deviation
        self.original_deviation = 0

        # samples variables
        self.n_samples = 0
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
            self.footprint_pd = pd.DataFrame(self.footprint)
            self.footprint_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_original_pd = pd.DataFrame(self.local_plan_original)
            self.local_plan_original_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp_pd = pd.DataFrame(self.plan_tmp)
            self.plan_tmp_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_pd = pd.DataFrame(self.global_plan)
            self.global_plan_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)

            # Save costmap_info instance to file
            self.local_costmap_info_pd = pd.DataFrame(self.local_costmap_info).transpose()
            self.local_costmap_info_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_pd = pd.DataFrame(self.amcl_pose).transpose()
            self.amcl_pose_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_pd = pd.DataFrame(self.tf_odom_map).transpose()
            self.tf_odom_map_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_pd = pd.DataFrame(self.tf_map_odom).transpose()
            self.tf_map_odom_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

            # Save odometry instance to file
            self.odom_pd = pd.DataFrame(self.odom).transpose()
            self.odom_pd.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        except Exception as e:
            print('exception = ', e)
            return False

        return True

    # saving important data to class variables
    def saveImportantData2ClassVars(self):
        #print(self.local_plan_original.shape)
        #print(self.local_plan_original)
        
        #print('\nself.local_costmap_info = ', self.local_costmap_info)
        #print('\nself.odom = ', self.odom)
        #print('\nself.global_plan = ', self.global_plan)

        try:
            self.t_om = np.asarray([self.tf_odom_map[0],self.tf_odom_map[1],self.tf_odom_map[2]])
            self.r_om = R.from_quat([self.tf_odom_map[3],self.tf_odom_map[4],self.tf_odom_map[5],self.tf_odom_map[6]])
            self.r_om = np.asarray(self.r_om.as_matrix())

            self.t_mo = np.asarray([self.tf_map_odom[0],self.tf_map_odom[1],self.tf_map_odom[2]])
            self.r_mo = R.from_quat([self.tf_map_odom[3],self.tf_map_odom[4],self.tf_map_odom[5],self.tf_map_odom[6]])
            self.r_mo = np.asarray(self.r_mo.as_matrix())    

            # save costmap info to class variables
            self.local_costmap_origin_x = self.local_costmap_info[3]
            #print('self.local_costmap_origin_x: ', self.local_costmap_origin_x)
            self.local_costmap_origin_y = self.local_costmap_info[4]
            #print('self.local_costmap_origin_y: ', self.local_costmap_origin_y)
            self.local_costmap_resolution = self.local_costmap_info[0]
            #print('self.local_costmap_resolution: ', self.local_costmap_resolution)
            self.localCostmapHeight = self.local_costmap_info[2]
            #print('self.localCostmapHeight: ', self.localCostmapHeight)
            self.localCostmapWidth = self.local_costmap_info[1]
            #print('self.localCostmapWidth: ', self.localCostmapWidth)

            # save robot odometry location to class variables
            self.odom_x = self.odom[0]
            # print('self.odom_x: ', self.odom_x)
            self.odom_y = self.odom[1]
            # print('self.odom_y: ', self.odom_y)

            # save indices of robot's odometry location in local costmap to class variables
            self.localCostmapIndex_x_odom = int((self.odom_x - self.local_costmap_origin_x) / self.local_costmap_resolution)
            # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
            self.localCostmapIndex_y_odom = int((self.odom_y - self.local_costmap_origin_y) / self.local_costmap_resolution)
            # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

            # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
            self.x_odom_index = [self.localCostmapIndex_x_odom]
            # print('self.x_odom_index: ', self.x_odom_index)
            self.y_odom_index = [self.localCostmapIndex_y_odom]
            # print('self.y_odom_index: ', self.y_odom_index)

            #'''
            # save robot odometry orientation to class variables
            self.odom_z = self.odom[2]
            self.odom_w = self.odom[2]
            # calculate Euler angles based on orientation quaternion
            [self.yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
            
            # find yaw angles projections on x and y axes and save them to class variables
            self.yaw_odom_x = math.cos(self.yaw_odom)
            self.yaw_odom_y = math.sin(self.yaw_odom)
            #'''

            self.transformed_plan_xs = []
            self.transformed_plan_ys = []
            for i in range(0, len(self.global_plan)):
                x_temp = int((self.global_plan[i][0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
                y_temp = int((self.global_plan[i][1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

                if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
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

            # call teb and get labels
            self.labels = []
            imgs = []
            rows = self.data

            batch_size = 2048
            
            segments_labels = np.unique(self.segments)

            ctr = 0
            for row in rows:
                temp = copy.deepcopy(self.local_costmap)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(self.segments.shape).astype(bool)
                for z in zeros:
                    mask[self.segments == segments_labels[z]] = True
                temp[mask] = self.fudged_image[mask]
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
                    fig.savefig(dirCurr + '/perturbation_' + str(ctr) + '.png', transparent=False)
                    fig.clf()
                    ctr += 1

                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    self.labels.extend(preds)
                    imgs = []

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
            x_temp = int((transformed_plan[i, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
            y_temp = int((transformed_plan[i, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

            if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)

        for ctr in range(0, sample_size):
            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []

            # find if there is local plan
            local_plans_local = local_plans.loc[local_plans['ID'] == ctr]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
                    y_temp = int((local_plans_local.iloc[j, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

                    if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
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
            np.savetxt(self.dir_curr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt(self.dir_curr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 2:
            np.savetxt(self.dir_curr + 'src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")
        end = time.time()
        print('classifier_fn: data preparation runtime = ', round(end-start,3))

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
        print('classifier_fn: local planner time = ', round(end-start,3))

        #print('\nC++ node ended')

       
        start = time.time()
        
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('local_plans = ', local_plans)
        # save original local plan for qsr_rt
        #self.local_plan_full_perturbation = local_plans.loc[local_plans['ID'] == 0]
        #print('self.local_plan_full_perturbation.shape = ', self.local_plan_full_perturbation.shape)
        #local_plan_full_perturbation.to_csv(self.dir_curr + '/' + self.dirData + '/local_plan_full_perturbation.csv', index=False)#, header=False)

        # load transformed global plan to /odom frame
        transformed_plan = np.array(pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        # save transformed plan for the qsr_rt
        #transformed_plan.to_csv(self.dir_curr + '/' + self.dirData + '/transformed_plan.csv', index=False)#, header=False)
       
        end = time.time()
        print('classifier_fn: load results runtime = ', round(end-start,2))


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
        print('classifier_fn: target calculation runtime = ', round(end-start,3))
        
        self.original_deviation = local_plan_deviation.iloc[0, 0]
        #print('\noriginal_deviation = ', self.original_deviation)

        cmd_vel_perturb['deviate'] = local_plan_deviation
        #return local_plan_deviation
        return np.array(cmd_vel_perturb.iloc[:, 3:])

    # explain function
    def explain(self, global_plan_tmp_copy, local_plan_x_list_copy, local_plan_y_list_copy, local_plan_tmp_copy, odom_tmp_copy, 
    amcl_pose_tmp_copy, footprint_tmp_copy, robot_position_map_copy, robot_orientation_map_copy, costmap_info_tmp, segments, data,
    tf_map_odom_tmp, tf_odom_map_tmp, image, fudged_image):
    
        explain_time_start = time.time()

        self.local_plan_xs_fixed = local_plan_x_list_copy
        self.local_plan_ys_fixed = local_plan_y_list_copy

        self.robot_position_map = robot_position_map_copy
        self.robot_orientation_map = robot_orientation_map_copy

        self.global_plan = global_plan_tmp_copy
        self.plan_tmp = global_plan_tmp_copy
        self.footprint = footprint_tmp_copy
        self.local_plan_original = local_plan_tmp_copy
        self.amcl_pose = amcl_pose_tmp_copy
        self.odom = odom_tmp_copy
        self.local_costmap_info = costmap_info_tmp
        self.segments = segments
        self.data = data
        self.tf_map_odom = tf_map_odom_tmp
        self.tf_odom_map = tf_odom_map_tmp
        self.local_costmap = image
        self.fudged_image = fudged_image
        
        # turn grayscale image to rgb image
        self.local_costmap_rgb = gray2rgb(self.local_costmap * 1.0)
        
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
        print('label creation runtime = ', round(end-start,2))
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

        '''
        # Explanation variables
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'
                
        try:
            start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.local_costmap_rgb, self.segments)
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
            output, exp = ret_exp.get_image_and_mask(label=0)
            end = time.time()
            print('\nGET EXP PIC TIME = ', end-start)
            print('exp: ', exp)

            centroids_for_plot = []
            lc_regions = regionprops(self.segments.astype(int))
            for lc_region in lc_regions:
                v = lc_region.label
                cy, cx = lc_region.centroid
                centroids_for_plot.append([v,cx,cy])

            
            #print('\nexp = ', exp)

            #pd.DataFrame(output[:,:,0]).to_csv(self.dir_curr + '/' + self.dirData + '/output_B.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,1]).to_csv(self.dir_curr + '/' + self.dirData + '/output_G.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,2]).to_csv(self.dir_curr + '/' + self.dirData + '/output_R.csv', index=False) #, header=False)


            if self.plot_explanation == True:
                dirCurr = self.explanation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass

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

                ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='x')
                ax.scatter(self.local_plan_xs_fixed, self.local_plan_ys_fixed, c='yellow', marker='o')
                ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
                ax.text(self.x_odom_index[0], self.y_odom_index[0], 'robot', c='white')
                ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')
                ax.imshow(output.astype('float64'), aspect='auto')
                for i in range(0, len(centroids_for_plot)):
                    ax.scatter(centroids_for_plot[i][1], self.local_costmap_size - centroids_for_plot[i][2], c='white', marker='o')   
                    ax.text(centroids_for_plot[i][1], self.local_costmap_size - centroids_for_plot[i][2], centroids_for_plot[i][0], c='white')

                fig.savefig(dirCurr + '/explanation_' + '.png', transparent=False)
                fig.clf()

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
                ax.imshow(output.astype('float64'), aspect='auto')
                fig.savefig('explanation.png', transparent=False)
                fig.clf()

                exp_img_start = time.time()
                # publish explanation image
                #output = output[:, :, [2, 1, 0]].astype(np.uint8)
                #output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                output = output[:, :, [2, 1, 0]]#.astype(np.uint8)
                output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                self.pub_exp_image.publish(output_cv)

                exp_img_end = time.time()
                print('\nexp_img_time = ', exp_img_end - exp_img_start)
            
            explain_time_end = time.time()
            with open(self.dir_main + '/explanation_time.csv','a') as file:
                file.write(str(explain_time_end-explain_time_start))
                file.write('\n')
            
            self.counter_global+=1
            
        except Exception as e:
            print('Exception: ', e)
            #print('Exception - explanation is skipped!!!')
            return
        '''
        


# ----------main-----------
# main function
# Initialize the ROS Node named 'lime_rt', allow multiple nodes to be run with this name
rospy.init_node('lime_rt', anonymous=True)

# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()