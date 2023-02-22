#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from skimage.segmentation import slic
from skimage.color import gray2rgb
import tf2_ros
import os
import time, copy
from skimage.measure import regionprops

# data received from each subscriber is saved to .csv files
class lime_rt_sub(object):
    # constructor
    def __init__(self):
        # plot segments and costmaps
        self.plot_segments = True
        self.plot_costmaps = True
        # global counter for plotting
        self.counter_global = 0
    
        # directories
        self.dir_curr = os.getcwd()
        
        self.dir_main = 'explanation_data'
        try:
            os.mkdir(self.dir_main)
        except FileExistsError:
            pass

        self.dir_data = self.dir_main + '/lime_rt_data'
        try:
            os.mkdir(self.dir_data)
        except FileExistsError:
            pass

        if self.plot_segments == True:
            self.dir_segmentation = self.dir_main + '/segmentation_images'
            try:
                os.mkdir(self.dir_segmentation)
            except FileExistsError:
                pass

        if self.plot_costmaps == True:
            self.dir_costmap = self.dir_main + '/costmap_images'
            try:
                os.mkdir(self.dir_costmap)
            except FileExistsError:
                pass

        # plans' variables
        #self.global_plan_xs = []
        #self.global_plan_ys = []
        self.global_plan = []
        self.local_plan_xs = [] 
        self.local_plan_ys = []
        self.local_plan = [] 

        # tf variables
        self.tf_odom_map = [] 
        self.tf_map_odom = [] 
        
        # footprint
        self.footprint = [] 
         
        # pose variables
        self.amcl_pose = [] 
        self.odom = []
        self.odom_x = 0
        self.odom_y = 0
  
        # costmap variables
        self.costmap_info = [] 
        self.costmap_size = 160
        self.local_costmap_origin_x = 0 
        self.local_costmap_origin_y = 0 
        self.local_costmap_resolution = 0
        
        # segmentation variables
        self.segments = np.array([])
        self.local_costmap = np.array([])

        # perturbation variables
        self.data = np.array([]) 
        self.n_samples = 0
        self.n_features = 0

    # define subscribers
    def main_(self):
        if self.plot_costmaps == True or self.plot_segments == True:
            self.fig = plt.figure(frameon=False)
            self.w = 1.6 * 3
            self.h = 1.6 * 3
            self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

        # local plan subscriber
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)
        
        # global plan subscriber
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        # The synonymous topic is "/move_base/TebLocalPlannerROS/global_plan"
        
        # footprint subscriber
        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        # odometry subscriber
        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        # amcl subscriber
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        # local costmap subscriber
        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

    # plot segments
    def plot_segments(self):
        start = time.time()

        dirCurr = self.dir_segmentation + '/' + str(self.counter_global)            
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
        segs = np.flip(self.segments, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        for i in range(0, len(centroids_segments)):
            self.ax.scatter(centroids_segments[i][1], self.costmap_size - centroids_segments[i][2], c='white', marker='o')   
            self.ax.text(centroids_segments[i][1], self.costmap_size - centroids_segments[i][2], centroids_segments[i][0], c='white')

        self.fig.savefig(dirCurr + '/' + 'segments_with_labels_' + str(self.counter_global) + '.png', transparent=False)
        self.fig.clf()
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        img = np.flip(self.local_costmap, axis=0)
        self.ax.imshow(img.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'local_costmap_' + str(self.counter_global) + '.png', transparent=False)
        self.fig.clf()

        end = time.time()
        print('SEGMENTS PLOTTING TIME = ' + str(end-start) + ' seconds!!!')

    # local costmap segmentation based on slic algorithm
    def find_segments(self):
        segment_start = time.time()

        # find local_costmap_rgb
        local_costmap_rgb = gray2rgb(self.local_costmap)

        # Find segments_slic
        segments_slic = slic(local_costmap_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=False, enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False)#, start_label=1, mask=None)
                        #slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False) #0.14.2

        segment_end = time.time()        
        print('SLIC runtime = ', segment_end - segment_start)
        
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

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(self.segments.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_segments = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_segments.append([v,cx,cy])
        
        # plot segments
        if self.plot_segments == True:
            self.plot_segments()

        return self.segments

    # create data--perturbations based on segments
    def create_data(self):
        # create N+1 perturbations for N segments--superpixels
        self.n_features = np.unique(self.segments).shape[0]
        self.n_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.n_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.n_samples, self.n_features))

    # plot costmaps
    def plot_costmaps(self):
        start = time.time()

        dirCurr = self.dir_costmap + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass
        
        self.image_99s_100s = copy.deepcopy(self.local_costmap)
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
        
        self.image_original = copy.deepcopy(self.local_costmap)
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
        
        self.image_100s = copy.deepcopy(self.local_costmap)
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
        
        self.image_99s = copy.deepcopy(self.local_costmap)
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
        
        self.image_less_than_99 = copy.deepcopy(self.local_costmap)
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

    # local costmap callback
    def local_costmap_callback(self, msg):
        # if you can get tf proceed
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
            self.tf_odom_map = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
            self.tf_map_odom = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
         
            # save tf 
            pd.DataFrame(self.tf_odom_map).to_csv(self.dir_curr + '/' + self.dir_data + '/tf_odom_map_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom).to_csv(self.dir_curr + '/' + self.dir_data + '/tf_map_odom_tmp.csv', index=False)#, header=False)
            
            # save costmap in a right image format
            self.local_costmap_origin_x = msg.info.origin.position.x
            self.local_costmap_origin_y = msg.info.origin.position.y
            self.local_costmap_resolution = msg.info.resolution
            self.costmap_info = [self.local_costmap_resolution, msg.info.width, msg.info.height, self.local_costmap_origin_x, self.local_costmap_origin_y, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create image object
            self.local_costmap = np.asarray(msg.data)
            self.local_costmap.resize((msg.info.height,msg.info.width))

            if self.plot_costmaps == True:
                self.plot_costmaps()
            
            # Turn non-lethal inflated area (< 99) to free space and 100s to 99s
            # POSSIBLE PLACE TO AFFECT SEGMENTATION
            self.local_costmap[self.local_costmap == 100] = 99
            self.local_costmap[self.local_costmap <= 98] = 0

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.local_costmap = self.local_costmap * 1.0

            # my change - to return grayscale to classifier_fn
            self.fudged_image = self.local_costmap.copy()
            self.fudged_image[:] = 0 #hide_color = 0

            # save indices of robot's odometry location in local costmap to class variables
            self.x_odom_index = round((self.odom_x - self.local_costmap_origin_x) / self.local_costmap_resolution)
            self.y_odom_index = round((self.odom_y - self.local_costmap_origin_y) / self.local_costmap_resolution)

            # find segments
            self.segments = self.find_segments(self.local_costmap)

            # create data -- perturbations
            self.create_data()

            # save data to the .csv files
            pd.DataFrame(self.costmap_info).to_csv(self.dir_curr + '/' + self.dir_data + '/costmap_info_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.local_costmap).to_csv(self.dir_curr + '/' + self.dir_data + '/image.csv', index=False) #, header=False)
            pd.DataFrame(self.fudged_image).to_csv(self.dir_curr + '/' + self.dir_data + '/fudged_image.csv', index=False)#, header=False)
            pd.DataFrame(self.segments).to_csv(self.dir_curr + '/' + self.dir_data + '/segments.csv', index=False)#, header=False)
            pd.DataFrame(self.data).to_csv(self.dir_curr + '/' + self.dir_data + '/data.csv', index=False)#, header=False)

            # increase the global counter
            self.counter_global += 1
        
        except Exception as e:
            print('exception = ', e)
            return
        
    # global plan callback
    def global_plan_callback(self, msg):
        #self.global_plan_xs = [] 
        #self.global_plan_ys = []
        self.global_plan = []
        
        for i in range(0,len(msg.poses)):
            #self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            #self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        pd.DataFrame(self.global_plan).to_csv(self.dir_curr + '/' + self.dir_data + '/global_plan.csv', index=False)#, header=False)
        
    # local plan callback
    def local_plan_callback(self, msg):
        try:        
            self.local_plan_xs = [] 
            self.local_plan_ys = [] 
            self.local_plan = []
            
            # transform local plan coordinates to pixel positions in the local costmap
            for i in range(0,len(msg.poses)):
                # 5 is a random ID number
                self.local_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.local_costmap_origin_x) / self.local_costmap_resolution)
                y_temp = int((msg.poses[i].pose.position.y - self.local_costmap_origin_y) / self.local_costmap_resolution)
                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.local_plan_xs.append(x_temp)
                    self.local_plan_ys.append(y_temp)

            # save data to the .csv files
            pd.DataFrame(self.local_plan_xs).to_csv(self.dir_curr + '/' + self.dir_data + '/local_plan_xs.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_ys).to_csv(self.dir_curr + '/' + self.dir_data + '/local_plan_ys.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan).to_csv(self.dir_curr + '/' + self.dir_data + '/local_plan.csv', index=False)#, header=False)
            pd.DataFrame(self.odom).to_csv(self.dir_curr + '/' + self.dir_data + '/odom.csv', index=False)#, header=False)
        
        except:
            pass

    # footprint callback
    def footprint_callback(self, msg):
        self.footprint = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        pd.DataFrame(self.footprint).to_csv(self.dir_curr + '/' + self.dir_data + '/footprint.csv', index=False)#, header=False)

    # odometry callback
    # odometry is received very often, and it is saved (to a variable) in the "local_plan_callback' function to save computing resources
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    # amcl callback
    def amcl_callback(self, msg):
        self.amcl_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pd.DataFrame(self.amcl_pose).to_csv(self.dir_curr + '/' + self.dir_data + '/amcl_pose.csv', index=False)#, header=False)

    
# ----------main-----------
# main function
# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Initialize the ROS Node named 'lime_rt_sub', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_sub', anonymous=True)

# declare tf2 transformation buffer
lime_rt_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_obj.tf_listener = tf2_ros.TransformListener(lime_rt_obj.tfBuffer)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()