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

#import skimage
#print(skimage.__version__) #0.14.2

# data received from each subscriber is saved to .csv file
class lime_rt_sub(object):
    # Constructor
    def __init__(self):
        # global counter for plotting
        self.counter_global = 0

        # segments
        self.plot_segments = True
        self.plot_costmaps = True
        
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

        # plans' variables
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.local_plan_tmp = [] 
        self.global_plan_tmp = []
        self.global_plan_empty = True
        self.local_plan_empty = True

        # tf variables
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 
        
        # footprint
        self.footprint_tmp = [] 
         
        # pose variables
        self.amcl_pose_tmp = [] 
        self.odom_tmp = []
        self.odom_x = 0
        self.odom_y = 0
  
        # costmap variables
        self.costmap_info_tmp = [] 
        self.image_rgb = np.array([]) 
        self.segments = np.array([])
        self.data = np.array([]) 
        self.image = np.array([])
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.costmap_size = 160
        self.local_costmap_empty = True

        # samples' variables
        self.num_samples = 0
        self.n_features = 0

        # deviation
        self.original_deviation = 0

    # LC segmentation algorithm based on slic
    def find_segments(self, image):
        segment_start = time.time()

        # find image_rgb
        image_rgb = gray2rgb(self.image)

        # Find segments_slic
        segments_slic = slic(image_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=False, enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False)#, start_label=1, mask=None)
                        #slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False) #0.14.2

        segment_end = time.time()        
        print('SLIC runtime = ', segment_end - segment_start)
        
        self.segments = np.zeros(image.shape, np.uint8)
        # make one free space segment
        ctr = 0
        self.segments[:, :] = ctr
        ctr = ctr + 1

        # add obstacle segments
        num_of_obstacles = 0        
        for i in np.unique(segments_slic):
            temp = image[segments_slic == i]
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
            img = np.flip(self.image, axis=0)
            self.ax.imshow(img.astype('float64'), aspect='auto')

            self.fig.savefig(dirCurr + '/' + 'local_costmap_' + str(self.counter_global) + '.png', transparent=False)
            self.fig.clf()

            end = time.time()
            print('SEGMENTS PLOTTING TIME = ' + str(end-start) + ' seconds!!!')

        return self.segments

    # Create data--perturbations based on segments
    def create_data(self):
        # create N+1 perturbations for N segments--superpixels
        self.n_features = np.unique(self.segments).shape[0]
        self.num_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.num_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.num_samples, self.n_features))

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        # if you can get tf proceed
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
         
            # save tf 
            pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom_tmp.csv', index=False)#, header=False)
            
            # save costmap in a right image format
            self.localCostmapOriginX = msg.info.origin.position.x
            self.localCostmapOriginY = msg.info.origin.position.y
            self.localCostmapResolution = msg.info.resolution
            self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create image object
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
            
            # Turn non-lethal inflated area (< 99) to free space and 100s to 99s
            # POSSIBLE PLACE TO AFFECT SEGMENTATION
            self.image[self.image == 100] = 99
            self.image[self.image <= 98] = 0

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0

            # my change - to return grayscale to classifier_fn
            self.fudged_image = self.image.copy()
            self.fudged_image[:] = 0 #hide_color = 0

            # save indices of robot's odometry location in local costmap to class variables
            self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

            # find segments
            self.segments = self.find_segments(self.image)

            # create data -- perturbations
            self.create_data()

            self.local_costmap_empty = False

            # save data to the .csv files
            pd.DataFrame(self.costmap_info_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/costmap_info_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.image).to_csv(self.dirCurr + '/' + self.dirData + '/image.csv', index=False) #, header=False)
            pd.DataFrame(self.fudged_image).to_csv(self.dirCurr + '/' + self.dirData + '/fudged_image.csv', index=False)#, header=False)
            pd.DataFrame(self.segments).to_csv(self.dirCurr + '/' + self.dirData + '/segments.csv', index=False)#, header=False)
            pd.DataFrame(self.data).to_csv(self.dirCurr + '/' + self.dirData + '/data.csv', index=False)#, header=False)

            # increase the global counter
            self.counter_global += 1
        
        except Exception as e:
            print('exception = ', e)
            return
        
    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        
        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        self.global_plan_empty = False

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv', index=False)#, header=False)
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        try:        
            self.local_plan_x_list = [] 
            self.local_plan_y_list = [] 
            self.local_plan_tmp = []
            # transform local plan coordinates to pixel positions in the local costmap
            for i in range(0,len(msg.poses)):
                # 5 is probably a random ID number
                self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.local_plan_x_list.append(x_temp)
                    self.local_plan_y_list.append(y_temp)

            self.local_plan_empty = False

            # save data to the .csv files
            pd.DataFrame(self.local_plan_x_list).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_x_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_y_list).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_y_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.odom_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/odom_tmp.csv', index=False)#, header=False)
        
        except:
            pass

    # Define a callback for the footprint
    def footprint_callback(self, msg):
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        pd.DataFrame(self.footprint_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/footprint_tmp.csv', index=False)#, header=False)

    # Define a callback for the odometry
    # Because odometry is received very often it is saved (to a variable) in the "local_plan_callback' function to save computing resources
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pd.DataFrame(self.amcl_pose_tmp).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose_tmp.csv', index=False)#, header=False)

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

        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        # The synonym topic is "/move_base/TebLocalPlannerROS/global_plan"

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)


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