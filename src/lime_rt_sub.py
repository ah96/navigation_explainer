#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
#import time
import pandas as pd
from skimage.segmentation import slic
from skimage.color import gray2rgb
#from scipy.spatial.transform import Rotation as R
#import copy
import tf2_ros
#import math
#from skimage.measure import regionprops
import os


class lime_rt_sub(object):
    # Constructor
    def __init__(self):
        # plans' variables
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.local_plan_tmp = [] 
        self.plan_tmp = [] 
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

        # segments
        self.plot_segments = True

        # samples' variables
        self.num_samples = 0
        self.n_features = 0

        # deviation
        self.original_deviation = 0
        
        # directory to save data
        self.dirCurr = os.getcwd()
        self.dirName = 'lime_rt_data'
        try:
            os.mkdir(self.dirName)
        except FileExistsError:
            pass

    # LC segmentation algorithm
    def segment_local_costmap(self, image):
        #print('segmentation algorithm')

        # find image_rgb
        image_rgb = gray2rgb(self.image)
        #image = np.stack(3 * (image,), axis=-1)

        # Find segments_slic
        segments_slic = slic(image_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                                multichannel=True, convert2lab=True,
                                enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                                start_label=1, mask=None)

        self.segments = np.zeros(image.shape, np.uint8)

        # make one free space segment
        ctr = 0
        self.segments[:, :] = ctr
        ctr = ctr + 1

        # add obstacle segments
        num_of_obstacles = 0        
        for i in np.unique(segments_slic):
            temp = image[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            #print('count: ', count)
            #print('temp: ', temp)
            #print('len(temp): ', temp.shape[0])
            if np.all(image[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                #print('obstacle')
                self.segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1
        #print('num_of_obstacles: ', num_of_obstacles)        


        if self.plot_segments:
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments_slic.astype('float64'), aspect='auto')
            fig.savefig('segments_slic.png', transparent=False)
            fig.clf()

            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')
            fig.savefig('segments_final.png', transparent=False)
            fig.clf()

        return self.segments

    # Create data--perturbations based on segments
    def create_data(self):
        # create data -- perturbations -- N+1 perturbations
        self.n_features = np.unique(self.segments).shape[0]
        self.num_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.num_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.num_samples, self.n_features))

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        #print('\nlocal_costmap_callback')

        # if you can get tf proceed
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            #t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
            #r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
            #r_ = np.asarray(r.as_matrix())

            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())

            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
        except:
            pass    

        # save tf 
        pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_odom_map_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_map_odom_tmp.csv', index=False)#, header=False)
        
        # save costmap in a right image format
        self.localCostmapOriginX = msg.info.origin.position.x
        self.localCostmapOriginY = msg.info.origin.position.y
        self.localCostmapResolution = msg.info.resolution
        self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

        # create image object
        self.image = np.asarray(msg.data)
        self.image.resize((msg.info.height,msg.info.width))

        # Turn non-lethal inflated area (< 99) to free space and 100s to 99s
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
        self.segments = self.segment_local_costmap(self.image)

        # create data -- perturbations
        self.create_data()

        self.local_costmap_empty = False

        pd.DataFrame(self.costmap_info_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/costmap_info_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.image).to_csv(self.dirCurr + '/' + self.dirName + '/image.csv', index=False) #, header=False)
        pd.DataFrame(self.fudged_image).to_csv(self.dirCurr + '/' + self.dirName + '/fudged_image.csv', index=False)#, header=False)
        pd.DataFrame(self.segments).to_csv(self.dirCurr + '/' + self.dirName + '/segments.csv', index=False)#, header=False)
        pd.DataFrame(self.data).to_csv(self.dirCurr + '/' + self.dirName + '/data.csv', index=False)#, header=False)
        
    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')

        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        self.plan_tmp = []
        #self.transformed_plan_xs = []
        #self.transformed_plan_ys = []

        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            '''
            p = np.array([self.global_plan_xs[-1], self.global_plan_ys[-1], 0.0])
            pnew = p.dot(r_) + t
            x_temp = round((pnew[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = round((pnew[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)
            '''

        self.global_plan_empty = False

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/global_plan_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/plan_tmp.csv', index=False)#, header=False)
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')
        
        self.local_plan_x_list = [] 
        self.local_plan_y_list = [] 
        self.local_plan_tmp = []

        for i in range(0,len(msg.poses)):
            self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

            x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)

        self.local_plan_empty = False

        pd.DataFrame(self.local_plan_x_list).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_x_list.csv', index=False)#, header=False)
        pd.DataFrame(self.local_plan_y_list).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_y_list.csv', index=False)#, header=False)
        pd.DataFrame(self.local_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/odom_tmp.csv', index=False)#, header=False)

    # Define a callback for the footprint
    def footprint_callback(self, msg):
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        pd.DataFrame(self.footprint_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/footprint_tmp.csv', index=False)#, header=False)

    # Define a callback for the odometry
    def odom_callback(self, msg):
        #print('odom_callback!!!')
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pd.DataFrame(self.amcl_pose_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/amcl_pose_tmp.csv', index=False)#, header=False)

    # Declare subscribers
    def main_(self):
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)



# main function
# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Initialize the ROS Node named 'lime_rt_sub', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_sub', anonymous=True)

# declare transformation buffer
lime_rt_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_obj.tf_listener = tf2_ros.TransformListener(lime_rt_obj.tfBuffer)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()