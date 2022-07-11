#!/usr/bin/env python3

from data.base_dataset import get_params, get_transform
import PIL.Image
#import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im

from scipy.spatial.transform import Rotation as R

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from nav_msgs.msg import OccupancyGrid, Odometry, Path
import rospy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tf2_ros
import os
import time


# data received from each subscriber is saved to .csv file
class gan_rt(object):
    # Constructor
    def __init__(self):
        # plans' variables
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.local_plan_tmp = [] 
        self.plan_tmp = [] 
        self.global_plan_tmp = []

        # tf variables
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 
        
        # pose variables
        self.odom_tmp = []
        self.odom_x = 0
        self.odom_y = 0
  
        # costmap variables
        self.costmap_info_tmp = [] 
        self.image = np.array([])
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.costmap_size = 160
      
        # plotting variables
        self.free_space_shade = 180
        self.obstacle_shade = 255
        self.w = self.h = 1.6

        # subscribers
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        # directory to save data
        self.dirCurr = os.getcwd()
        self.dirName = 'gan_rt_data'
        try:
            os.mkdir(self.dirName)
        except FileExistsError:
            pass

        self.publish_gan_bool = False
        if self.publish_gan_bool == True:
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

            self.header = Header()
            self.header.frame_id = 'odom'

            # GAN variables
            self.opt = TestOptions().parse()  # get test options
            # hard-code some parameters for test
            self.opt.num_threads = 0   # test code only supports num_threads = 0
            self.opt.batch_size = 1    # test code only supports batch_size = 1
            self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
            self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
            self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
            self.model = create_model(self.opt)      # create a model given opt.model and other options
            self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
            self.input_nc = 3
            self.transform_params = get_params(self.opt, (160, 160)) #input.size)
            self.input_transform = get_transform(self.opt, self.transform_params, grayscale=(self.input_nc == 1))

            # publishers
            self.pub_exp_image = rospy.Publisher('/local_explanation_image', Image, queue_size=10)
            self.pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)
            self.br = CvBridge()

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        #print('\nlocal_costmap_callback!!!')
        # if you can get tf proceed
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
        
        except:
            #print('\nexcept!!!')
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
        self.image[self.image >= 99] = self.obstacle_shade
        self.image[self.image <= 98] = self.free_space_shade

        pd.DataFrame(self.costmap_info_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/costmap_info_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.image).to_csv(self.dirCurr + '/' + self.dirName + '/image.csv', index=False) #, header=False)
        
    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        self.plan_tmp = []

        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/global_plan_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/plan_tmp.csv', index=False)#, header=False)

        if self.publish_gan_bool == True:
            #image = gray2rgb(image)
            self.image = np.stack(3 * (self.image,), axis=-1)

            # save indices of robot's odometry location in local costmap to class variables
            self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

            self.publish_gan()

    # publish gan
    def publish_gan(self):
        # transform global plan to from /map to /odom
        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())
        transformed_plan_xs = []
        transformed_plan_ys = []
        #global_plan_start = time.time()
        for i in range(0, len(self.global_plan_xs)):
            p = np.array([self.global_plan_xs[i], self.global_plan_ys[i], 0.0])
            pnew = p.dot(r_) + t
            x_temp = round((pnew[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = round((pnew[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)
        #global_plan_end = time.time()
        #print('global_plan_transform_time = ', global_plan_end - global_plan_start)

        # plot costmap with plans
        #plot_start = time.time()
        fig = plt.figure(frameon=False)
        fig.set_size_inches(self.w, self.h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax) 
        ax.imshow(self.image.astype(np.uint8), aspect='auto')
        ax.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_xs, self.local_plan_ys, c='yellow', marker='o')
        ax.scatter([self.x_odom_index], [self.y_odom_index], c='white', marker='o')
        fig.canvas.draw()
        fig.canvas.tostring_argb()
        #plot_end = time.time()
        #print('plot_time = ', plot_end - plot_start)

        # get GAN output
        input = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        fig.clf()
        input = self.input_transform(input)
        self.model.set_input_one(input)  # unpack data from data loader
        forward_start = time.time()
        self.model.forward()
        forward_end = time.time()
        output = tensor2im(self.model.fake_B)
        print('gan_output_time = ', forward_end - forward_start)

        # RGB to BGR
        #start_bgr = time.time()
        output = output[:, :, [2, 1, 0]]
        #end_bgr = time.time()
        #print('\nBGR time = ', end_bgr - start_bgr)

        # publish explanation layer
        #points_start = time.time()
        z = 0.0
        a = 255                    
        points = []
        for i in range(0, 160):
            for j in range(0, 160):
                x = self.localCostmapOriginX + i * self.localCostmapResolution
                y = self.localCostmapOriginY + j * self.localCostmapResolution
                r = output[j, i, 2]
                g = output[j, i, 1]
                b = output[j, i, 0]
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [x, y, z, rgb]
                points.append(pt)
        #points_end = time.time()
        #print('\npoints_time = ', points_end - points_start)

        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.pub_exp_pointcloud.publish(pc2)
        #rospy.sleep(1.0)

        # publish explanation image
        #flip_start = time.time()
        output[:,:,0] = np.flip(output[:,:,0], axis=1)
        output[:,:,1] = np.flip(output[:,:,1], axis=1)
        output[:,:,2] = np.flip(output[:,:,2], axis=1)
        #flip_end = time.time()
        #print('\nflip_time = ', flip_end - flip_start)
        output_cv = self.br.cv2_to_imgmsg(output) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3 - it seems so
        self.pub_exp_image.publish(output_cv)

    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        try:        
            self.local_plan_xs = [] 
            self.local_plan_ys = [] 
            self.local_plan_tmp = []

            for i in range(0,len(msg.poses)):
                self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.local_plan_xs.append(x_temp)
                    self.local_plan_ys.append(y_temp)

            # save data to the .csv files
            pd.DataFrame(self.local_plan_xs).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_x_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_ys).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_y_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/odom_tmp.csv', index=False)#, header=False)

        except:
            pass
        
    # Define a callback for the odometry
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]


# ----------main-----------
# main function
# define gan_rt_sub object
gan_rt_obj = gan_rt()

# Initialize the ROS Node named 'gan_rt_sub', allow multiple nodes to be run with this name
rospy.init_node('gan_rt_sub', anonymous=True)

# declare tf2 transformation buffer
gan_rt_obj.tfBuffer = tf2_ros.Buffer()
gan_rt_obj.tf_listener = tf2_ros.TransformListener(gan_rt_obj.tfBuffer)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()






