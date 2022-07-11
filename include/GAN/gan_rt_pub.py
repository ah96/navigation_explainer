#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import tf2_ros
from data.base_dataset import get_params, get_transform
import PIL.Image
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import time


class gan_rt_pub(object):
    # Constructor
    def __init__(self):
        # directory variables
        self.dirCurr = os.getcwd()
        self.dirName = 'gan_rt_data'
        self.file_path_2 = self.dirName + '/plan_tmp.csv'
        self.file_path_3 = self.dirName + '/global_plan_tmp.csv'
        self.file_path_4 = self.dirName + '/costmap_info_tmp.csv'
        self.file_path_6 = self.dirName + '/tf_odom_map_tmp.csv'
        self.file_path_7 = self.dirName + '/tf_map_odom_tmp.csv'
        self.file_path_8 = self.dirName + '/odom_tmp.csv'
        self.file_path_9 = self.dirName + '/local_plan_x_list.csv'
        self.file_path_10 = self.dirName + '/local_plan_y_list.csv'
        self.file_path_11 = self.dirName + '/local_plan_tmp.csv'
        self.file_path_14 = self.dirName + '/image.csv'
        
        # plans' variables
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list_fixed = []
        self.local_plan_x_list_fixed = []
        self.local_plan_tmp_fixed = []
        self.global_plan_empty = True
        self.local_plan_empty = True
        self.local_plan_counter = 0

        # costmap variables
        self.costmap_size = 160
        self.pd_image_size = (self.costmap_size,self.costmap_size) 
        self.local_costmap_empty = True
        self.localCostmapOriginX = 0
        self.localCostmapOriginY = 0
        self.localCostmapResolution = 0
   
        # plotting variables
        self.free_space_shade = 180
        self.obstacle_shade = 255
        self.w = self.h = 1.6

        # point_cloud variables
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]

        # header
        self.header = Header()

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

        self.pub_exp_image = rospy.Publisher('/local_explanation_image', Image, queue_size=10)
        self.pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)
        self.br = CvBridge()
    
    # load lime_rt_sub data
    def load_data(self):
        print_data = False
        try:
            if os.path.getsize(self.file_path_2) == 0 or os.path.exists(self.file_path_2) == False:
                return False
            self.plan_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/plan_tmp.csv')
            if print_data == True:
                print('self.plan_tmp.shape = ', self.plan_tmp.shape)
            
            if os.path.getsize(self.file_path_3) == 0 or os.path.exists(self.file_path_3) == False:
                return False
            self.global_plan_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/global_plan_tmp.csv')
            if print_data == True:
                print('self.global_plan_tmp.shape = ', self.global_plan_tmp.shape)
            
            if os.path.getsize(self.file_path_4) == 0 or os.path.exists(self.file_path_4) == False:
                return False
            self.costmap_info_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/costmap_info_tmp.csv')
            if print_data == True:
                print('self.costmap_info_tmp.shape = ', self.costmap_info_tmp.shape)
            
            if os.path.getsize(self.file_path_6) == 0 or os.path.exists(self.file_path_6) == False:
                return False
            self.tf_odom_map_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/tf_odom_map_tmp.csv')
            if print_data == True:
                print('self.tf_odom_map_tmp.shape = ', self.tf_odom_map_tmp.shape)
            
            if os.path.getsize(self.file_path_7) == 0 or os.path.exists(self.file_path_7) == False:
                return False
            self.tf_map_odom_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/tf_map_odom_tmp.csv')
            if print_data == True:
                print('self.tf_map_odom_tmp.shape = ', self.tf_map_odom_tmp.shape)
            
            if os.path.getsize(self.file_path_8) == 0 or os.path.exists(self.file_path_8) == False:
                return False
            self.odom_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/odom_tmp.csv')
            if print_data == True:
                print('self.odom_tmp.shape = ', self.odom_tmp.shape)

            if os.path.getsize(self.file_path_9) == 0 or os.path.exists(self.file_path_9) == False:
                return False
            self.local_plan_x_list_fixed = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/local_plan_x_list.csv'))
            if print_data == True:
                print('self.local_plan_x_list_fixed.shape = ', self.local_plan_x_list_fixed.shape)
            
            if os.path.getsize(self.file_path_10) == 0 or os.path.exists(self.file_path_10) == False:
                return False         
            self.local_plan_y_list_fixed = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/local_plan_y_list.csv'))
            if print_data == True:
                print('self.local_plan_y_list_fixed.shape = ', self.local_plan_y_list_fixed.shape)
            
            if os.path.getsize(self.file_path_11) == 0 or os.path.exists(self.file_path_11) == False:
                return False        
            self.local_plan_tmp_fixed = pd.read_csv(self.dirCurr + '/' + self.dirName + '/local_plan_tmp.csv')
            if print_data == True:
                print('self.local_plan_tmp_fixed.shape = ', self.local_plan_tmp_fixed.shape)
            
            if os.path.getsize(self.file_path_14) == 0 or os.path.exists(self.file_path_14) == False:
                return False        
            self.image = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/image.csv'))
            if print_data == True:
                print('self.image.shape = ', self.image.shape)
            
            # if anything is empty or not of the right shape do not explain
            if self.plan_tmp.empty or self.global_plan_tmp.empty or self.costmap_info_tmp.empty or self.tf_odom_map_tmp.empty or self.tf_map_odom_tmp.empty or self.odom_tmp.empty or self.local_plan_tmp_fixed.empty:
                return False

            if self.local_plan_x_list_fixed.size == 0 or self.local_plan_y_list_fixed.size == 0 or self.image.size == 0:
                return False

            if self.local_plan_x_list_fixed.shape != self.local_plan_y_list_fixed.shape:
                return False      

            if self.image.shape != self.pd_image_size:
                return False

            if self.costmap_info_tmp.shape != (7, 1) or self.tf_map_odom_tmp.shape != (7, 1) or self.tf_odom_map_tmp.shape != (7, 1) or self.odom_tmp.shape != (6, 1):
                return False

        except:
            return False

        return True

    # explain function
    def explain(self):
        # try to load lime_rt_sub data
        # if data not loaded do not explain
        if self.load_data() == False:
            print('\nData not loaded!')
            return

        # get current local costmap data
        self.localCostmapOriginX = self.costmap_info_tmp.iloc[3,0]
        self.localCostmapOriginY = self.costmap_info_tmp.iloc[4,0]
        self.localCostmapResolution = self.costmap_info_tmp.iloc[0,0]    

        # transform global plan to from /map to /odom
        t = np.asarray([self.tf_map_odom_tmp.iloc[0,0],self.tf_map_odom_tmp.iloc[1,0],self.tf_map_odom_tmp.iloc[2,0]])
        r = R.from_quat([self.tf_map_odom_tmp.iloc[3,0],self.tf_map_odom_tmp.iloc[4,0],self.tf_map_odom_tmp.iloc[5,0],self.tf_map_odom_tmp.iloc[6,0]])
        r_ = np.asarray(r.as_matrix())
        transformed_plan_xs = []
        transformed_plan_ys = []
        #global_plan_start = time.time()
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array([self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], 0.0])
            pnew = p.dot(r_) + t
            x_temp = round((pnew[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = round((pnew[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)
        #global_plan_end = time.time()
        #print('global_plan_transform_time = ', global_plan_end - global_plan_start)

        # save indices of robot's odometry location in local costmap to class variables
        x_odom_index = round((self.odom_tmp.iloc[0,0] - self.localCostmapOriginX) / self.localCostmapResolution)
        y_odom_index = round((self.odom_tmp.iloc[1,0] - self.localCostmapOriginY) / self.localCostmapResolution)

        self.image = np.stack(3 * (self.image,), axis=-1)

        # plot costmap with plans
        #plot_start = time.time()
        fig = plt.figure(frameon=False)
        fig.set_size_inches(self.w, self.h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax) 
        ax.imshow(self.image.astype(np.uint8), aspect='auto')
        ax.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list_fixed, self.local_plan_y_list_fixed, c='yellow', marker='o')
        ax.scatter([x_odom_index], [y_odom_index], c='white', marker='o')
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
        

# ----------main-----------
# main function
# define gan_rt_pub object
gan_rt_pub_obj = gan_rt_pub()

# Initialize the ROS Node named 'gan_rt_pub', allow multiple nodes to be run with this name
rospy.init_node('gan_rt_pub', anonymous=True)

# declare transformation buffer
gan_rt_pub_obj.tfBuffer = tf2_ros.Buffer()
gan_rt_pub_obj.tf_listener = tf2_ros.TransformListener(gan_rt_pub_obj.tfBuffer)

#rate = rospy.Rate(1)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #rate.sleep()
    gan_rt_pub_obj.explain()