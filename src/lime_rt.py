#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from skimage.segmentation import slic
from skimage.color import gray2rgb
from scipy.spatial.transform import Rotation as R
from skimage.measure import regionprops
import copy
import tf2_ros
import itertools
import sklearn
import shlex
from psutil import Popen
import math
from functools import partial
from sklearn.utils import check_random_state
import sklearn.metrics
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import lime_base
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from skimage.measure import regionprops
from std_msgs.msg import Float32MultiArray

global br, pub_exp_image, pub_lime, pub_exp_pointcloud
    
class ImageExplanation(object):
    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label):
        #print('get_image_and_mask starting')

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        #print('\nself.local_exp = ', self.local_exp)
        #print('\nexp = ', exp)

        temp = np.zeros(self.image.shape)

        color_free_space = False
        use_maximum_weight = True
        all_weights_zero = False

        val_low = 0.0
        val_high = 255.0
        gray_shade = 180

        w_sum = 0.0
        w_s = []
        for f, w in exp:
            w_sum += abs(w)
            w_s.append(abs(w))
        max_w = max(w_s)
        if max_w == 0:
            all_weights_zero = True
            max_w = 1

        if all_weights_zero == True:
            temp[self.image == 0] = gray_shade
            temp[self.image != 0] = val_low
            return temp, exp        

        for f, w in exp:
            #print('\n(f, w): ', (f, w))

            if w < -0.01:
                c = -1
            elif w > 0.01:
                c = 1
            else:
                c = 0
            #print('c = ', c)
            
            x1 = np.bincount(image[segments == f][:,0] > 0.0)
            x2 = len(image[segments == f][:,0])
            free_space_percentage = x1[0] / x2
            #print('free_space_percentage: ', free_space_percentage)

            # free space
            if free_space_percentage > 0.9:
                if color_free_space == False:
                    temp[segments == f, 0] = gray_shade
                    temp[segments == f, 1] = gray_shade
                    temp[segments == f, 2] = gray_shade
            # obstacle
            else:
                if color_free_space == False:
                    if c == 1:
                        temp[segments == f, 0] = 0.0
                        if use_maximum_weight == True:
                            temp[segments == f, 1] = val_low + (val_high - val_low) * abs(w) / max_w
                        else:
                            temp[segments == f, 1] = val_low + (val_high - val_low) * abs(w) / w_sum 
                        temp[segments == f, 2] = 0.0
                    elif c == 0:
                        temp[segments == f, 0] = 0.0
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                    elif c == -1:
                        if use_maximum_weight == True:
                            temp[segments == f, 0] = val_low + (val_high - val_low) * abs(w) / max_w
                        else:
                            temp[segments == f, 0] = val_low + (val_high - val_low) * abs(w) / w_sum 
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                                        
        #print('get_image_and_mask ending')
        return temp, exp

class lime_rt(object):
    def __init__(self):
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.footprint_tmp = [] 
        self.local_plan_tmp = [] 
        self.plan_tmp = [] 
        self.global_plan_tmp = [] 
        self.costmap_info_tmp = [] 
        self.amcl_pose_tmp = [] 
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 
        self.odom_tmp = []
        self.image_rgb = np.array([]) 
        self.segments = np.array([])
        self.labels = np.array([]) 
        self.data = np.array([]) 
        self.distances = np.array([])
        self.image = np.array([])
        self.odom_x = 0
        self.odom_y = 0
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.original_deviation = 0

        #'''
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
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=random_state)
        #'''

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]

        self.header = Header()

        self.start = time.time()
        self.end = time.time()
            
    # save data for local planner in explanation with image
    def SaveImageDataForLocalPlanner(self):
        # Saving data to .csv files for C++ node - local navigation planner
        # Save footprint instance to a file
        self.footprint_tmp = pd.DataFrame(self.footprint_tmp)#.transpose()
        self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

        # Save local plan instance to a file
        self.local_plan_tmp = pd.DataFrame(self.local_plan_tmp)#.transpose()
        self.local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan (from global planner) instance to a file
        self.plan_tmp = pd.DataFrame(self.plan_tmp)#.transpose()
        self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save global plan instance to a file
        self.global_plan_tmp = pd.DataFrame(self.global_plan_tmp)#.transpose()
        self.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)

        # Save costmap_info instance to file
        self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
        self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose instance to file
        self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
        self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map instance to file
        self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
        self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom instance to file
        self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
        self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save odometry instance to file
        self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
        self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

    # classifier function for lime image
    def classifier_fn(self, sampled_instance):
        print('\nclassifier_fn_image_lime started')

        print('\nsampled_instance.shape = ', sampled_instance.shape)

        sampled_instance_shape_len = len(sampled_instance.shape)
        sample_size = 1 if sampled_instance_shape_len == 2 else sampled_instance.shape[0]

        if sampled_instance_shape_len > 3:
            temp = np.delete(sampled_instance,2,3)
            temp = np.delete(temp,1,3)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt('/home/amar/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt('/home/amar/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instance_shape_len == 2:
            np.savetxt('/home/amar/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")

        # calling ROS C++ node
        #print('\nstarting C++ node')

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        #print('\nC++ node ended')

        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')

        # load transformed global plan to /odom frame
        transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')

        costmap_size = 160

        # fill the list of transformed plan coordinates
        self.transformed_plan_xs = []
        self.transformed_plan_ys = []
        for i in range(0, transformed_plan.shape[0]):
            x_temp = int((transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < costmap_size and 0 <= y_temp < costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)

        # calculate original deviation - sum of minimal point-to-point distances
        original_deviation = -1.0
        diff_x = 0
        diff_y = 0
        devs = []
        for j in range(0, len(self.local_plan_x_list)):
            local_diffs = []
            for k in range(0, len(self.transformed_plan_xs)):
                diff_x = (self.local_plan_x_list[j] - self.transformed_plan_xs[k]) ** 2
                diff_y = (self.local_plan_y_list[j] - self.transformed_plan_ys[k]) ** 2
                diff = math.sqrt(diff_x + diff_y)
                local_diffs.append(diff)                        
            devs.append(min(local_diffs))   
        self.original_deviation = sum(devs)
        #print('\noriginal_deviation = ', original_deviation)
        # original_deviation for big_deviation = 745.5051688094327
        # original_deviation for big_deviation without wall = 336.53749938826286
        # original_deviation for no_deviation = 56.05455197218764
        # original_deviation for small_deviation = 69.0
        # original_deviation for rotate_in_place = 307.4962940090125

        # DETERMINE THE DEVIATION TYPE
        determine_dev_type = True
        if determine_dev_type == True:
            #start_determine_dev = time.time()
            
            # thresholds
            local_plan_gap_threshold = 48.0
            big_deviation_threshold = 85.0
            small_deviation_threshold = 32.0 #30
            no_deviation_threshold = 0.0

            # test for the original local plan gap
            local_plan_original_gap = False
            local_plan_gaps = []
            print('LOCAL PLAN LENGTH = ', len(self.local_plan_x_list))
            diff = 0
            for j in range(0, len(self.local_plan_x_list) - 1):
                diff = math.sqrt((self.local_plan_x_list[j]-self.local_plan_x_list[j+1])**2 + (self.local_plan_y_list[j]-self.local_plan_y_list[j+1])**2 )
                local_plan_gaps.append(diff)
            if max(local_plan_gaps) > local_plan_gap_threshold:
                local_plan_original_gap = True

            # local gap too big - stop (rotate_in_place)
            if local_plan_original_gap == True:
                deviation_type = 'stop'
                local_plan_gap_threshold = 55.0
            
            # no local gap - test further    
            elif original_deviation >= big_deviation_threshold:
                deviation_type = 'big_deviation'
            elif original_deviation >= small_deviation_threshold:
                deviation_type = 'small_deviation'
            else:
                deviation_type = 'no_deviation'    

            #end_determine_dev = time.time()
            #determine_dev_time = end_determine_dev - start_determine_dev
            #print('\ndetermine deviation type runtime = ', determine_dev_time)

            # PRINTING RESULTS                                       
            print('\ndeviation_type: ', deviation_type)

        local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sample_size), columns=['deviate'])

        # fill in deviation dataframe
        dev_original = 0
        #for i in range(0, sampled_instance.shape[0]):
        for i in range(0, sample_size):
            #print('\ni = ', i)
            local_plan_xs = []
            local_plan_ys = []
            local_plan_found = False
            
            # find if there is local plan
            local_plans_local = local_plans.loc[local_plans['ID'] == i]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < 160 and 0 <= y_temp < 160:
                        local_plan_xs.append(x_temp)
                        local_plan_ys.append(y_temp)
                        local_plan_found = True
            
            # this happens almost never when only obstacles are segments, but let it stay for now
            if local_plan_found == False:
                if deviation_type == 'stop':
                    local_plan_deviation.iloc[i, 0] = dev_original
                elif deviation_type == 'no_deviation':
                    local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                    local_plan_deviation.iloc[i, 0] = 0.0
                continue             

            # find deviation as a sum of minimal point-to-point differences
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, len(local_plan_xs)):
                local_diffs = []
                for k in range(0, len(self.transformed_plan_xs)):
                    diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))   

            if i == 0:
                dev_original = sum(devs)    

            local_plan_deviation.iloc[i, 0] = sum(devs)

        cmd_vel_perturb['deviate'] = local_plan_deviation
        
        print('\nclassifier_fn_image_lime ended\n')

        return np.array(cmd_vel_perturb.iloc[:, 3:])

    # segmentation algorithm
    def sm_only_obstacles(self, image, img_rgb, x_odom, y_odom, plan_x_list, plan_y_list):
        print('segmentation algorithm')
        # show original image
        img = copy.deepcopy(image)

        #start = time.time()

        #regions = regionprops(img.astype(int))
        #for props in regions:
            #v = props.label  # value of label
            #cx, cy = props.centroid  # centroid coordinates
            #print("(cy, cx, v) = ", (cy, cx, v))   
            
        '''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_rgb.astype('float64'), aspect='auto')
        fig.savefig('img_rgb.png', transparent=False)
        fig.clf()
        '''

        # Find segments_slic
        segments_slic = slic(img_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                                multichannel=True, convert2lab=True,
                                enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                                start_label=1, mask=None)

        '''
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
        '''

        segments = np.zeros(img.shape, np.uint8)


        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)
        #print('abs(k) = ', abs(k)) 

        # make one free space segment
        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        num_of_obstacles = 0
        # add obstacle segments        
        for i in np.unique(segments_slic):
            temp = img[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            #print('count: ', count)
            #print('temp: ', temp)
            #print('len(temp): ', temp.shape[0])
            if np.all(img[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                #print('obstacle')
                segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1

        #print('num_of_obstacles: ', num_of_obstacles)        

        if 8 > num_of_obstacles > 0:
            # divide segment obstacles    
            seg_labels = np.unique(segments)[1:]        

            num_of_seg = len(seg_labels)
            num_of_wanted_seg = 8

            #print('\nnumber of wanted segments: ', num_of_wanted_seg)
            #print('number of current segments: ', num_of_seg)

            seg_sizes = []

            if num_of_seg < num_of_wanted_seg:
                for i in range(0, num_of_seg):
                    #print(len(segments[segments == seg_labels[i]]))
                    seg_sizes.append(len(segments[segments == seg_labels[i]]))

            #print('\nsizes of segments original: ', seg_sizes)
            #print('labels of segments original: ', seg_labels)
            seg_labels = [x for _, x in sorted(zip(seg_sizes, seg_labels))]
            seg_labels.reverse()
            #print('\nsizes of segemnts sorted: ', seg_sizes)
            #print('labels of segemnts sorted: ', seg_labels)

            seg_missing = num_of_wanted_seg - num_of_seg
            #print('\nnumber of segments missing: ', seg_missing)

            # if a number of missing segments is smaller or equal than the number of existing segments
            if seg_missing <= num_of_seg:
                label_current = len(seg_labels) + 1
                for i in range(0, seg_missing):
                    temp = segments[segments == seg_labels[i]]
                    #print('temp = ', temp)

                    # check obstacle shape
                    w_min = 161
                    w_max = -1
                    h_min = 161
                    h_max = -1
                    for j in range(0, segments.shape[0]):
                        for q in range(0, segments.shape[1]):
                            if segments[j, q] == seg_labels[i]:
                                if j > h_max:
                                    h_max = j
                                if j < h_min:
                                    h_min = j
                                if q > w_max:
                                    w_max = q
                                if q < w_min:
                                    w_min = q           

                    #print('\n(h_min, h_max): ', (h_min, h_max))
                    #print('(w_min, w_max): ', (w_min, w_max))

                    height = h_max - h_min + 1
                    #print('\nheight', height)
                    width = w_max - w_min + 1
                    #print('width', width)
            

                    # if upright
                    if abs(k) >= 1:
                        if height > width:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == seg_labels[i]] = temp
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            finished = False
                            for q in range(0, segments.shape[1]):
                                if finished == True:
                                    break
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        segments[j, q] = label_current
                                        counter += 1
                                        if counter == int(num_of_pixels / 2 + 0.5):
                                            label_current += 1
                                            finished = True
                                            break        

                    # if to the side
                    elif abs(k) < 1:
                        #print('OVAJ SLUCAJ')
                        if width > height:
                            #print('width > height')
                            #print('label_current: ', label_current)
                            label_current
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == seg_labels[i]] = temp   
                        else:
                            #print('width < height')
                            #print('label_current: ', label_current)
                            label_original = temp[0]
                            #print('label_original = ', label_original)
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        #print('IN')
                                        #print('counter = ', counter)
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1

                            '''
                            for j in range(0, height):
                                for q in range(0, int(width/2)):
                                    if j * width + q > len(temp) - 1:
                                        continue
                                    temp[j * width + q] = label_original
                                for q in range(int(width/2), width):
                                    if j * width + q > len(temp) - 1:
                                        continue
                                    temp[j * width + q] = label_current
                            '''
                    
                    label_current += 1

            # if a number of missing segments is greater than the number of existing segments
            else:
                num_of_new_seg_per_old_seg = int(seg_missing / num_of_seg)
                
                # if a number of new segment per old segments is integer and same for all old segments
                if num_of_new_seg_per_old_seg == seg_missing / num_of_seg:
                    #print('CIO BROJ')
                    #print('\nnumber of new segments per existing segment: ', num_of_new_seg_per_old_seg)

                    label_current = len(seg_labels) + 1
                    
                    for i in range(0, num_of_seg):
                        temp = segments[segments == seg_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == seg_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height > width:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break
                        # if to the side
                        else:
                            #print('UPRIGHT')
                            if width > height:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                # if a number of new segment per old segments is integer and same for all old segments
                else:
                    #print('NON-CIO BROJ')

                    whole_part = int(seg_missing / num_of_seg)
                    #print('whole part = ', whole_part)
                    rest = seg_missing % num_of_seg
                    #print('rest = ', rest)

                    put_rest_to_biggest_segment = False

                    num_of_new_seg_per_old_seg_list = [whole_part] * num_of_seg

                    if put_rest_to_biggest_segment == False:
                        for i in range(0, rest):
                            num_of_new_seg_per_old_seg_list[i] += 1
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)    
                    else:
                        num_of_new_seg_per_old_seg_list[0] += rest
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)


                    label_current = len(seg_labels) + 1
                    #print('\nlabel_current = ', label_current)
                    
                    for i in range(0, num_of_seg):
                        temp = segments[segments == seg_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == seg_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height > width:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg_list[i]:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                        # if to the side
                        else:
                            #print('SIDE')
                            if width > height:
                                #print('WIDTH')
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    #print('label_current: ', label_current)
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                #print('HEIGHT')
                                #print('len(temp): ', len(temp))
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                #print('step: ', step)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    #print('counter + 1 = ', counter + 1)
                                                    #print('label_current = ', label_current)
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg_list[i]:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                #print('counter_end = ', counter)
                                                label_current += 1
                                                finished = True
                                                break


        #end = time.time()

        #print("\nsm7 runtime: ", end - start)

        '''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments_final.png', transparent=False)
        fig.clf()
        '''

        # fix labels of segments
        seg_labels = np.unique(segments)
        for i in range(1, len(seg_labels)):
            label = seg_labels[i]
            if label != i:
                segments[segments == label] = i

        #print('\nnp.unique(segments): ', np.unique(segments))
        #print('\nlen(np.unique(segments)): ', len(np.unique(segments)))

        if len(np.unique(segments)) > 9:
            # make one free space segment
            ctr = 0
            segments[:, :] = ctr
            ctr = ctr + 1

            num_of_obstacles = 0
            # add obstacle segments        
            for i in np.unique(segments_slic):
                if np.all(img[segments_slic == i] == 99):
                    #print('obstacle')
                    segments[segments_slic == i] = ctr
                    ctr = ctr + 1
                    num_of_obstacles += 1
            '''
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig('segments_final_corrected.png', transparent=False)
            fig.clf()
            '''

        return segments

    # call teb
    def data_labels(self, image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10):
        #print('data_labels starts')

        n_features = np.unique(segments).shape[0]

        num_samples = n_features
        lst = [[1]*n_features]
        for i in range(1, num_samples):
            lst.append([1]*n_features)
            lst[i][n_features-i] = 0    
        data = np.array(lst).reshape((num_samples, n_features))

        labels = []
        
        imgs = []
        rows = data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        #print('data_labels ends')

        return data, np.array(labels)

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')

        # save costmap in a right image format
        self.localCostmapOriginX = msg.info.origin.position.x
        self.localCostmapOriginY = msg.info.origin.position.y
        self.localCostmapResolution = msg.info.resolution

        self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

        self.image = np.asarray(msg.data)
        self.image.resize((msg.info.height,msg.info.width))

        # Turn inflated area to free space and 100s to 99s
        self.image[self.image == 100] = 99
        self.image[self.image <= 98] = 0

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        self.image = self.image * 1.0

        self.image_rgb = gray2rgb(self.image)
        #image = np.stack(3 * (image,), axis=-1)

        # my change - to return grayscale to classifier_fn
        self.fudged_image = self.image.copy()
        self.fudged_image[:] = hide_color = 0

        # save indices of robot's odometry location in local costmap to class variables
        self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

        # find segments
        self.segments = self.sm_only_obstacles(self.image, self.image_rgb, self.x_odom_index, self.y_odom_index, self.transformed_plan_xs, self.transformed_plan_ys)

        # save data for teb
        self.SaveImageDataForLocalPlanner()

        # call teb
        self.labels=(1,)
        self.top = self.labels
        self.data, self.labels = self.data_labels(self.image, self.fudged_image, self.segments,
                                        self.classifier_fn, num_samples=1000,
                                        batch_size=2048)

        # find distances
        # distance_metric = 'jaccard' - alternative distance metric
        distance_metric='cosine'
        self.distances = sklearn.metrics.pairwise_distances(
            self.data,
            self.data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # Explanation part
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'

        # find explanation
        ret_exp = ImageExplanation(self.image_rgb, self.segments)
        if top_labels:
            #print('top_labels usao')
            top = np.argsort(self.labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        #print('labels =', self.labels)
        for label in top:
            #print('label = ', label)
            (ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                self.data, self.labels, self.distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=feature_selection)

        # get explanation image
        output, exp = ret_exp.get_image_and_mask(label=0)
        print('\nexp = ', exp)

        output = output[:, :, [2, 1, 0]].astype(np.uint8)
        # publish explanation layer
        #points_start = time.time()
        z = 0.0
        a = 255                    
        points = []
        for i in range(0, 160):
            for j in range(0, 160):
                x = self.localCostmapOriginX + i * self.localCostmapResolution
                y = self.localCostmapOriginY + j * self.localCostmapResolution
                r = int(output[j, i, 2])
                g = int(output[j, i, 1])
                b = int(output[j, i, 0])
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [x, y, z, rgb]
                points.append(pt)
        #points_end = time.time()
        #print('\npoints_time = ', points_end - points_start)
        self.header.frame_id = 'odom'
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        pub_exp_pointcloud.publish(pc2)
        #rospy.sleep(1.0)

        # publish explanation image
        output[:,:,0] = np.flip(output[:,:,0], axis=1)
        output[:,:,1] = np.flip(output[:,:,1], axis=1)
        output[:,:,2] = np.flip(output[:,:,2], axis=1)
        #print('\nBGR time = ', end_bgr - start_bgr)
        output_cv = br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3 - it seems so
        pub_exp_image.publish(output_cv)

        # publish explanation coefficients
        transf = tfBuffer.lookup_transform('odom', 'map', rospy.Time())
        exp_with_centroids = Float32MultiArray()
        self.segments += 1
        regions = regionprops(self.segments.astype(int))
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            for j in range(0, len(exp)):
                if exp[j][0] == v - 1:
                    cx = cx*self.localCostmapResolution + self.localCostmapOriginX
                    cy = cy*self.localCostmapResolution + self.localCostmapOriginY
                    p = np.array([cx, cy, 0.0])
                    t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
                    r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
                    r_ = np.asarray(r.as_matrix())
                    pnew = p.dot(r_) + t
                    exp_with_centroids.data.append(pnew[0])
                    exp_with_centroids.data.append(pnew[1])
                    exp_with_centroids.data.append(exp[j][1])
                    break
        exp_with_centroids.data.append(self.original_deviation) # append original deviation as the last element
        pub_lime.publish(exp_with_centroids)
        #self.segments-1

    # Define a callback for the local plan
    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        self.plan_tmp = []
        self.transformed_plan_xs = []
        self.transformed_plan_ys = []

        # catch transform from /map to /odom and vice versa
        transf = tfBuffer.lookup_transform('map', 'odom', rospy.Time())
        t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
        r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
        r_ = np.asarray(r.as_matrix())

        transf_ = tfBuffer.lookup_transform('odom', 'map', rospy.Time())

        self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
        self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]

        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            p = np.array([self.global_plan_xs[-1], self.global_plan_ys[-1], 0.0])
            pnew = p.dot(r_) + t
            x_temp = round((pnew[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = round((pnew[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < 160 and 0 <= y_temp < 160:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)

    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        print('\nlocal_plan_callback')
        
        self.local_plan_x_list = [] 
        self.local_plan_y_list = [] 
        self.local_plan_tmp = []

        for i in range(0,len(msg.poses)):
            self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

            x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < 160 and 0 <= y_temp < 160:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)
                
    # Define a callback for the footprint
    def footprint_callback(self, msg):
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
            
    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

    
# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('lime_rt', anonymous=True)

lime_rt_obj = lime_rt()

tfBuffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tfBuffer)

sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, lime_rt_obj.local_plan_callback)

sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, lime_rt_obj.global_plan_callback)

sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, lime_rt_obj.footprint_callback)

# Initalize a subscriber to the odometry
sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, lime_rt_obj.odom_callback)

sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, lime_rt_obj.amcl_callback)

# Initalize a subscriber to the local costmap
sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, lime_rt_obj.local_costmap_callback)

pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
br = CvBridge()

pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)

pub_lime = rospy.Publisher("/lime_exp", Float32MultiArray, queue_size=10)


# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()