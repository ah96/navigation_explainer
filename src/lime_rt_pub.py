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
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from skimage.measure import regionprops
from std_msgs.msg import Float32MultiArray
import scipy as sp

from lime_rt_sub import lime_rt_obj


#global br, pub_exp_image, pub_lime, pub_exp_pointcloud

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

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
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
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

        self.color_free_space = False
        self.use_maximum_weight = False
        self.all_weights_zero = False


        self.val_low = 0.0
        self.val_high = 255.0
        self.gray_shade = 180

    def get_image_and_mask(self, label):
        #print('get_image_and_mask starting')

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        #segments = self.segments
        #image = self.image
        exp = self.local_exp[label]

        temp = np.zeros(self.image.shape)

        w_sum = 0.0
        w_s = []
        for f, w in exp:
            w_sum += abs(w)
            w_s.append(abs(w))
        max_w = max(w_s)
        if max_w == 0:
            self.all_weights_zero = True
            max_w = 1

        if self.all_weights_zero == True:
            temp[self.image == 0] = self.gray_shade
            temp[self.image != 0] = self.val_low
            return temp, exp

        print('exp = ', exp)
        print('np.unique(self.segments) = ', np.unique(self.segments))

        for f, w in exp:
            print('\n(f, w): ', (f, w))

            if w < -0.01:
                c = -1
            elif w > 0.01:
                c = 1
            else:
                c = 0
            #print('c = ', c)
            
            '''
            x1 = np.bincount(self.image[self.segments == f][:,0] > 0.0)
            if x1 == []:
                free_space_percentage = 1.0
            else:
                print('x1 = ', x1)
                x2 = len(self.image[self.segments == f][:,0])
                free_space_percentage = x1[0] / x2
            #print('free_space_percentage: ', free_space_percentage)
            '''

            # free space
            if f == 0:
                print('free_space, (f, w) = ', (f, w))
                if self.color_free_space == False:
                    temp[self.segments == f, 0] = self.gray_shade
                    temp[self.segments == f, 1] = self.gray_shade
                    temp[self.segments == f, 2] = self.gray_shade
            # obstacle
            else:
                if self.color_free_space == False:
                    if c == 1:
                        temp[self.segments == f, 0] = 0.0
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 2] = 0.0
                    elif c == 0:
                        temp[self.segments == f, 0] = 0.0
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                                        
        #print('get_image_and_mask ending')
        return temp, exp

class lime_rt_pub(object):
    # Constructor
    def __init__(self):
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.labels = np.array([]) 
        self.distances = np.array([])
        self.costmap_size = 160
        self.original_deviation = 0

        self.local_plan_x_list_fixed = []
        self.local_plan_x_list_fixed = []
        self.local_plan_tmp_fixed = []

        self.global_plan_empty = True
        self.local_costmap_empty = True
        self.local_plan_empty = True

        self.labels = []
        self.distances = []
        self.num_samples = 0
        self.n_features = 0

        self.divide_obstacles = False

        self.local_plan_counter = 0

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
        self.base = LimeBase(kernel_fn, verbose, random_state=random_state)
        #'''

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]

        self.header = Header()
            
    # save data for local planner in explanation with image
    def SaveImageDataForLocalPlanner(self):
        # Saving data to .csv files for C++ node - local navigation planner
        # Save footprint instance to a file
        lime_rt_obj.footprint_tmp = pd.DataFrame(lime_rt_obj.footprint_tmp)#.transpose()
        lime_rt_obj.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

        # Save local plan instance to a file
        lime_rt_obj.local_plan_tmp_fixed = pd.DataFrame(lime_rt_obj.local_plan_tmp_fixed)#.transpose()
        lime_rt_obj.local_plan_tmp_fixed.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan (from global planner) instance to a file
        lime_rt_obj.plan_tmp = pd.DataFrame(lime_rt_obj.plan_tmp)#.transpose()
        lime_rt_obj.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save global plan instance to a file
        lime_rt_obj.global_plan_tmp = pd.DataFrame(lime_rt_obj.global_plan_tmp)#.transpose()
        lime_rt_obj.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)

        # Save costmap_info instance to file
        lime_rt_obj.costmap_info_tmp = pd.DataFrame(lime_rt_obj.costmap_info_tmp).transpose()
        lime_rt_obj.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose instance to file
        lime_rt_obj.amcl_pose_tmp = pd.DataFrame(lime_rt_obj.amcl_pose_tmp).transpose()
        lime_rt_obj.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map instance to file
        lime_rt_obj.tf_odom_map_tmp = pd.DataFrame(lime_rt_obj.tf_odom_map_tmp).transpose()
        lime_rt_obj.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom instance to file
        lime_rt_obj.tf_map_odom_tmp = pd.DataFrame(lime_rt_obj.tf_map_odom_tmp).transpose()
        lime_rt_obj.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save odometry instance to file
        lime_rt_obj.odom_tmp = pd.DataFrame(lime_rt_obj.odom_tmp).transpose()
        lime_rt_obj.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

    # classifier function for lime image
    def classifier_fn(self, sampled_instance):
        #print('\nclassifier_fn_image_lime started')

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

        #rospy.sleep(0.4)

        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')

        # load transformed global plan to /odom frame
        transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        
        # fill the list of transformed plan coordinates
        self.transformed_plan_xs = []
        self.transformed_plan_ys = [] 
        for i in range(0, transformed_plan.shape[0]):
            if math.isnan(transformed_plan.iloc[i, 0]) == True or math.isnan(transformed_plan.iloc[i, 1]) == True:
                continue
            x_temp = int((transformed_plan.iloc[i, 0] - lime_rt_obj.localCostmapOriginX) / lime_rt_obj.localCostmapResolution)
            y_temp = int((transformed_plan.iloc[i, 1] - lime_rt_obj.localCostmapOriginY) / lime_rt_obj.localCostmapResolution)

            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)

        # calculate original deviation - sum of minimal point-to-point distances
        calculate_original_deviation = True
        if calculate_original_deviation == True:
            original_deviation = -1.0
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, len(self.local_plan_x_list_fixed)):
                local_diffs = []
                for k in range(0, len(self.transformed_plan_xs)):
                    diff_x = (self.local_plan_x_list_fixed[j] - self.transformed_plan_xs[k]) ** 2
                    diff_y = (self.local_plan_y_list_fixed[j] - self.transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)
                if local_diffs == []:
                    continue                               
                devs.append(min(local_diffs))   
            self.original_deviation = sum(devs)
            #print('\noriginal_deviation = ', original_deviation)
            # original_deviation for big_deviation = 745.5051688094327
            # original_deviation for big_deviation without wall = 336.53749938826286
            # original_deviation for no_deviation = 56.05455197218764
            # original_deviation for small_deviation = 69.0
            # original_deviation for rotate_in_place = 307.4962940090125

        # DETERMINE THE DEVIATION TYPE
        determine_dev_type = False
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
            print('LOCAL PLAN LENGTH = ', len(self.local_plan_x_list_fixed))
            diff = 0
            for j in range(0, len(self.local_plan_x_list_fixed) - 1):
                diff = math.sqrt((self.local_plan_x_list_fixed[j]-self.local_plan_x_list_fixed[j+1])**2 + (self.local_plan_y_list_fixed[j]-self.local_plan_y_list_fixed[j+1])**2 )
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
        #dev_original = 0
        #for i in range(0, sampled_instance.shape[0]):
        for i in range(0, sample_size):
            #print('\ni = ', i)
            local_plan_xs = []
            local_plan_ys = []
            #local_plan_found = False
            
            # find if there is local plan
            local_plans_local = local_plans.loc[local_plans['ID'] == i]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - lime_rt_obj.localCostmapOriginX) / lime_rt_obj.localCostmapResolution)
                    y_temp = int((local_plans_local.iloc[j, 1] - lime_rt_obj.localCostmapOriginY) / lime_rt_obj.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        local_plan_xs.append(x_temp)
                        local_plan_ys.append(y_temp)
                        #local_plan_found = True
            
            # this happens almost never when only obstacles are segments, but let it stay for now
            '''
            if local_plan_found == False:
                if deviation_type == 'stop':
                    local_plan_deviation.iloc[i, 0] = dev_original
                elif deviation_type == 'no_deviation':
                    local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                    local_plan_deviation.iloc[i, 0] = 0.0
                continue
            '''             

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

            #if i == 0:
            #    dev_original = sum(devs)    

            local_plan_deviation.iloc[i, 0] = sum(devs)

        
        
        #print('\nclassifier_fn_image_lime ended\n')

        cmd_vel_perturb['deviate'] = local_plan_deviation
        return np.array(cmd_vel_perturb.iloc[:, 3:])
        #return local_plan_deviation

    # call teb
    def create_labels(self, image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    batch_size=10):
        #print('data_labels starts')

        # call teb and get labels
        self.labels = []
        imgs = []
        rows = self.data
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
                self.labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            self.labels.extend(preds)

        #print('data_labels ends')

        self.labels = np.array(self.labels)

    # explain
    def explain(self):
        print('\nexplain!!!')

        if lime_rt_obj.local_plan_empty == True or lime_rt_obj.global_plan_empty == True or lime_rt_obj.local_costmap_empty == True or lime_rt_obj.data == []:
            print('something empty!!!')
            return

        self.local_plan_x_list_fixed = copy.deepcopy(lime_rt_obj.local_plan_x_list)
        self.local_plan_y_list_fixed = copy.deepcopy(lime_rt_obj.local_plan_y_list)
        self.local_plan_tmp_fixed = copy.deepcopy(lime_rt_obj.local_plan_tmp)
        self.segments = copy.deepcopy(lime_rt_obj.segments)
        self.data = copy.deepcopy(lime_rt_obj.data)

        # save data for teb
        self.SaveImageDataForLocalPlanner()

        # call teb
        self.labels=(1,)
        self.top = self.labels
        self.create_labels(lime_rt_obj.image, lime_rt_obj.fudged_image, self.segments,
                                        self.classifier_fn, batch_size=2048)

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
        ret_exp = ImageExplanation(lime_rt_obj.image_rgb, self.segments)
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
                x = lime_rt_obj.localCostmapOriginX + i * lime_rt_obj.localCostmapResolution
                y = lime_rt_obj.localCostmapOriginY + j * lime_rt_obj.localCostmapResolution
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
        transf = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
        self.pub_exp_pointcloud.publish(pc2)
        #rospy.sleep(1.0)

        # publish explanation image
        output[:,:,0] = np.flip(output[:,:,0], axis=1)
        output[:,:,1] = np.flip(output[:,:,1], axis=1)
        output[:,:,2] = np.flip(output[:,:,2], axis=1)
        #print('\nBGR time = ', end_bgr - start_bgr)
        output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3 - it seems so
        self.pub_exp_image.publish(output_cv)

        # publish explanation coefficients
        exp_with_centroids = Float32MultiArray()
        self.segments += 1
        regions = regionprops(self.segments.astype(int))
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            for j in range(0, len(exp)):
                if exp[j][0] == v - 1:
                    px = cy*lime_rt_obj.localCostmapResolution + lime_rt_obj.localCostmapOriginX
                    py = cx*lime_rt_obj.localCostmapResolution + lime_rt_obj.localCostmapOriginY
                    p = np.array([px, py, 0.0])
                    t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
                    r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
                    r_ = np.asarray(r.as_matrix())
                    pnew = p.dot(r_) + t
                    exp_with_centroids.data.append(px) #(pnew[0])
                    exp_with_centroids.data.append(py) #(pnew[1])
                    exp_with_centroids.data.append(exp[j][1])
                    break
        exp_with_centroids.data.append(self.original_deviation) # append original deviation as the last element
        self.pub_lime.publish(exp_with_centroids)
    
    # initialize publishers
    def main_(self):        
        self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
        self.br = CvBridge()

        self.pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)

        self.pub_lime = rospy.Publisher("/lime_exp", Float32MultiArray, queue_size=10)


lime_rt_pub_obj = lime_rt_pub()

lime_rt_pub_obj.main_()

# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_pub', anonymous=True)

rate = rospy.Rate(1)


# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    print('spinning lime_rt_pub')
    lime_rt_pub_obj.explain()
    rate.sleep()
    #rospy.spin()