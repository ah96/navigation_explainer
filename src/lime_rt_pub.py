#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
from skimage.color import gray2rgb
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
from std_msgs.msg import Float32MultiArray
import scipy as sp
import tf2_ros
from skimage.measure import regionprops
import time
import csv


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
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

        self.color_free_space = False
        self.use_maximum_weight = True
        self.all_weights_zero = False


        self.val_low = 0.0
        self.val_high = 255.0
        self.free_space_shade = 180

    def get_image_and_mask(self, segments, label):

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
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
            temp[self.image == 0] = self.free_space_shade
            temp[self.image != 0] = 0.0
            return temp, exp

        segments_labels = np.unique(segments)
        
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
                                        
        return temp, exp

class lime_rt_pub(object):
    # Constructor
    def __init__(self):
        self.counter_global = 0

        # directory variables
        self.dirCurr = os.getcwd()
        self.dirName = 'lime_rt_data'
        self.file_path_1 = self.dirName + '/footprint_tmp.csv'
        self.file_path_2 = self.dirName + '/plan_tmp.csv'
        self.file_path_3 = self.dirName + '/global_plan_tmp.csv'
        self.file_path_4 = self.dirName + '/costmap_info_tmp.csv'
        self.file_path_5 = self.dirName + '/amcl_pose_tmp.csv' 
        self.file_path_6 = self.dirName + '/tf_odom_map_tmp.csv'
        self.file_path_7 = self.dirName + '/tf_map_odom_tmp.csv'
        self.file_path_8 = self.dirName + '/odom_tmp.csv'
        self.file_path_9 = self.dirName + '/local_plan_x_list.csv'
        self.file_path_10 = self.dirName + '/local_plan_y_list.csv'
        self.file_path_11 = self.dirName + '/local_plan_tmp.csv'
        self.file_path_12 = self.dirName + '/segments.csv'
        self.file_path_13 = self.dirName + '/data.csv'
        self.file_path_14 = self.dirName + '/image.csv'
        self.file_path_15 = self.dirName + '/fudged_image.csv'

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
        
        # point_cloud variables
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]

        # header
        self.header = Header()

        # bool variables
        self.divide_obstacles = False
        self.publish_explanation_coeffs = True  
        self.publish_explanation_image = True
        self.publish_pointcloud = False
            
    # load lime_rt_sub data
    def load_data(self):
        print_data = False
        try:
            if os.path.getsize(self.file_path_1) == 0 or os.path.exists(self.file_path_1) == False:
                return False
            self.footprint_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/footprint_tmp.csv')
            if print_data == True:
                print('self.footprint_tmp.shape = ', self.footprint_tmp.shape)
            
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
            
            if os.path.getsize(self.file_path_5) == 0 or os.path.exists(self.file_path_5) == False:
                return False
            self.amcl_pose_tmp = pd.read_csv(self.dirCurr + '/' + self.dirName + '/amcl_pose_tmp.csv')
            if print_data == True:
                print('self.amcl_pose_tmp.shape = ', self.amcl_pose_tmp.shape)
            
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
            
            if os.path.getsize(self.file_path_12) == 0 or os.path.exists(self.file_path_12) == False:
                return False       
            self.segments = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/segments.csv'))
            if print_data == True:
                print('self.segments.shape = ', self.segments.shape)
               
            if os.path.getsize(self.file_path_13) == 0 or os.path.exists(self.file_path_13) == False:
                return False       
            self.data = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/data.csv'))
            if print_data == True:
                print('self.data.shape = ', self.data.shape)
            
            if os.path.getsize(self.file_path_14) == 0 or os.path.exists(self.file_path_14) == False:
                return False        
            self.image = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/image.csv'))
            if print_data == True:
                print('self.image.shape = ', self.image.shape)
            
            if os.path.getsize(self.file_path_15) == 0 or os.path.exists(self.file_path_15) == False:
                return False        
            self.fudged_image = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/fudged_image.csv'))
            if print_data == True:
                print('self.fudged_image.shape = ', self.fudged_image.shape)

            #if os.path.getsize(self.file_path_15) == 0 or os.path.exists(self.file_path_15) == False:
            #    return False        
            #self.segments_not_inflated = np.array(pd.read_csv(self.dirCurr + '/' + self.dirName + '/segments_not_inflated.csv'))
            #if print_data == True:
            #    print('self.segments_not_inflated.shape = ', self.segments_not_inflated.shape)

            # if anything is empty or not of the right shape do not explain
            if self.plan_tmp.empty or self.global_plan_tmp.empty or self.costmap_info_tmp.empty or self.amcl_pose_tmp.empty or self.tf_odom_map_tmp.empty or self.tf_map_odom_tmp.empty or self.odom_tmp.empty or self.local_plan_tmp_fixed.empty or self.footprint_tmp.empty: 
                return False

            if self.local_plan_x_list_fixed.size == 0 or self.local_plan_y_list_fixed.size == 0 or self.segments.size == 0 or self.data.size == 0 or self.image.size == 0 or self.fudged_image.size == 0:
                return False

            if self.local_plan_x_list_fixed.shape != self.local_plan_y_list_fixed.shape:
                return False      

            if self.segments.shape != self.pd_image_size or self.image.shape != self.pd_image_size or self.fudged_image.shape != self.pd_image_size:
                return False

            if self.footprint_tmp.shape != (16, 4) or self.costmap_info_tmp.shape != (7, 1) or self.amcl_pose_tmp.shape != (4, 1) or self.tf_map_odom_tmp.shape != (7, 1) or self.tf_odom_map_tmp.shape != (7, 1) or self.odom_tmp.shape != (6, 1):
                return False

            if self.data.shape[0] != len(np.unique(self.segments)):
                return False
        except:
            return False

        return True

    # save data for local planner
    def saveImageDataForLocalPlanner(self):
        # Saving data to .csv files for C++ node - local navigation planner
        
        try:
            # Save footprint instance to a file
            self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_tmp_fixed.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)

            # Save costmap_info instance to file
            self.costmap_info_tmp = self.costmap_info_tmp.transpose()
            self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_tmp = self.amcl_pose_tmp.transpose()
            self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_tmp = self.tf_odom_map_tmp.transpose()
            self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_tmp = self.tf_map_odom_tmp.transpose()
            self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

            # Save odometry instance to file
            self.odom_tmp = self.odom_tmp.transpose()
            self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        except:
            return False

        return True

    # call teb
    def create_labels(self, image, fudged_image, segments, classifier_fn, batch_size=10):
        try:
            # call teb and get labels
            self.labels = []
            imgs = []
            rows = self.data
            
            segments_labels = np.unique(self.segments)

            for row in rows:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == segments_labels[z]] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    self.labels.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                self.labels.extend(preds)

            self.labels = np.array(self.labels)
        
        except:
            return False

        return True    

    # classifier function for lime image
    def classifier_fn(self, sampled_instance):
        # save perturbations for the local planner
        start = time.time()
        sampled_instance_shape_len = len(sampled_instance.shape)
        sample_size = 1 if sampled_instance_shape_len == 2 else sampled_instance.shape[0]
        print('sample_size = ', sample_size)

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
        
        time.sleep(0.35)
        end = time.time()
        print('REAL TEB TIME = ', end-start)

        #print('\nC++ node ended')

        start = time.time()
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        # save original local plan for qsr_rt
        self.local_plan_original = local_plans.loc[local_plans['ID'] == 0]
        #print('self.local_plan_original.shape = ', self.local_plan_original.shape)
        #local_plan_original.to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_original.csv', index=False)#, header=False)

        # load transformed global plan to /odom frame
        transformed_plan = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv')
        # save transformed plan for the qsr_rt
        transformed_plan.to_csv(self.dirCurr + '/' + self.dirName + '/transformed_plan.csv', index=False)#, header=False)
        end = time.time()
        print('BETWEEN TEB AND TARGET TIME = ', end-start)

        local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sample_size), columns=['deviate'])


        start = time.time()
        #transformed_plan = np.array(transformed_plan)

        # fill in deviation dataframe
        # transform transformed_plan to list
        transformed_plan_xs = []
        transformed_plan_ys = []
        for i in range(0, transformed_plan.shape[0]):
            transformed_plan_xs.append(transformed_plan.iloc[i, 0])
            transformed_plan_ys.append(transformed_plan.iloc[i, 1])
        
        for i in range(0, sample_size):
            #print('i = ', i)
            
            local_plan_xs = []
            local_plan_ys = []
            
            # transform local_plan to list
            local_plans_local = (local_plans.loc[local_plans['ID'] == i])
            #local_plans_local = np.array(local_plans_local)
            for j in range(0, local_plans_local.shape[0]):
                    local_plan_xs.append(local_plans_local.iloc[j, 0])
                    local_plan_ys.append(local_plans_local.iloc[j, 1])
            
            # find deviation as a sum of minimal point-to-point differences
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, local_plans_local.shape[0]):
                local_diffs = []
                for k in range(0, len(transformed_plan)):
                    #diff_x = (local_plans_local[j, 0] - transformed_plan[k, 0]) ** 2
                    #diff_y = (local_plans_local[j, 1] - transformed_plan[k, 1]) ** 2
                    diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
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
    def explain(self):
        # try to load lime_rt_sub data
        # if data not loaded do not explain
        start = time.time()
        if self.load_data() == False:
            print('\nData not loaded!')
            return
        end = time.time()
        print('\nDATA LOADING TIME = ', end-start)

        # turn grayscale image to rgb image
        self.image_rgb = gray2rgb(self.image * 1.0)
        # get current local costmap data
        self.localCostmapOriginX = self.costmap_info_tmp.iloc[3]
        self.localCostmapOriginY = self.costmap_info_tmp.iloc[4]
        self.localCostmapResolution = self.costmap_info_tmp.iloc[0]

        # save data for teb
        if self.saveImageDataForLocalPlanner() == False:
            #print('\nData not saved correctly!')
            return

        start = time.time()
        # call teb
        self.labels=(0,)
        self.top = self.labels
        if self.create_labels(self.image, self.fudged_image, self.segments, self.classifier_fn, batch_size=2048) == False:
            return
        end = time.time()
        print('\nCALLING TEB TIME = ', end-start)

        # find distances
        # distance_metric = 'jaccard' - alternative distance metric
        distance_metric='cosine'
        self.distances = sklearn.metrics.pairwise_distances(
            self.data,
            self.data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # Explanation variables
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'

        try:
            start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.image_rgb, self.segments)
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
            output, exp = ret_exp.get_image_and_mask(self.segments, label=0)
            end = time.time()
            print('\nGET EXP PIC TIME = ', end-start)

            output[:,:,0] = np.flip(output[:,:,0], axis=0)
            output[:,:,1] = np.flip(output[:,:,1], axis=0)
            output[:,:,2] = np.flip(output[:,:,2], axis=0)

            #print('\nexp = ', exp)

            #pd.DataFrame(output[:,:,0]).to_csv(self.dirCurr + '/' + self.dirName + '/output_B.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,1]).to_csv(self.dirCurr + '/' + self.dirName + '/output_G.csv', index=False) #, header=False)
            #pd.DataFrame(output[:,:,2]).to_csv(self.dirCurr + '/' + self.dirName + '/output_R.csv', index=False) #, header=False)

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
                exp_img_start = time.time()
                # publish explanation image
                #output = output[:, :, [2, 1, 0]].astype(np.uint8)
                #output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                output = output[:, :, [2, 1, 0]]#.astype(np.uint8)
                output_cv = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                self.pub_exp_image.publish(output_cv)

                exp_img_end = time.time()
                print('\nexp_img_time = ', exp_img_end - exp_img_start)

            if self.publish_pointcloud:
                # publish explanation layer
                points_start = time.time()
                z = 0.0
                a = 255                    
                points = []
                output = output[:, :, [2, 1, 0]].astype(np.uint8)
                for i in range(0, self.costmap_size):
                    for j in range(0, self.costmap_size):
                        x = self.localCostmapOriginX + i * self.localCostmapResolution
                        y = self.localCostmapOriginY + j * self.localCostmapResolution
                        r = int(output[j, i, 2])
                        g = int(output[j, i, 1])
                        b = int(output[j, i, 0])
                        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                        pt = [x, y, z, rgb]
                        points.append(pt)
                points_end = time.time()
                print('\npointcloud_points_time = ', points_end - points_start)
                self.header.frame_id = 'odom'
                pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
                pc2.header.stamp = rospy.Time.now()
                self.pub_exp_pointcloud.publish(pc2)
                
        except Exception as e:
            print('Exception: ', e)
            #print('Exception - explanation is skipped!!!')
            return
    
    # initialize publishers
    def main_(self):
        self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
        self.br = CvBridge()

        self.pub_exp_pointcloud = rospy.Publisher("/lime_explanation_layer", PointCloud2)

        # N_segments * (label, coefficient) + (original_deviation)
        self.pub_lime = rospy.Publisher("/lime_rt_exp", Float32MultiArray, queue_size=10)


# ----------main-----------
# main function
# define lime_rt_pub object
lime_rt_pub_obj = lime_rt_pub()
# call main to initialize publishers
lime_rt_pub_obj.main_()

# Initialize the ROS Node named 'lime_rt_pub', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_pub', anonymous=True)

# declare transformation buffer
lime_rt_pub_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_pub_obj.tf_listener = tf2_ros.TransformListener(lime_rt_pub_obj.tfBuffer)

#rate = rospy.Rate(1)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #rate.sleep()
    start = time.time()
    lime_rt_pub_obj.explain()
    end = time.time()
    with open('TIME.csv','a') as file:
        file.write(str(1000 * (end-start)))
        file.write('\n')