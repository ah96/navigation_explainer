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
import time
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion, Pose, Twist, Point
import cv2

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
        self.val_high = 1.0
        self.free_space_shade = 0.7

    def get_image_and_mask(self, segments, label):

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        exp = self.local_exp[label]

        temp = np.zeros(self.image.shape)

        w_sum_abs = 0.0
        w_s_abs = []
        for f, w in exp:
            w_sum_abs += abs(w)
            w_s_abs.append(abs(w))
        max_w_abs = max(w_s_abs)
        if max_w_abs == 0:
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
                            temp[segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                        else:
                            temp[segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs 
                        temp[segments == f, 2] = 0.0
                    elif c == 0:
                        temp[segments == f, 0] = 0.0
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                        else:
                            temp[segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs 
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                                        
        return temp, exp


class lime_rt_pub(object):
    # constructor
    def __init__(self):
        # global counter
        self.counter_global = 0

        # publish bool variables
        self.publish_explanation_coeffs_bool = True  
        self.publish_explanation_image_bool = True
        self.publish_pointcloud_bool = True

        # plotting
        self.plot_explanation_bool = False
        self.plot_perturbations_bool = False 
        self.plot_classifier_bool = False

        # directories        
        self.dir_curr = os.getcwd()
        self.dir_name = 'lime_rt_data'

        self.dir_main = 'explanation_data'
        try:
            os.mkdir(self.dir_main)
        except FileExistsError:
            pass
        
        self.dirData = self.dir_main + '/' + self.dir_name
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_explanation_bool == True:
            self.explanation_dir = self.dir_main + '/explanation_images'
            try:
                os.mkdir(self.explanation_dir)
            except FileExistsError:
                pass

        if self.plot_perturbations_bool == True: 
            self.perturbation_dir = self.dir_main + '/perturbation_images'
            try:
                os.mkdir(self.perturbation_dir)
            except FileExistsError:
                pass
        
        if self.plot_classifier_bool == True:
            self.classifier_dir = self.dir_main + '/classifier_images'
            try:
                os.mkdir(self.classifier_dir)
            except FileExistsError:
                pass

        # directory variables
        self.file_path_footprint = self.dir_main + '/' + self.dir_name + '/footprint.csv'
        self.file_path_global_plan = self.dir_main + '/' + self.dir_name + '/global_plan.csv'
        self.file_path_local_costmap_info = self.dir_main + '/' + self.dir_name + '/local_costmap_info.csv'
        self.file_path_amcl_pose = self.dir_main + '/' + self.dir_name + '/amcl_pose.csv' 
        self.file_path_tf_odom_map = self.dir_main + '/' + self.dir_name + '/tf_odom_map.csv'
        self.file_path_tf_map_odom = self.dir_main + '/' + self.dir_name + '/tf_map_odom.csv'
        self.file_path_odom = self.dir_main + '/' + self.dir_name + '/odom.csv'
        self.file_path_local_plan = self.dir_main + '/' + self.dir_name + '/local_plan.csv'
        self.file_path_segments = self.dir_main + '/' + self.dir_name + '/segments.csv'
        self.file_path_data = self.dir_main + '/' + self.dir_name + '/data.csv'
        self.file_path_local_costmap = self.dir_main + '/' + self.dir_name + '/local_costmap.csv'
        self.file_path_fudged_image = self.dir_main + '/' + self.dir_name + '/fudged_image.csv'

        # plans' variables
        self.local_plan = []

        # costmap variables
        self.labels = np.array([]) 
        self.distances = np.array([])
        self.local_costmap_size = 160

        # deviation
        self.original_deviation = 0

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
        
        if self.publish_pointcloud_bool:
            # point_cloud variables
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

            # header
            self.header = Header()
            
    # load lime_rt_sub data
    def load_data(self):
        print_data = False
        try:
            start = time.time()

            # footprint
            if os.path.getsize(self.file_path_footprint) == 0 or os.path.exists(self.file_path_footprint) == False:
                return False
            self.footprint = pd.read_csv(self.dir_curr + '/' + self.file_path_footprint)
            if print_data == True:
                print('self.footprint.shape = ', self.footprint.shape)
            
            # global plan
            if os.path.getsize(self.file_path_global_plan) == 0 or os.path.exists(self.file_path_global_plan) == False:
                return False
            self.global_plan = pd.read_csv(self.dir_curr + '/' + self.file_path_global_plan)
            if print_data == True:
                print('self.global_plan.shape = ', self.global_plan.shape)
            
            # costmap info
            if os.path.getsize(self.file_path_local_costmap_info) == 0 or os.path.exists(self.file_path_local_costmap_info) == False:
                return False
            self.local_costmap_info = pd.read_csv(self.dir_curr + '/' + self.file_path_local_costmap_info)
            if print_data == True:
                print('self.local_costmap_info.shape = ', self.local_costmap_info.shape)
            
            # amcl pose
            if os.path.getsize(self.file_path_amcl_pose) == 0 or os.path.exists(self.file_path_amcl_pose) == False:
                return False
            self.amcl_pose = pd.read_csv(self.dir_curr + '/' + self.file_path_amcl_pose)
            if print_data == True:
                print('self.amcl_pose.shape = ', self.amcl_pose.shape)
            
            # tf odom map
            if os.path.getsize(self.file_path_tf_odom_map) == 0 or os.path.exists(self.file_path_tf_odom_map) == False:
                return False
            self.tf_odom_map = pd.read_csv(self.dir_curr + '/' + self.file_path_tf_odom_map)
            if print_data == True:
                print('self.tf_odom_map.shape = ', self.tf_odom_map.shape)
            
            # tf map odom
            if os.path.getsize(self.file_path_tf_map_odom) == 0 or os.path.exists(self.file_path_tf_map_odom) == False:
                return False
            self.tf_map_odom = pd.read_csv(self.dir_curr + '/' + self.file_path_tf_map_odom)
            if print_data == True:
                print('self.tf_map_odom.shape = ', self.tf_map_odom.shape)
            
            # odom
            if os.path.getsize(self.file_path_odom) == 0 or os.path.exists(self.file_path_odom) == False:
                return False
            self.odom = pd.read_csv(self.dir_curr + '/' + self.file_path_odom)
            if print_data == True:
                print('self.odom.shape = ', self.odom.shape)

            # local plan
            if os.path.getsize(self.file_path_local_plan) == 0 or os.path.exists(self.file_path_local_plan) == False:
                return False        
            self.local_plan = pd.read_csv(self.dir_curr + '/' + self.file_path_local_plan)
            if print_data == True:
                print('self.local_plan.shape = ', self.local_plan.shape)
            
            # segments
            if os.path.getsize(self.file_path_segments) == 0 or os.path.exists(self.file_path_segments) == False:
                return False       
            self.segments = np.array(pd.read_csv(self.dir_curr + '/' + self.file_path_segments))
            if print_data == True:
                print('self.segments.shape = ', self.segments.shape)

            # perturbation data   
            if os.path.getsize(self.file_path_data) == 0 or os.path.exists(self.file_path_data) == False:
                return False       
            self.data = np.array(pd.read_csv(self.dir_curr + '/' + self.file_path_data))
            if print_data == True:
                print('self.data.shape = ', self.data.shape)
            
            # image -- local costmap
            if os.path.getsize(self.file_path_local_costmap) == 0 or os.path.exists(self.file_path_local_costmap) == False:
                return False        
            self.image = np.array(pd.read_csv(self.dir_curr + '/' + self.file_path_local_costmap))
            if print_data == True:
                print('self.image.shape = ', self.image.shape)
            
            # fudged image
            if os.path.getsize(self.file_path_fudged_image) == 0 or os.path.exists(self.file_path_fudged_image) == False:
                return False        
            self.fudged_image = np.array(pd.read_csv(self.dir_curr + '/' + self.file_path_fudged_image))
            if print_data == True:
                print('self.fudged_image.shape = ', self.fudged_image.shape)

            # if anything is empty or not of the right shape do not explain
            if self.global_plan.empty or self.local_costmap_info.empty or self.amcl_pose.empty or self.tf_odom_map.empty or self.tf_map_odom.empty or self.odom.empty or self.local_plan.empty or self.footprint.empty: 
                return False

            if self.segments.size == 0 or self.data.size == 0 or self.image.size == 0 or self.fudged_image.size == 0:
                return False

            end = time.time()
            print('\n\nDATA LOADING RUNTIME = ', round(end-start,3))
        
        except Exception as e:
            print('exception = ', e)
            print('\nData not loaded!')
            return False

        return True

    # save data for local planner
    def save_data_for_local_planner(self):
        # Saving data to .csv files for C++ node - local navigation planner   
        try:
            start = time.time()
            
            # Save footprint instance to a file
            self.footprint.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False, header=False)
            self.global_plan.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save costmap_info instance to file
            self.local_costmap_info = self.local_costmap_info.transpose()
            self.local_costmap_info.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

            # Save amcl_pose instance to file
            self.amcl_pose = self.amcl_pose.transpose()
            self.amcl_pose.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map = self.tf_odom_map.transpose()
            self.tf_odom_map.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom = self.tf_map_odom.transpose()
            self.tf_map_odom.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

            # Save odometry instance to file
            self.odom = self.odom.transpose()
            self.odom.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

            end = time.time()
            print('DATA LOCAL PLANNER SAVING RUNTIME = ', round(end-start,3))

        except Exception as e:
            print('exception = ', e)
            print('\nData for local planner not saved correctly!')
            return False

        return True

    # call local planner
    def create_labels(self, image, fudged_image, segments, classifier_fn):
        try:
            start = time.time()

            if self.plot_perturbations_bool == True:
                dirCurr = self.perturbation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass

            # call teb and get labels
            self.labels = []
            imgs = []
            rows = self.data
            #print('rows.shape = ', rows.shape)
            #print('rows = ', rows)

            #print('batch_size = ', batch_size)

            segments_labels = np.unique(self.segments)
            #print('segments_labels = ', segments_labels)

            ctr = 0
            for row in rows:
                #print('row = ', row)
                temp = copy.deepcopy(image)
                #print('temp.shape = ', temp.shape)
                zeros = np.where(row == 0)[0]
                #print('zeros.shape = ', zeros.shape)
                mask = np.zeros(segments.shape).astype(bool)
                #print('mask.shape = ', mask.shape)
                for z in zeros:
                    mask[segments == segments_labels[z]] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)

                # plot perturbation
                if self.plot_perturbations_bool:
                    #fig = plt.figure(frameon=False)
                    #w = 1.6 * 3
                    #h = 1.6 * 3
                    #fig.set_size_inches(w, h)
                    ax = plt.Axes(self.fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    self.fig.add_axes(ax)
                    img = np.flipud(imgs[-1])
                    ax.imshow(img.astype('float64'), aspect='auto')
                    self.fig.savefig(dirCurr + '/perturbation_' + str(ctr) + '.png', transparent=False)
                    self.fig.clf()
                    pd.DataFrame(img).to_csv(dirCurr + '/perturbation_' + str(ctr) + '.csv', index=False)#, header=False)
                    ctr += 1

                #if len(imgs) == batch_size:
                #    preds = classifier_fn(np.array(imgs))
                #    self.labels.extend(preds)
                #    imgs = []
            
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                self.labels.extend(preds)

            self.labels = np.array(self.labels)

            end = time.time()
            print('LABELS CREATION RUNTIME = ', round(end-start,3))
        
        except Exception as e:
            print('exception = ', e)
            print('\nLabels not created!')
            return False

        return True    

    # plot_explanation
    def plot_explanation(self):
        # tf vars
        #self.t_om = np.asarray([self.tf_odom_map[0],self.tf_odom_map[1],self.tf_odom_map[2]])
        #self.r_om = R.from_quat([self.tf_odom_map[3],self.tf_odom_map[4],self.tf_odom_map[5],self.tf_odom_map[6]])
        #self.r_om = np.asarray(self.r_om.as_matrix())

        #self.t_mo = np.asarray([self.tf_map_odom[0],self.tf_map_odom[1],self.tf_map_odom[2]])
        #self.r_mo = R.from_quat([self.tf_map_odom[3],self.tf_map_odom[4],self.tf_map_odom[5],self.tf_map_odom[6]])
        #self.r_mo = np.asarray(self.r_mo.as_matrix())    

        try:
            robot_x = self.amcl_pose.iloc[0,0]
            robot_y = self.amcl_pose.iloc[0,1]

            robot_x_idx = int((robot_x - self.local_costmap_origin_x) / self.local_costmap_resolution)
            robot_y_idx = self.local_costmap_size - 1 - int((robot_y - self.local_costmap_origin_y) / self.local_costmap_resolution)

            robot_orient_z = self.amcl_pose.iloc[0,2]
            robot_orient_w = self.amcl_pose.iloc[0,3]
            # calculate Euler angles based on orientation quaternion
            [robot_yaw, robot_pitch, robot_roll] = quaternion_to_euler(0.0, 0.0, robot_orient_z, robot_orient_w)
            
            # find yaw angles projections on x and y axes and save them to class variables
            robot_yaw_x = math.cos(robot_yaw)
            robot_yaw_y = math.sin(robot_yaw)
        
            global_plan = np.array(self.global_plan)
            global_plan_xs_idx = []
            global_plan_ys_idx = []
            for i in range(0, global_plan.shape[0]):
                x_temp = int((global_plan[i, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
                y_temp = int((global_plan[i, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

                if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                    global_plan_xs_idx.append(x_temp)
                    global_plan_ys_idx.append(self.local_costmap_size - 1 - y_temp)

            local_plan = np.array(self.local_plan)
            local_plan_xs_idx = []
            local_plan_ys_idx = []
            for i in range(0, local_plan.shape[0]):
                x_temp = int((local_plan[i, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
                y_temp = int((local_plan[i, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

                if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                    local_plan_xs_idx.append(x_temp)
                    local_plan_ys_idx.append(self.local_costmap_size - 1 - y_temp)
            

            dirCurr = self.explanation_dir + '/' + str(self.counter_global)
            try:
                os.mkdir(dirCurr)
            except FileExistsError:
                pass

            #fig = plt.figure(frameon=False)
            #w = 1.6 * 3
            #h = 1.6 * 3
            #fig.set_size_inches(w, h)
            ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            self.fig.add_axes(ax)

            ax.scatter(global_plan_xs_idx, global_plan_ys_idx, c='blue', marker='x')
            ax.scatter(local_plan_xs_idx, local_plan_ys_idx, c='yellow', marker='o')
            ax.scatter([robot_x_idx], [robot_y_idx], c='white', marker='o')
            ax.text(robot_x_idx, robot_y_idx, 'robot', c='white')
            ax.quiver(robot_x_idx, robot_y_idx, robot_yaw_x, robot_yaw_y, color='white')
            ax.imshow(self.output.astype('float64'), aspect='auto')

            centroids_for_plot = []
            lc_regions = regionprops(self.segments.astype(int))
            for lc_region in lc_regions:
                v = lc_region.label
                cy, cx = lc_region.centroid
                centroids_for_plot.append([v,cx,self.local_costmap_size - 1 - cy])

            for i in range(0, len(centroids_for_plot)):
                ax.scatter(centroids_for_plot[i][1], centroids_for_plot[i][2], c='white', marker='o')   
                ax.text(centroids_for_plot[i][1], centroids_for_plot[i][2], centroids_for_plot[i][0], c='white')

            self.fig.savefig(dirCurr + '/explanation.png', transparent=False)
            self.fig.clf()

            pd.DataFrame(global_plan_xs_idx).to_csv(dirCurr + '/global_plan_xs_idx.csv', index=False)#, header=False)
            pd.DataFrame(global_plan_ys_idx).to_csv(dirCurr + '/global_plan_ys_idx.csv', index=False)#, header=False)
            pd.DataFrame(local_plan_xs_idx).to_csv(dirCurr + '/local_plan_xs_idx.csv', index=False)#, header=False)
            pd.DataFrame(local_plan_ys_idx).to_csv(dirCurr + '/local_plan_ys_idx.csv', index=False)#, header=False)
            pd.DataFrame([robot_x_idx, robot_y_idx]).to_csv(dirCurr + '/robot_idx.csv', index=False)#, header=False)
            pd.DataFrame(centroids_for_plot).to_csv(dirCurr + '/centroids_for_plot.csv', index=False)#, header=False)
            pd.DataFrame(self.output[:,:,0]).to_csv(dirCurr + '/output_B.csv', index=False) #, header=False)
            pd.DataFrame(self.output[:,:,1]).to_csv(dirCurr + '/output_G.csv', index=False) #, header=False)
            pd.DataFrame(self.output[:,:,2]).to_csv(dirCurr + '/output_R.csv', index=False) #, header=False)


        except Exception as e:
            print('exception = ', e)
            print('\nError while plotting explanation!')
            return False

        return True 

    # create distances between the instance of interest and perturbations
    def create_distances(self):
        start = time.time()
        # find distances
        # distance_metric = 'jaccard' - alternative distance metric
        distance_metric='cosine'
        self.distances = sklearn.metrics.pairwise_distances(
            self.data,
            self.data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        end = time.time()
        print('DISTANCES CREATION RUNTIME = ', round(end-start,3))

    # plot local planner outputs for every perturbation
    def plot_classifier(self, transformed_plan, local_plans, sampled_instances, sample_size):
        start = time.time()

        dirCurr = self.classifier_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        robot_x = self.amcl_pose.iloc[0,0]
        robot_y = self.amcl_pose.iloc[0,1]

        robot_x_idx = int((robot_x - self.local_costmap_origin_x) / self.local_costmap_resolution)
        robot_y_idx = self.local_costmap_size - 1 - int((robot_y - self.local_costmap_origin_y) / self.local_costmap_resolution)

        robot_orient_z = self.amcl_pose.iloc[0,2]
        robot_orient_w = self.amcl_pose.iloc[0,3]
        # calculate Euler angles based on orientation quaternion
        [robot_yaw, robot_pitch, robot_roll] = quaternion_to_euler(0.0, 0.0, robot_orient_z, robot_orient_w)
        
        # find yaw angles projections on x and y axes and save them to class variables
        robot_yaw_x = math.cos(robot_yaw)
        robot_yaw_y = math.sin(robot_yaw)

        transformed_plan_xs_idx = []
        transformed_plan_ys_idx = []
        for i in range(0, transformed_plan.shape[0]):
            x_temp = int((transformed_plan[i, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
            y_temp = int((transformed_plan[i, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

            if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                transformed_plan_xs_idx.append(x_temp)
                transformed_plan_ys_idx.append(self.local_costmap_size - 1 - y_temp)

        pd.DataFrame(transformed_plan_xs_idx).to_csv(dirCurr + '/transformed_plan_xs_idx.csv', index=False)#, header=False)
        pd.DataFrame(transformed_plan_ys_idx).to_csv(dirCurr + '/transformed_plan_ys_idx.csv', index=False)#, header=False)
        pd.DataFrame([robot_x_idx, robot_y_idx]).to_csv(dirCurr + '/robot_idx.csv', index=False)#, header=False)
        
        for ctr in range(0, sample_size):
            # indices of local plan's poses in local costmap
            local_plan_xs_idx = []
            local_plan_ys_idx = []

            # find if there is local plan
            local_plan = local_plans.loc[local_plans['ID'] == ctr]
            for j in range(0, local_plan.shape[0]):
                    x_temp = int((local_plan.iloc[j, 0] - self.local_costmap_origin_x) / self.local_costmap_resolution)
                    y_temp = int((local_plan.iloc[j, 1] - self.local_costmap_origin_y) / self.local_costmap_resolution)

                    if 0 <= x_temp < self.local_costmap_size and 0 <= y_temp < self.local_costmap_size:
                        local_plan_xs_idx.append(x_temp)
                        local_plan_ys_idx.append(self.local_costmap_size - 1 - y_temp)

            #fig = plt.figure(frameon=True)
            #w = 1.6*3
            #h = 1.6*3
            #fig.set_size_inches(w, h)
            ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            self.fig.add_axes(ax)
            img = np.flipud(sampled_instances[ctr]).astype(np.uint8)
            ax.imshow(img)
            plt.scatter(transformed_plan_xs_idx, transformed_plan_ys_idx, c='blue', marker='x')
            plt.scatter(local_plan_xs_idx, local_plan_ys_idx, c='red', marker='x')
            ax.scatter([robot_x_idx], [robot_y_idx], c='white', marker='o')
            ax.text(robot_x_idx, robot_y_idx, 'robot', c='white')
            ax.quiver(robot_x_idx, robot_y_idx, robot_yaw_x, robot_yaw_y, color='white')
            self.fig.savefig(dirCurr + '/perturbation_' + str(ctr) + '.png')
            self.fig.clf()

            pd.DataFrame(img).to_csv(dirCurr + '/perturbation_' + str(ctr) + '.csv', index=False)#, header=False)
            pd.DataFrame(local_plan_xs_idx).to_csv(dirCurr + '/local_plan_xs_idx_' + str(ctr) + '.csv', index=False)#, header=False)
            pd.DataFrame(local_plan_ys_idx).to_csv(dirCurr + '/local_plan_ys_idx_' + str(ctr) + '.csv', index=False)#, header=False)
        
        end = time.time()
        print('PLOT CLASSIFIER RUNTIME = ', round(end-start,3))

    # classifier function for lime image
    def classifier_fn(self, sampled_instances):
        # save perturbations for the local planner
        start = time.time()
        sampled_instances_shape_len = len(sampled_instances.shape)
        sample_size = 1 if sampled_instances_shape_len == 2 else sampled_instances.shape[0]
        print('sample_size = ', sample_size)

        if sampled_instances_shape_len > 3:
            temp = np.delete(sampled_instances,2,3)
            temp = np.delete(temp,1,3)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt(self.dir_curr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instances_shape_len == 3:
            temp = sampled_instances.reshape(sampled_instances.shape[0]*160,160)
            np.savetxt(self.dir_curr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instances_shape_len == 2:
            np.savetxt(self.dir_curr + 'src/teb_local_planner/src/Data/costmap_data.csv', sampled_instances, delimiter=",")
        end = time.time()
        print('classifier_fn: LOCAL PLANNER DATA PREPARATION RUNTIME = ', round(end-start,3))

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
        print('classifier_fn: REAL LOCAL PLANNER RUNTIME = ', round(end-start,3))

        #print('\nC++ node ended')

        start = time.time()
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        local_plans = pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/local_plans.csv')
        
        # load transformed global plan to /odom frame
        transformed_plan = np.array(pd.read_csv(self.dir_curr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        end = time.time()
        print('classifier_fn: RESULTS LOADING RUNTIME = ', round(end-start,3))

        # plot local planner outputs for every perturbation
        if self.plot_classifier_bool:
            self.plot_classifier(transformed_plan, local_plans, sampled_instances, sample_size)

        local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sample_size), columns=['deviate'])

        start = time.time()
        #transformed_plan = np.array(transformed_plan)

        # fill in deviation dataframe
        # transform transformed_plan to list
        transformed_plan_xs = []
        transformed_plan_ys = []
        for i in range(0, transformed_plan.shape[0]):
            transformed_plan_xs.append(transformed_plan[i, 0])
            transformed_plan_ys.append(transformed_plan[i, 1])
        
        for i in range(0, sample_size):
            #print('i = ', i)
            
            # transform the current local_plan to list
            local_plan = (local_plans.loc[local_plans['ID'] == i])
            local_plan = np.array(local_plan)
            #if i == 0:
            #    local_plan = np.array(self.local_plan)
            #else:
            #    local_plan = np.array(local_plan)
            if local_plan.shape[0] == 0:
                local_plan_deviation.iloc[i, 0] = 0.0
                continue
            local_plan_xs = []
            local_plan_ys = []
            for j in range(0, local_plan.shape[0]):
                local_plan_xs.append(local_plan[j, 0])
                local_plan_ys.append(local_plan[j, 1])
            
            # find deviation as a sum of minimal point-to-point differences
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, local_plan.shape[0]):
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
        print('classifier_fn: TARGET CALC RUNTIME = ', round(end-start,3))
        
        if self.publish_explanation_coeffs_bool:
            self.original_deviation = local_plan_deviation.iloc[0, 0]
            #print('\noriginal_deviation = ', self.original_deviation)

        cmd_vel_perturb['deviate'] = local_plan_deviation
        
        # return local_plan_deviation
        return np.array(cmd_vel_perturb.iloc[:, 3:])

    # explain function
    def explain(self):
        # try to load lime_rt_sub data
        # if data not loaded do not explain
        if self.load_data() == False:
            return
        
        # save data for the local planner
        if self.save_data_for_local_planner() == False:
            return

        # turn grayscale image to rgb image
        self.image_rgb = gray2rgb(self.image * 1.0)
        # get current local costmap data
        self.local_costmap_origin_x = self.local_costmap_info.iloc[0,3]
        self.local_costmap_origin_y = self.local_costmap_info.iloc[0,4]
        self.local_costmap_resolution = self.local_costmap_info.iloc[0,0]
        self.local_costmap_size = self.local_costmap_info.iloc[0,1]

        # call the local planner
        self.labels=(0,)
        self.top = self.labels
        if self.create_labels(self.image, self.fudged_image, self.segments, self.classifier_fn) == False:
            return

        # create distances between the instance of interest and perturbations
        self.create_distances()

        # pure explanation part
        try:
            # Explanation variables
            top_labels=1 #10
            model_regressor = None
            num_features=100000
            feature_selection='auto'

            # train interpretable model
            start = time.time()
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
            print('MODEL FITTING RUNTIME = ', round(end-start,3))

            start = time.time()
            # get explanation image
            self.output, exp = ret_exp.get_image_and_mask(self.segments, label=0)
            end = time.time()
            print('GET EXPLANATION IMAGE RUNTIME = ', round(end-start,3))
            print('exp = ', exp)

            self.output[:,:,0] = np.flip(self.output[:,:,0], axis=0)
            self.output[:,:,1] = np.flip(self.output[:,:,1], axis=0)
            self.output[:,:,2] = np.flip(self.output[:,:,2], axis=0)

            # plot explanation
            if self.plot_explanation_bool == True:
                self.plot_explanation()

            self.counter_global+=1    

            if self.publish_explanation_coeffs_bool:
                # publish explanation coefficients
                exp_with_centroids = Float32MultiArray()
                segs_unique = np.unique(self.segments)
                for k in range(0, len(exp)):
                    exp_with_centroids.data.append(segs_unique[exp[k][0]])
                    exp_with_centroids.data.append(exp[k][1]) 
                exp_with_centroids.data.append(self.original_deviation) # append original deviation as the last element
                self.pub_lime.publish(exp_with_centroids) # N_segments * (label, coefficient) + (original_deviation)

            if self.publish_explanation_image_bool:
                exp_img_start = time.time()
                
                output = self.output[:, :, [2, 1, 0]] * 255.0 #.astype(np.uint8)
                output_msg = self.br.cv2_to_imgmsg(output.astype(np.uint8)) #,encoding="rgb8: CV_8UC3") - encoding not supported in Python3
                
                self.pub_exp_image.publish(output_msg)

                exp_img_end = time.time()
                print('PUBLISH EXPLANATION IMAGE RUNTIME = ', round(exp_img_end - exp_img_start,3))

            if self.publish_pointcloud_bool:
                # publish explanation layer
                points_start = time.time()
                z = 0.0
                a = 255                    
                points = []

                # flip the rgb image up-down
                output = cv2.flip(self.output, 0)
                #output = np.zeros(self.output.shape)
                #print(output[:,:,0].shape)
                #output[:,:,0] = np.flipud(self.output[:,:,0])
                #output[:,:,1] = np.flipud(self.output[:,:,1])
                #output[:,:,2] = np.flipud(self.output[:,:,2])

                output = output[:, :, [2, 1, 0]] * 255.0
                output = output.astype(np.uint8)
                for i in range(0, int(self.local_costmap_size)):
                    for j in range(0, int(self.local_costmap_size)):
                        x = self.local_costmap_origin_x + i * self.local_costmap_resolution
                        y = self.local_costmap_origin_y + j * self.local_costmap_resolution
                        r = int(output[j, i, 2])
                        g = int(output[j, i, 1])
                        b = int(output[j, i, 0])
                        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                        pt = [x, y, z, rgb]
                        points.append(pt)
                points_end = time.time()
                print('POINT CLOUD RUNTIME = ', round(points_end - points_start,3))
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
        if self.publish_explanation_image_bool == True:
            self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
            self.br = CvBridge()

        if self.publish_pointcloud_bool == True:
            self.pub_exp_pointcloud = rospy.Publisher("/lime_explanation_layer", PointCloud2)

        if self.publish_explanation_coeffs_bool == True:
            # N_segments * (label, coefficient) + (original_deviation)
            self.pub_lime = rospy.Publisher("/lime_rt_exp", Float32MultiArray, queue_size=10)

        if self.plot_explanation_bool==True or self.plot_classifier_bool==True or self.plot_perturbations_bool==True:
            self.fig = plt.figure(frameon=False)
            self.w = 1.6 * 3
            self.h = 1.6 * 3
            self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)


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

dir_curr = os.getcwd()
dir_main = 'explanation_data'

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #rate.sleep()
    start = time.time()
    lime_rt_pub_obj.explain()
    end = time.time()
    with open(dir_curr + '/' + dir_main + '/runtime.csv', 'a') as file:
        file.write(str(round(end-start,3)))
        file.write('\n')

