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
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
import scipy as sp
import tf2_ros
from skimage.measure import regionprops
import time
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import itertools
            
# global variables
PI = math.pi

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
    def __init__(self, image, segments, object_affordance_pairs, ontology):
        self.image = image
        self.segments = segments
        self.object_affordance_pairs = object_affordance_pairs
        self.ontology = ontology
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

        #print('get image and mask starting')
        #print('self.object_affordance_pairs: ', self.object_affordance_pairs)

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
            #temp[self.image == 0] = self.free_space_shade
            #temp[self.image != 0] = 0.0
            #return temp, exp

        segments_labels = np.unique(self.segments)
        #print('segments_labels: ', segments_labels)

        w_s = [0]*len(exp)
        f_s = [0]*len(exp)
        imgs = [np.zeros(self.image.shape)]*len(exp)
        rgb_values = []
        for f, w in exp:
            #print('(f, w): ', (f, w))
            #print(self.object_affordance_pairs[f][1] + '_' + self.object_affordance_pairs[f][2] + ' has weight ' + str(w))
            w_s[f] = w

            temp = np.zeros(self.image.shape)
            v = self.object_affordance_pairs[f][0]

            # color free space with gray
            temp[self.segments == 0, 0] = self.free_space_shade
            temp[self.segments == 0, 1] = self.free_space_shade
            temp[self.segments == 0, 2] = self.free_space_shade

            # color the obstacle-affordance pair with green or red
            if w > 0:
                if self.use_maximum_weight:
                    temp[self.segments == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * w / max_w_abs)
                else:
                    temp[self.segments == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
                    rgb_values.append(self.val_high * w / w_sum_abs)
            elif w < 0:
                if self.use_maximum_weight:
                    temp[self.segments == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * abs(w) / max_w_abs)
                else:
                    temp[self.segments == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
                    rgb_values.append(self.val_high * abs(w) / w_sum_abs)

            imgs[f] = temp    

        #print('weights = ', w_s)
        #print('get image and mask ending')

        return imgs, exp, w_s, rgb_values

# lime_rt publisher class
class lime_rt_pub(object):
    # Constructor
    def __init__(self):
        self.simulation = True

        self.counter_global = 0

        # plot bool vars
        self.plot_perturbations_bool = False 
        self.plot_classification_bool = True
        self.plot_explanation_bool = False

        # publish bool vars
        self.publish_explanation_coeffs_bool = False  
        self.publish_explanation_image_bool = False

        self.hard_obstacle = 99
        
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

        if self.plot_explanation_bool == True:
            self.explanation_dir = self.dirMain + '/explanation_images'
            try:
                os.mkdir(self.explanation_dir)
            except FileExistsError:
                pass

        if self.plot_perturbations_bool == True: 
            self.perturbation_dir = self.dirMain + '/perturbation_images'
            try:
                os.mkdir(self.perturbation_dir)
            except FileExistsError:
                pass
        
        if self.plot_classification_bool == True:
            self.classifier_dir = self.dirMain + '/classifier_images'
            try:
                os.mkdir(self.classifier_dir)
            except FileExistsError:
                pass

        # directory variables
        self.file_path_footprint = self.dirData + '/footprint_tmp.csv'
        self.file_path_gp = self.dirData + '/global_plan_tmp.csv'
        self.file_path_costmap_info = self.dirData + '/costmap_info_tmp.csv'
        self.file_path_amcl = self.dirData + '/amcl_pose_tmp.csv' 
        self.file_path_tf_om = self.dirData + '/tf_odom_map_tmp.csv'
        self.file_path_tf_mo = self.dirData + '/tf_map_odom_tmp.csv'
        self.file_path_odom = self.dirData + '/odom_tmp.csv'
        self.file_path_lp_x = self.dirData + '/local_plan_x_list.csv'
        self.file_path_lp_y = self.dirData + '/local_plan_y_list.csv'
        self.file_path_lp = self.dirData + '/local_plan_tmp.csv'
        self.file_path_segs = self.dirData + '/segments.csv'
        self.file_path_costmap = self.dirData + '/image.csv'

        # plans' variables
        self.global_plan_original_xs = [] 
        self.global_plan_original_ys = []
        self.global_plan_original = []
        self.local_plan_xs_original = []
        self.local_plan_ys_original = []
        self.local_plan_original = []

        # local map variables
        self.labels = np.array([]) 
        self.distances = np.array([])
        self.local_map_size = 160
   
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
        
        # header
        self.header = Header()

    # initialize publishers
    def main_(self):
        if self.publish_explanation_image_bool:
            self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
            self.br = CvBridge()

        if self.publish_explanation_coeffs_bool:
            # N_segments * (label, coefficient) + (original_deviation)
            self.pub_lime = rospy.Publisher("/lime_rt_exp", Float32MultiArray, queue_size=10)

        if self.publish_pointcloud)bool == True:
            self.pub_exp_pointcloud = rospy.Publisher("/lime_explanation_layer", PointCloud2)

        if self.plot_explanation_bool==True or self.plot_classifier_bool==True or self.plot_perturbations_bool==True:
            self.fig = plt.figure(frameon=False)
            self.w = 1.6 * 3
            self.h = 1.6 * 3
            self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

    # flip matrix horizontally or vertically
    def matrixflip(self,m,d):
        if d=='h':
            tempm = np.fliplr(m)
            return(tempm)
        elif d=='v':
            tempm = np.flipud(m)
            return(tempm)

    # load lime_rt_sub data
    def load_data(self):
        print_data = False
        try:
            # footprint
            if os.path.getsize(self.file_path_footprint) == 0 or os.path.exists(self.file_path_footprint) == False:
                return False
            self.footprint_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/footprint_tmp.csv')
            if print_data == True:
                print('self.footprint_tmp.shape = ', self.footprint_tmp.shape)
            
            # global plan
            if os.path.getsize(self.file_path_gp) == 0 or os.path.exists(self.file_path_gp) == False:
                return False
            self.global_plan_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/global_plan_tmp.csv')
            if print_data == True:
                print('self.global_plan_tmp.shape = ', self.global_plan_tmp.shape)
            self.plan_tmp = self.global_plan_tmp

            # costmap info
            if os.path.getsize(self.file_path_costmap_info) == 0 or os.path.exists(self.file_path_costmap_info) == False:
                return False
            self.costmap_info_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/costmap_info_tmp.csv')
            if print_data == True:
                print('self.costmap_info_tmp.shape = ', self.costmap_info_tmp.shape)

            # global (amcl) pose
            if os.path.getsize(self.file_path_amcl) == 0 or os.path.exists(self.file_path_amcl) == False:
                return False
            self.amcl_pose_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose_tmp.csv')
            if print_data == True:
                print('self.amcl_pose_tmp.shape = ', self.amcl_pose_tmp.shape)

            # transformation between odom and map
            if os.path.getsize(self.file_path_tf_om) == 0 or os.path.exists(self.file_path_tf_om) == False:
                return False
            self.tf_odom_map_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map_tmp.csv')
            tf_odom_map_tmp = np.array(self.tf_odom_map_tmp)
            if print_data == True:
                print('self.tf_odom_map_tmp.shape = ', self.tf_odom_map_tmp.shape)
            self.t_om = np.asarray([tf_odom_map_tmp[0][0],tf_odom_map_tmp[1][0],tf_odom_map_tmp[2][0]])
            self.r_om = R.from_quat([tf_odom_map_tmp[3][0],tf_odom_map_tmp[4][0],tf_odom_map_tmp[5][0],tf_odom_map_tmp[6][0]])
            self.r_om = np.asarray(self.r_om.as_matrix())    

            # transformation between map and odom
            if os.path.getsize(self.file_path_tf_mo) == 0 or os.path.exists(self.file_path_tf_mo) == False:
                return False
            self.tf_map_odom_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom_tmp.csv')
            tf_map_odom_tmp = np.array(self.tf_map_odom_tmp)
            if print_data == True:
                print('self.tf_map_odom_tmp.shape = ', self.tf_map_odom_tmp.shape)
            self.t_mo = np.asarray([tf_map_odom_tmp[0][0],tf_map_odom_tmp[1][0],tf_map_odom_tmp[2][0]])
            self.r_mo = R.from_quat([tf_map_odom_tmp[3][0],tf_map_odom_tmp[4][0],tf_map_odom_tmp[5][0],tf_map_odom_tmp[6][0]])
            self.r_mo = np.asarray(self.r_mo.as_matrix())

            # odometry pose
            if os.path.getsize(self.file_path_odom) == 0 or os.path.exists(self.file_path_odom) == False:
                return False
            self.odom_tmp = pd.read_csv(self.dirCurr + '/' + self.dirData + '/odom_tmp.csv')
            if print_data == True:
                print('self.odom_tmp.shape = ', self.odom_tmp.shape)

            # local plan
            if os.path.getsize(self.file_path_lp_x) == 0 or os.path.exists(self.file_path_lp_x) == False:
                pass
            else:
                self.local_plan_xs_original = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_plan_x_list.csv'))
                if print_data == True:
                    print('self.local_plan_xs_original.shape = ', self.local_plan_xs_original.shape)

            if os.path.getsize(self.file_path_lp_y) == 0 or os.path.exists(self.file_path_lp_y) == False:
                pass
            else:           
                self.local_plan_ys_original = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_plan_y_list.csv'))
                if print_data == True:
                    print('self.local_plan_ys_original.shape = ', self.local_plan_ys_original.shape)
            
            if os.path.getsize(self.file_path_lp) == 0 or os.path.exists(self.file_path_lp) == False:
                pass        
            else:
                self.local_plan_original = pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_plan_tmp.csv')
                if print_data == True:
                    print('self.local_plan_original.shape = ', self.local_plan_original.shape)

            # segments
            if os.path.getsize(self.file_path_segs) == 0 or os.path.exists(self.file_path_segs) == False:
                return False       
            self.segments = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/segments.csv'))
            if print_data == True:
                print('self.segments.shape = ', self.segments.shape)

            # local costmap
            if os.path.getsize(self.file_path_costmap) == 0 or os.path.exists(self.file_path_costmap) == False:
                return False        
            self.image = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/image.csv'))
            if print_data == True:
                print('self.image.shape = ', self.image.shape)

            # robot position and orientation in global (map) frame
            #self.robot_position_map = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/robot_position_map.csv'))
            #self.robot_orientation_map = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/robot_orientation_map.csv'))
            self.robot_position_map = np.array([self.amcl_pose_tmp.iloc[0],self.amcl_pose_tmp.iloc[1],0.0])
            self.robot_orientation_map = np.array([0.0,0.0,self.amcl_pose_tmp.iloc[2],self.amcl_pose_tmp.iloc[3]])

        except Exception as e:
            print('e: ', e)
            return False

        return True

    # save data for local planner
    def saveImageDataForLocalPlanner(self):
        # Saving data to .csv files for C++ node - local navigation planner
        
        try:
            # Save footprint instance to a file
            self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_original.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

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

    # saving important data to class variables
    def saveImportantData2ClassVars(self):
        #print(self.local_plan_original.shape)
        #print(self.local_plan_original)

        # save costmap info to class variables
        self.localCostmapOriginX = self.costmap_info_tmp.iloc[3, 0]
        #print('self.localCostmapOriginX: ', self.localCostmapOriginX)
        self.localCostmapOriginY = self.costmap_info_tmp.iloc[4, 0]
        #print('self.localCostmapOriginY: ', self.localCostmapOriginY)
        self.localCostmapResolution = self.costmap_info_tmp.iloc[0, 0]
        #print('self.localCostmapResolution: ', self.localCostmapResolution)
        self.localCostmapHeight = self.costmap_info_tmp.iloc[2, 0]
        #print('self.localCostmapHeight: ', self.localCostmapHeight)
        self.localCostmapWidth = self.costmap_info_tmp.iloc[1, 0]
        #print('self.localCostmapWidth: ', self.localCostmapWidth)

        # save robot odometry location to class variables
        self.odom_x = self.odom_tmp.iloc[0, 0]
        # print('self.odom_x: ', self.odom_x)
        self.odom_y = self.odom_tmp.iloc[1, 0]
        # print('self.odom_y: ', self.odom_y)

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        self.global_plan_original_xs = []
        self.global_plan_original_ys = []
        self.global_plan_original = []
        for i in range(0, self.global_plan_tmp.shape[0]):
            x_temp = int((self.global_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.global_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                self.global_plan_original_xs.append(x_temp)
                self.global_plan_original_ys.append(self.local_map_size - y_temp)
                self.global_plan_original.append([x_temp, y_temp])

        #'''
        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[2, 0]
        self.odom_w = self.odom_tmp.iloc[3, 0]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        #'''

    # call local planner
    def create_labels(self, classifier_fn):
        try:
            # load ontology
            # ontology is updated in the subscriber as the robot moves (in real-world case, when the new detected objects are not already in the ontology)
            self.ontology = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv'))
            
            # load object-affordance pairs
            # object-affordance pairs are updated in the subscriber as the robot is moving
            self.object_affordance_pairs = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv'))
            #print('self.object_affordance_pairs = ', self.object_affordance_pairs)
            
            # load inflated segments (changes the traditional local costmap)
            # segments and inflated segments are also updated in the subscriber as the robot is moving
            self.segments_inflated = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/segments_inflated.csv'))
            #print('self.segments_inflated.shape = ', self.segments_inflated.shape)


            # create data (encoded perturbations)
            # 2**N segments -- explore all object-affordance perturbations
            n_features = self.object_affordance_pairs.shape[0]
            num_samples = 2**n_features
            lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
            self.data = np.array(lst).reshape((num_samples, n_features))
            #print('self.data = ', self.data)
            #print('self.data.shape = ', self.data.shape)
            #print('type(self.data) = ', type(self.data))
            
            # N+1 or X=O(n) segments
            # explore arbitrary number of object-affordance perturbations with linear complexity in the numver of object-affordance pairs
            #self.num_samples = self.n_features + 1
            #lst = [[1]*self.n_features]
            #for i in range(1, self.num_samples):
            #    lst.append([1]*self.n_features)
            #    lst[i][i-1] = 0    
            #self.data = np.array(lst).reshape((self.num_samples, self.n_features))


            # create perturbation semantic maps and get labels
            self.labels = []
            imgs = []

            for i in range(0, num_samples):
                temp = copy.deepcopy(self.segments_inflated)
                temp[temp > 0] = self.hard_obstacle
                
                # find the indices of features (obj.-aff. pairs) which are in their alternative state
                zero_indices = np.where(self.data[i] == 0)[0]
                #print('zero_indices = ', zero_indices)
                
                for j in range(0, zero_indices.shape[0]):
                    # if feature has 0 in self.data it is in its alternative affordance state (currently: either not there or opened) --> original semantic map must be modified
                    idx = zero_indices[j]
                    #print('idx = ', idx)

                    # the object movability affordance is in its alternative state
                    # the object must be moved
                    if self.object_affordance_pairs[idx][2] == 'movability':

                        #print(self.object_affordance_pairs[idx])
                        temp[self.segments_inflated == self.object_affordance_pairs[idx][0]] = 0

                    # openability affordance                
                    elif self.object_affordance_pairs[idx][2] == 'openability':

                        # if the previous obj.-aff. pair was the same object with movability affordance, then do not draw it
                        # if this is the first obj.-aff. pair then this object does not have movability affordance, because it goes before other affordances
                        if j > 0:
                            idx_previous = zero_indices[j-1]
                            if self.object_affordance_pairs[idx_previous][2] == 'movability' and self.object_affordance_pairs[idx_previous][1] == self.object_affordance_pairs[idx][1]:
                                continue
                    
                        # objects are immediately inflated when opened
                        if self.object_affordance_pairs[idx][1] == 'door':
                            label = self.object_affordance_pairs[idx][0]

                            temp[self.segments_inflated == label] = 0
                            
                            # WRITE AN INDEPENDENT CODE FOR OPENING/CLOSING DOORS
                            # otvara se preko manje/krace strane - pogledaj malu svesku
                            # object"s vertices from /map frame to /odom and /lc frames
                            door_tl_map_x = self.ontology[-1][2] + 0.5*self.ontology[-1][4]
                            door_tl_map_y = self.ontology[-1][3] + 0.5*self.ontology[-1][5]
                            p_map = np.array([door_tl_map_x, door_tl_map_y, 0.0])
                            p_odom = p_map.dot(self.r_mo) + self.t_mo
                            tl_odom_x = p_odom[0]
                            tl_odom_y = p_odom[1]
                            door_tl_pixel_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                            door_tl_pixel_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                            
                            x_size = int(self.ontology[label - 1][4] / self.localCostmapResolution)
                            y_size = int(self.ontology[label - 1][5] / self.localCostmapResolution)
                            #print('(x_size, y_size) = ', (x_size, y_size))

                            #temp[door_tl_pixel_y:door_tl_pixel_y + x_size, door_tl_pixel_x:door_tl_pixel_x + y_size] = label
                            inflation_x = int((max(23, x_size) - x_size) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                            inflation_y = int((max(23, y_size) - y_size) / 2)                 
                            temp[max(0, door_tl_pixel_y-inflation_x):min(self.local_map_size-1, door_tl_pixel_y+x_size+inflation_x), max(0,door_tl_pixel_x-inflation_y):min(self.local_map_size-1, door_tl_pixel_x+y_size+inflation_y)] = self.hard_obstacle

                        # MORAS DODATI U  ONTOLOGY SA KOJE SE STRANE VRATA OD ORMARA OTVARAJU: TOP, LEFT, RIGHT, BOTTOM 
                        # I OVO IMPLEMENTIRATI U OVOM KODU
                        elif self.object_affordance_pairs[idx][1] == 'cabinet':                       
                            label = self.object_affordance_pairs[idx][0]
                            
                            # tl
                            cab_tl_map_x = self.ontology[label-1][2] - 0.5*self.ontology[label-1][4]
                            cab_tl_map_y = self.ontology[label-1][3] + 0.5*self.ontology[label-1][5]
                            p_map = np.array([cab_tl_map_x, cab_tl_map_y, 0.0])
                            p_odom = p_map.dot(self.r_mo) + self.t_mo
                            tl_odom_x = p_odom[0]
                            tl_odom_y = p_odom[1]
                            cab_tl_pixel_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                            cab_tl_pixel_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                            #print('(cab_tl_pixel_x, cab_tl_pixel_y) = ', (cab_tl_pixel_x, cab_tl_pixel_y))
                            
                            # tr
                            cab_tr_map_x = self.ontology[label-1][2] + 0.5*self.ontology[label-1][4]
                            cab_tr_map_y = self.ontology[label-1][3] + 0.5*self.ontology[label-1][5]
                            p_map = np.array([cab_tr_map_x, cab_tr_map_y, 0.0])
                            p_odom = p_map.dot(self.r_mo) + self.t_mo
                            tr_odom_x = p_odom[0]
                            tr_odom_y = p_odom[1]
                            cab_tr_pixel_x = int((tr_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                            cab_tr_pixel_y = int((tr_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
                            #print('(cab_tr_pixel_x, cab_tr_pixel_y) = ', (cab_tr_pixel_x, cab_tr_pixel_y))

                            inflation_x = int((max(23, self.ontology[label-1][4]) - self.ontology[label-1][4]) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                            inflation_y = int((max(23, self.ontology[label-1][5]) - self.ontology[label-1][5]) / 2) 
                            x_size = int(1.0 * self.ontology[label - 1][4] / self.localCostmapResolution)
                            y_size = int(1.0 * self.ontology[label - 1][5] / self.localCostmapResolution)
                            if (0 <= cab_tl_pixel_x < self.local_map_size and 0 <= cab_tl_pixel_y < self.local_map_size):                            
                                temp[max(0,cab_tl_pixel_y-inflation_y):min(self.local_map_size-1, cab_tl_pixel_y+y_size+inflation_y), max(0,cab_tl_pixel_x-inflation_x):min(self.local_map_size-1, cab_tl_pixel_x+x_size+inflation_x)] = self.hard_obstacle

                imgs.append(temp)                        


            # plot perturbations
            if self.plot_perturbations_bool and len(imgs)>0:
                dirCurr = self.perturbation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(dirCurr)
                except FileExistsError:
                    pass

                for i in range(0, len(imgs)):
                    fig = plt.figure(frameon=False)
                    w = 1.6 * 3
                    h = 1.6 * 3
                    fig.set_size_inches(w, h)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    pert_img = np.flipud(imgs[i]) 
                    ax.imshow(pert_img.astype('float64'), aspect='auto')
                    fig.savefig(dirCurr + '/perturbation_' + str(i) + '.png', transparent=False)
                    fig.clf()
                    
                    pd.DataFrame(pert_img).to_csv(dirCurr + '/perturbation_' + str(i) + '.csv', index=False)#, header=False)
            
            # call predictor and store labels
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
            x_temp = int((transformed_plan[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = self.local_map_size - int((transformed_plan[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)

        fig = plt.figure(frameon=True)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()

        self.y_odom_index = [self.local_map_size - self.y_odom_index[0]]
            
        for ctr in range(0, sample_size):
            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []

            # find if there is local plan
            local_plans_local = local_plans.loc[local_plans['ID'] == ctr]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = self.local_map_size - int((local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)


            fig.add_axes(ax)
            img = np.flipud(sampled_instance[ctr])
            ax.imshow(img.astype(np.uint8))
            plt.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='x')
            plt.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='x')
            ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')
            fig.savefig(dirCurr + '/perturbation_' + str(ctr) + '.png')
            fig.clf()

            pd.DataFrame(img).to_csv(dirCurr + '/perturbation_' + str(ctr) + '.csv', index=False)#, header=False)

    # classifier function for the explanation algorithm (LIME)
    def classifier_fn(self, sampled_instance):
        # save perturbations for the local planner
        start = time.time()
        sampled_instance_shape_len = len(sampled_instance.shape)
        #print('sampled_instance.shape = ', sampled_instance.shape)
        self.sample_size = 1 if sampled_instance_shape_len == 2 else sampled_instance.shape[0]
        #print('sample_size = ', sample_size)

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
        print('DATA PREPARATION TIME = ', end-start)

        # calling ROS C++ node
        #print('\nC++ node started')

        start = time.time()
        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        #rospy.wait_for_service("/perturb_node_image/finished")

        # kill ROS node
        #Popen(shlex.split('rosnode kill /perturb_node_image'))
        
        #time.sleep(0.45)
        
        end = time.time()
        print('TEB CALL TIME = ', end-start)

        #print('\nC++ node ended')

       
        start = time.time()
        
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')

        # load local plans - output from local planner
        self.local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('local_plans = ', local_plans)
        # save original local plan for qsr_rt
        #self.local_plan_full_perturbation = local_plans.loc[local_plans['ID'] == 0]
        #print('self.local_plan_full_perturbation.shape = ', self.local_plan_full_perturbation.shape)
        #local_plan_full_perturbation.to_csv(self.dirCurr + '/' + self.dirData + '/local_plan_full_perturbation.csv', index=False)#, header=False)

        # load transformed global plan to /odom frame
        self.transformed_plan = np.array(pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        # save transformed plan for the qsr_rt
        #transformed_plan.to_csv(self.dirCurr + '/' + self.dirData + '/transformed_plan.csv', index=False)#, header=False)

        # transform transformed_plan to list
        #self.transformed_plan_xs = []
        #self.transformed_plan_ys = []
        #for i in range(0, self.transformed_plan.shape[0]):
        #    self.transformed_plan_xs.append(self.transformed_plan.iloc[i, 0])
        #    self.transformed_plan_ys.append(self.transformed_plan.iloc[i, 1])
       
        end = time.time()
        print('CLASSIFIER_FN RESULTS LOADING TIME = ', end-start)

        self.global_plan_original = np.array(self.global_plan_original)
        self.local_plan_original = np.array(self.local_plan_original)

        if self.plot_classification_bool == True:
            self.classifier_fn_plot(self.global_plan_original, self.local_plans, sampled_instance, self.sample_size)


        local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])

        start = time.time()
        #transformed_plan = np.array(transformed_plan)

        # fill in deviation dataframe
        local_plan_local = []

        
        for i in range(0, self.sample_size):
            #print('i = ', i)
            
            #local_plan_xs = []
            #local_plan_ys = []
            
            # transform local_plan to list
            if i != self.sample_size - 1:
                local_plan_local = np.array(self.local_plans.loc[self.local_plans['ID'] == i])
            else:
                local_plan_local = self.local_plan_original
                #print('LAST LOCAL PLAN: ', local_plan_local)
                #print('LAST LOCAL PLAN.shape: ', local_plan_local.shape)

            #local_plans_local = np.array(local_plans_local)
            #for j in range(0, local_plans_local.shape[0]):
            #        local_plan_xs.append(local_plans_local.iloc[j, 0])
            #        local_plan_ys.append(local_plans_local.iloc[j, 1])

            # if no local plan is created, that means 'no' deviation
            if local_plan_local.shape[0] == 0:
                local_plan_deviation.iloc[i, 0] = 0
                continue

            # local plan connectedness
            #local_plan_connectedness = []
            
            # find deviation as a sum of minimal point-to-point differences
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, local_plan_local.shape[0]):
                local_diffs = []
                for k in range(0, self.transformed_plan.shape[0]):
                    #diff_x = (local_plans_local[j, 0] - transformed_plan[k, 0]) ** 2
                    #diff_y = (local_plans_local[j, 1] - transformed_plan[k, 1]) ** 2
                    diff_x = (local_plan_local[j][0] - self.transformed_plan[k][0]) ** 2
                    diff_y = (local_plan_local[j][1] - self.transformed_plan[k][0]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))

                # calculate local plan connectedness
                #if j+1 < local_plan_local.shape[0]:
                #    diff_x = (local_plan_local[j+1][0] - local_plan_local[j][0]) ** 2
                #    diff_y = (local_plan_local[j+1][1] - local_plan_local[j][1]) ** 2
                #    diff = math.sqrt(diff_x + diff_y)
                #    local_plan_connectedness.append(diff)

            # if local plan is not continuous --> a very big deviation
            #smallest_difference = min(local_plan_connectedness)
            #largest_difference = max(local_plan_connectedness)
            #if largest_difference >= 10*smallest_difference:
            #    local_plan_deviation.iloc[i, 0] = 500
            #    continue

            # if a local plan is normal, then calculate the deviation
            local_plan_deviation.iloc[i, 0] = sum(devs) #max(devs)
        end = time.time()
        print('TARGET CALC TIME = ', end-start)
        
        self.original_deviation = local_plan_deviation.iloc[-1, 0]
        #print('\noriginal_deviation = ', self.original_deviation)
        if self.plot_classification_bool:
            dirCurr = self.classifier_dir + '/' + str(self.counter_global)
            pd.DataFrame(local_plan_deviation).to_csv(dirCurr + '/deviations.csv', index=False, header=False)

        cmd_vel_perturb['deviate'] = local_plan_deviation
        #return local_plan_deviation
        return np.array(cmd_vel_perturb.iloc[:, 3:])

    # explain function
    def explain(self):
        self.counter_global+=1

        # try to load sub_hri data
        # if data not loaded do not explain
        explain_time_start = time.time()

        start = time.time()
        if self.load_data() == False:
            print('\nData not loaded!')
            return
        end = time.time()
        print('\nDATA LOADING TIME = ', end-start)

        # turn grayscale image to rgb image
        self.image_rgb = gray2rgb(self.image * 1.0)
        
        # saving important data to class variables
        self.saveImportantData2ClassVars()

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
        print('\nCALLING TEB TIME = ', end-start)
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

        #'''
        # Explanation variables
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'

        try:
            start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.image_rgb, self.segments, self.object_affordance_pairs, self.ontology)
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
            outputs, exp, weights, rgb_values = ret_exp.get_image_and_mask(label=0)
            end = time.time()
            print('\nGET EXP PIC TIME = ', end-start)
            print('exp: ', exp)

            centroids_for_plot = []
            lc_regions = regionprops(self.segments.astype(int))
            for lc_region in lc_regions:
                v = lc_region.label
                cy, cx = lc_region.centroid
                centroids_for_plot.append([v,cx,cy,self.ontology[v-1][1]])


            if self.plot_explanation_bool == True:
                self.dirPlotExp = self.explanation_dir + '/' + str(self.counter_global)
                try:
                    os.mkdir(self.dirPlotExp)
                except FileExistsError:
                    pass

                fig = plt.figure(frameon=False)
                w = 1.6 * 3
                h = 1.6 * 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                
                self.y_odom_index = [self.local_map_size - self.y_odom_index[0]]

                local_plan_x_list = []
                local_plan_y_list = []

                # find if there is local plan
                for j in range(0, self.local_plan_original.shape[0]):
                        x_temp = int((self.local_plan_original[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = self.local_map_size - int((self.local_plan_original[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                            local_plan_x_list.append(x_temp)
                            local_plan_y_list.append(y_temp)

                for k in range(0, len(outputs)):
                    output = outputs[k]
                    output[:,:,0] = np.flip(output[:,:,0], axis=0)
                    output[:,:,1] = np.flip(output[:,:,1], axis=0)
                    output[:,:,2] = np.flip(output[:,:,2], axis=0)

                    fig.add_axes(ax)
                    ax.scatter(self.global_plan_original_xs, self.global_plan_original_ys, c='blue', marker='x')
                    ax.scatter(local_plan_x_list, local_plan_y_list, c='yellow', marker='o')
                    ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
                    ax.text(self.x_odom_index[0], self.local_map_size - self.y_odom_index[0], 'robot', c='white')
                    ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')
                    ax.imshow(output.astype('float64'), aspect='auto')
                    for i in range(0, len(centroids_for_plot)):
                        ax.scatter(centroids_for_plot[i][1], self.local_map_size - centroids_for_plot[i][2], c='white', marker='o')   
                        ax.text(centroids_for_plot[i][1], self.local_map_size - centroids_for_plot[i][2], centroids_for_plot[i][3], c='white')
                    fig.savefig(self.dirPlotExp + '/explanation_' + self.object_affordance_pairs[k][1] + '_' + self.object_affordance_pairs[k][2] + '.png', transparent=False)
                    fig.clf()

                pd.DataFrame(weights).to_csv(self.dirPlotExp + '/weights.csv')
                pd.DataFrame(self.object_affordance_pairs).to_csv(self.dirPlotExp + '/object_affordance_pairs.csv')
                pd.DataFrame(rgb_values).to_csv(self.dirPlotExp + '/rgb_values.csv')
                    
            explain_time_end = time.time()
            with open(self.dirMain + '/explanation_time.csv','a') as file:
                file.write(str(explain_time_end-explain_time_start))
                file.write('\n')
            
    

        except Exception as e:
            print('Exception: ', e)
            #print('Exception - explanation is skipped!!!')
            return
        #'''

# ----------main-----------
# main function
# define lime_rt_pub object
lime_rt_pub_obj = lime_rt_pub()
# call main to initialize publishers
lime_rt_pub_obj.main_()

# Initialize the ROS Node named 'lime_rt_semantic_pub', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_semantic_pub', anonymous=True)

# declare transformation buffer
lime_rt_pub_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_pub_obj.tf_listener = tf2_ros.TransformListener(lime_rt_pub_obj.tfBuffer)

#rate = rospy.Rate(1)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #rate.sleep()
    lime_rt_pub_obj.explain()