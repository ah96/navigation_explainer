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
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
            
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
    def __init__(self, image, semantic_map, object_affordance_pairs, ontology):
        self.image = image
        self.semantic_map = semantic_map
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

        semantic_map_labels = np.unique(self.semantic_map)
        #print('semantic_map_labels: ', semantic_map_labels)

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
            temp[self.semantic_map == 0, 0] = self.free_space_shade
            temp[self.semantic_map == 0, 1] = self.free_space_shade
            temp[self.semantic_map == 0, 2] = self.free_space_shade

            # color the obstacle-affordance pair with green or red
            if w > 0:
                if self.use_maximum_weight:
                    temp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * w / max_w_abs)
                else:
                    temp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
                    rgb_values.append(self.val_high * w / w_sum_abs)
            elif w < 0:
                if self.use_maximum_weight:
                    temp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * abs(w) / max_w_abs)
                else:
                    temp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
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
        self.plot_classifier_bool = False
        self.plot_explanation_bool = False

        # publish bool vars
        self.publish_explanation_coeffs_bool = True  
        self.publish_explanation_image_bool = True
        self.publish_pointcloud_bool = True

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
        
        if self.plot_classifier_bool == True:
            self.classifier_dir = self.dirMain + '/classifier_images'
            try:
                os.mkdir(self.classifier_dir)
            except FileExistsError:
                pass

        # directory variables
        self.file_path_footprint = self.dirData + '/footprint.csv'
        self.file_path_gp = self.dirData + '/global_plan.csv'
        self.file_path_local_map_info = self.dirData + '/local_map_info.csv'
        self.file_path_amcl = self.dirData + '/amcl_pose.csv' 
        self.file_path_tf_om = self.dirData + '/tf_odom_map.csv'
        self.file_path_tf_mo = self.dirData + '/tf_map_odom.csv'
        self.file_path_odom = self.dirData + '/odom.csv'
        self.file_path_lp = self.dirData + '/local_plan.csv'
        self.file_path_semantic_map = self.dirData + '/semantic_map.csv'
        self.file_path_semantic_map_inflated = self.dirData + '/semantic_map_inflated.csv'
        self.file_path_local_map = self.dirData + '/local_map.csv'

        # plans' variables
        self.global_plan = []
        self.local_plan = []

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
        
        if self.publish_pointcloud_bool:
            # point_cloud variables
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

            # header
            self.header = Header()

    # initialize publishers
    def main_(self):
        if self.publish_explanation_image_bool:
            self.pub_exp_image = rospy.Publisher('/lime_explanation_image', Image, queue_size=10)
            self.br = CvBridge()

        if self.publish_explanation_coeffs_bool:
            # N_semantic_map * (label, coefficient) + (original_deviation)
            self.pub_lime = rospy.Publisher("/lime_rt_exp", Float32MultiArray, queue_size=10)

        if self.publish_pointcloud_bool == True:
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
            start = time.time()

            # footprint
            if os.path.getsize(self.file_path_footprint) == 0 or os.path.exists(self.file_path_footprint) == False:
                return False
            self.footprint = pd.read_csv(self.dirCurr + '/' + self.dirData + '/footprint.csv')
            if print_data == True:
                print('self.footprint.shape = ', self.footprint.shape)
            
            # global plan
            if os.path.getsize(self.file_path_gp) == 0 or os.path.exists(self.file_path_gp) == False:
                return False
            self.global_plan = pd.read_csv(self.dirCurr + '/' + self.dirData + '/global_plan.csv')
            if print_data == True:
                print('self.global_plan.shape = ', self.global_plan.shape)
            self.plan = self.global_plan

            # local_map info
            if os.path.getsize(self.file_path_local_map_info) == 0 or os.path.exists(self.file_path_local_map_info) == False:
                return False
            self.local_map_info = pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_map_info.csv')
            if print_data == True:
                print('self.local_map_info.shape = ', self.local_map_info.shape)

            # global (amcl) pose
            if os.path.getsize(self.file_path_amcl) == 0 or os.path.exists(self.file_path_amcl) == False:
                return False
            self.amcl_pose = pd.read_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose.csv')
            if print_data == True:
                print('self.amcl_pose.shape = ', self.amcl_pose.shape)

            # transformation between odom and map
            if os.path.getsize(self.file_path_tf_om) == 0 or os.path.exists(self.file_path_tf_om) == False:
                return False
            self.tf_odom_map = pd.read_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map.csv')
            if print_data == True:
                print('self.tf_odom_map.shape = ', self.tf_odom_map.shape)
            #tf_odom_map = np.array(self.tf_odom_map)
            #self.t_om = np.asarray([tf_odom_map[0][0],tf_odom_map[1][0],tf_odom_map[2][0]])
            #self.r_om = R.from_quat([tf_odom_map[3][0],tf_odom_map[4][0],tf_odom_map[5][0],tf_odom_map[6][0]])
            #self.r_om = np.asarray(self.r_om.as_matrix())    

            # transformation between map and odom
            if os.path.getsize(self.file_path_tf_mo) == 0 or os.path.exists(self.file_path_tf_mo) == False:
                return False
            self.tf_map_odom = pd.read_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom.csv')
            if print_data == True:
                print('self.tf_map_odom.shape = ', self.tf_map_odom.shape)
            tf_map_odom = np.array(self.tf_map_odom)
            self.t_mo = np.asarray([tf_map_odom[0][0],tf_map_odom[1][0],tf_map_odom[2][0]])
            self.r_mo = R.from_quat([tf_map_odom[3][0],tf_map_odom[4][0],tf_map_odom[5][0],tf_map_odom[6][0]])
            self.r_mo = np.asarray(self.r_mo.as_matrix())

            # odometry pose
            if os.path.getsize(self.file_path_odom) == 0 or os.path.exists(self.file_path_odom) == False:
                return False
            self.odom = pd.read_csv(self.dirCurr + '/' + self.dirData + '/odom.csv')
            if print_data == True:
                print('self.odom.shape = ', self.odom.shape)

            # local plan
            if os.path.getsize(self.file_path_lp) == 0 or os.path.exists(self.file_path_lp) == False:
                pass        
            else:
                self.local_plan = pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_plan.csv')
                if print_data == True:
                    print('self.local_plan.shape = ', self.local_plan.shape)

            # semantic_map
            if os.path.getsize(self.file_path_semantic_map) == 0 or os.path.exists(self.file_path_semantic_map) == False:
                return False       
            self.semantic_map = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/semantic_map.csv'))
            if print_data == True:
                print('self.semantic_map.shape = ', self.semantic_map.shape)

            # semantic_map_inflated
            if os.path.getsize(self.file_path_semantic_map_inflated) == 0 or os.path.exists(self.file_path_semantic_map_inflated) == False:
                return False       
            self.semantic_map_inflated = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/semantic_map_inflated.csv'))
            if print_data == True:
                print('self.semantic_map_inflated.shape = ', self.semantic_map_inflated.shape)

            # local_map
            if os.path.getsize(self.file_path_local_map) == 0 or os.path.exists(self.file_path_local_map) == False:
                return False        
            self.local_map = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/local_map.csv'))
            if print_data == True:
                print('self.local_map.shape = ', self.local_map.shape)

            # robot position and orientation in global (map,amcl) frame
            #self.robot_position_map = np.array([self.amcl_pose.iloc[0],self.amcl_pose.iloc[1],0.0])
            #self.robot_orientation_map = np.array([0.0,0.0,self.amcl_pose.iloc[2],self.amcl_pose.iloc[3]])

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

            # Save local_map_info instance to file
            self.local_map_info = self.local_map_info.transpose()
            self.local_map_info.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

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

    # call local planner
    def create_labels(self, classifier_fn):
        try:
            start = time.time()

            # load ontology
            # ontology is updated in the subscriber as the robot moves (in real-world case, when the new detected objects are not already in the ontology)
            self.ontology = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv'))
            
            # load object-affordance pairs
            # object-affordance pairs are updated in the subscriber as the robot is moving
            self.object_affordance_pairs = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/object_affordance_pairs.csv'))
            #print('self.object_affordance_pairs = ', self.object_affordance_pairs)
            
            # load inflated semantic_map (changes the traditional local local_map)
            # semantic_map and inflated semantic_map are also updated in the subscriber as the robot is moving
            #self.semantic_map_inflated = np.array(pd.read_csv(self.dirCurr + '/' + self.dirData + '/semantic_map_inflated.csv'))
            #print('self.semantic_map_inflated.shape = ', self.semantic_map_inflated.shape)


            # create data (encoded perturbations)
            # 2**N semantic_map -- explore all object-affordance perturbations
            n_features = self.object_affordance_pairs.shape[0]
            num_samples = 2**n_features
            lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
            self.data = np.array(lst).reshape((num_samples, n_features))
            #print('self.data = ', self.data)
            #print('self.data.shape = ', self.data.shape)
            #print('type(self.data) = ', type(self.data))
            
            # N+1 or X=O(n) semantic_map
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
                temp = copy.deepcopy(self.semantic_map_inflated)
                temp[temp > 0] = self.hard_obstacle
                
                # find the indices of features (obj.-aff. pairs) which are in their alternative state
                zero_indices = np.where(self.data[i] == 0)[0]
                #print('zero_indices = ', zero_indices)
                
                for j in range(0, zero_indices.shape[0]):
                    # if feature has 0 in self.data it is in its alternative affordance state (currently: either not there or opened) --> original semantic map must be modified
                    idx = zero_indices[j]
                    #print('idx = ', idx)

                    # movability affordance 
                    # the object movability affordance is in its alternative state
                    # the object must be moved
                    if self.object_affordance_pairs[idx][2] == 'movability':

                        #print(self.object_affordance_pairs[idx])
                        temp[self.semantic_map_inflated == self.object_affordance_pairs[idx][0]] = 0

                    
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

                            temp[self.semantic_map_inflated == label] = 0
                            
                            # WRITE AN INDEPENDENT CODE FOR OPENING/CLOSING DOORS
                            # otvara se preko manje/krace strane - pogledaj malu svesku
                            # object"s vertices from /map frame to /odom and /lc frames
                            door_tl_map_x = self.ontology[-1][2] + 0.5*self.ontology[-1][4]
                            door_tl_map_y = self.ontology[-1][3] + 0.5*self.ontology[-1][5]
                            p_map = np.array([door_tl_map_x, door_tl_map_y, 0.0])
                            p_odom = p_map.dot(self.r_mo) + self.t_mo
                            tl_odom_x = p_odom[0]
                            tl_odom_y = p_odom[1]
                            door_tl_pixel_x = int((tl_odom_x - self.local_map_origin_x) / self.local_map_resolution)
                            door_tl_pixel_y = int((tl_odom_y - self.local_map_origin_y) / self.local_map_resolution)
                            
                            x_size = int(self.ontology[label - 1][4] / self.local_map_resolution)
                            y_size = int(self.ontology[label - 1][5] / self.local_map_resolution)
                            #print('(x_size, y_size) = ', (x_size, y_size))

                            #temp[door_tl_pixel_y:door_tl_pixel_y + x_size, door_tl_pixel_x:door_tl_pixel_x + y_size] = label
                            inflation_x = int((max(23, x_size) - x_size) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                            inflation_y = int((max(23, y_size) - y_size) / 2)
                            #print('error with door!!!')
                            #print(max(0, door_tl_pixel_y-inflation_x))
                            #print(min(self.local_map_size-1, door_tl_pixel_y+x_size+inflation_x))
                            #print(max(0,door_tl_pixel_x-inflation_y))
                            #print(min(self.local_map_size-1, door_tl_pixel_x+y_size+inflation_y))                 
                            temp[max(0, door_tl_pixel_y-inflation_x):min(self.local_map_size-1, door_tl_pixel_y+x_size+inflation_x), max(0,door_tl_pixel_x-inflation_y):min(self.local_map_size-1, door_tl_pixel_x+y_size+inflation_y)] = self.hard_obstacle
                                            
                        # MORAS DODATI U ONTOLOGY SA KOJE SE STRANE VRATA OD ORMARA OTVARAJU: TOP, LEFT, RIGHT, BOTTOM 
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
                            cab_tl_pixel_x = int((tl_odom_x - self.local_map_origin_x) / self.local_map_resolution)
                            cab_tl_pixel_y = int((tl_odom_y - self.local_map_origin_y) / self.local_map_resolution)
                            #print('(cab_tl_pixel_x, cab_tl_pixel_y) = ', (cab_tl_pixel_x, cab_tl_pixel_y))
                            
                            # tr
                            cab_tr_map_x = self.ontology[label-1][2] + 0.5*self.ontology[label-1][4]
                            cab_tr_map_y = self.ontology[label-1][3] + 0.5*self.ontology[label-1][5]
                            p_map = np.array([cab_tr_map_x, cab_tr_map_y, 0.0])
                            p_odom = p_map.dot(self.r_mo) + self.t_mo
                            tr_odom_x = p_odom[0]
                            tr_odom_y = p_odom[1]
                            cab_tr_pixel_x = int((tr_odom_x - self.local_map_origin_x) / self.local_map_resolution)
                            cab_tr_pixel_y = int((tr_odom_y - self.local_map_origin_y) / self.local_map_resolution)
                            #print('(cab_tr_pixel_x, cab_tr_pixel_y) = ', (cab_tr_pixel_x, cab_tr_pixel_y))

                            inflation_x = int((max(23, self.ontology[label-1][4]) - self.ontology[label-1][4]) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                            inflation_y = int((max(23, self.ontology[label-1][5]) - self.ontology[label-1][5]) / 2) 
                            x_size = int(1.0 * self.ontology[label - 1][4] / self.local_map_resolution)
                            y_size = int(1.0 * self.ontology[label - 1][5] / self.local_map_resolution)
                            #print('error with cabinet!!!')
                            #print(max(0,cab_tl_pixel_y-inflation_y))
                            #print(min(self.local_map_size-1, cab_tl_pixel_y+y_size+inflation_y))
                            #print(max(0,cab_tl_pixel_x-inflation_x))
                            #print(min(self.local_map_size-1, cab_tl_pixel_x+x_size+inflation_x))
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
        
            end = time.time()
            print('LABELS CREATION RUNTIME = ', round(end-start,3))
        
        except Exception as e:
            print('exception = ', e)
            print('\nLabels not created!')
            return False

        return True   

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

        robot_x_idx = int((robot_x - self.local_map_origin_x) / self.local_map_resolution)
        robot_y_idx = self.local_map_size - 1 - int((robot_y - self.local_map_origin_y) / self.local_map_resolution)

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
            x_temp = int((transformed_plan[i, 0] - self.local_map_origin_x) / self.local_map_resolution)
            y_temp = int((transformed_plan[i, 1] - self.local_map_origin_y) / self.local_map_resolution)

            if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                transformed_plan_xs_idx.append(x_temp)
                transformed_plan_ys_idx.append(self.local_map_size - 1 - y_temp)

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
                    x_temp = int((local_plan.iloc[j, 0] - self.local_map_origin_x) / self.local_map_resolution)
                    y_temp = int((local_plan.iloc[j, 1] - self.local_map_origin_y) / self.local_map_resolution)

                    if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                        local_plan_xs_idx.append(x_temp)
                        local_plan_ys_idx.append(self.local_map_size - 1 - y_temp)

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
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instances_shape_len == 3:
            temp = sampled_instances.reshape(sampled_instances.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif sampled_instances_shape_len == 2:
            np.savetxt(self.dirCurr + 'src/teb_local_planner/src/Data/costmap_data.csv', sampled_instances, delimiter=",")
        end = time.time()
        print('classifier_fn: LOCAL PLANNER DATA PREPARATION RUNTIME = ', round(end-start,3))

        # calling ROS C++ node
        #print('\nC++ node started')

        start = time.time()
        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        #rospy.wait_for_service("/perturb_node_image/finished")

        time.sleep(1.5)
        
        # kill ROS node
        #Popen(shlex.split('rosnode kill /perturb_node_image'))
        
        end = time.time()
        print('classifier_fn: REAL LOCAL PLANNER RUNTIME = ', round(end-start,3))

        #print('\nC++ node ended')

        start = time.time()
        # load local path planner's outputs
        # load command velocities - output from local planner
        cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')
        print('cmd_vel_perturb = ', cmd_vel_perturb)

        # load local plans - output from local planner
        local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        print('local_plans = ', local_plans)

        # load transformed global plan to /odom frame
        transformed_plan = np.array(pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        print('transformed_plan = ', transformed_plan)

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
        # if data not loaded do not explain
        if self.load_data() == False:
            return
        
        # save data for the local planner
        if self.save_data_for_local_planner() == False:
            return

        # get current local costmap data
        #print(self.local_map_info)
        self.local_map_origin_x = self.local_map_info.iloc[0,2]
        self.local_map_origin_y = self.local_map_info.iloc[0,3]
        self.local_map_resolution = self.local_map_info.iloc[0,1]
        self.local_map_size = int(self.local_map_info.iloc[0,0])

        '''
        # call the local planner
        self.labels=(0,)
        self.top = self.labels
        if self.create_labels(self.classifier_fn) == False:
            return

        # create distances between the instance of interest and perturbations
        self.create_distances()

        self.counter_global += 1

        # Explanation variables
        top_labels=1 #10
        model_regressor = None
        num_features=100000
        feature_selection='auto'

        try:
            start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.image_rgb, self.semantic_map, self.object_affordance_pairs, self.ontology)
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
            print('\nMODEL FITTING TIME = ', round(end-start,3))

            start = time.time()
            # get explanation image
            outputs, exp, weights, rgb_values = ret_exp.get_image_and_mask(label=0)
            end = time.time()
            print('\nGET EXP PIC TIME = ', round(end-start,3))
            print('exp: ', exp)

            centroids_for_plot = []
            lc_regions = regionprops(self.semantic_map.astype(int))
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
                for j in range(0, self.local_plan.shape[0]):
                        x_temp = int((self.local_plan[j, 0] - self.locallocal_mapOriginX) / self.locallocal_mapResolution)
                        y_temp = self.local_map_size - int((self.local_plan[j, 1] - self.locallocal_mapOriginY) / self.locallocal_mapResolution)

                        if 0 <= x_temp < self.local_map_size and 0 <= y_temp < self.local_map_size:
                            local_plan_x_list.append(x_temp)
                            local_plan_y_list.append(y_temp)

                for k in range(0, len(outputs)):
                    output = outputs[k]
                    output[:,:,0] = np.flip(output[:,:,0], axis=0)
                    output[:,:,1] = np.flip(output[:,:,1], axis=0)
                    output[:,:,2] = np.flip(output[:,:,2], axis=0)

                    fig.add_axes(ax)
                    ax.scatter(self.global_plan_xs, self.global_plan_ys, c='blue', marker='x')
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
        '''

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