#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovariance, PoseWithCovarianceStamped, PoseStamped, Pose
import rospy
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
import copy
import tf2_ros
import os
from gazebo_msgs.msg import ModelStates, ModelState
import math
from skimage.measure import regionprops
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
import message_filters
import torch
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import cv2
from sensor_msgs import point_cloud2
import struct
import math
import shlex
from psutil import Popen
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
import sklearn.metrics
import matplotlib
from scipy import misc
import PIL
from visualization_msgs.msg import Marker, MarkerArray
from PIL import ImageFont, ImageDraw
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
import actionlib
import tf

class klasa(object):
    def what_to_explain(control_param, locality_param):
        """
        what_to_explain does blah blah blah.

        :control_param: decodes what to explain:
            1. immediate robot action - only localvsglobal
            2. current contextualised robot action/behavior - more localvsglobal or globalvsglobal
            3. navigation history so far - both + some more info
            4. complete trajectory after reaching goal - 3 scoped on the whole start-goal navigation
        :return: wanted explanation
        """ 
        if control_param == 1:
            pass
        elif control_param == 2:
            pass
        elif control_param == 3:
            pass
        elif control_param == 4:
            pass

    def when_to_explain(control_param):
        """
        when_to_explain does blah blah blah.

        :control_param: decodes what to explain:
            1. every time step
            2. when human is detected
            3. when human is need
            4. when human asks a question
        :return: wanted explanation
        """ 
        pass

    def how_to_explain(control_param):
        """
        how_to_explain does blah blah blah.

        :control_param: decodes what to explain:
            1. visual
            2. textual
            3. verbal
            4. visual + textual
            5. visual + verbal
            6. textual + verbal
            7. visual+textual+verbal
        :return: wanted explanation
        """ 
        pass

    def how_long_to_explain(control_param):

        """
        how_long_to_explain does blah blah blah.

        :control_param: decodes what to explain:
            1. until current action is finished
            2. until human need is fulfilled
            3. until human finishes discussion
        :return: wanted explanation
        """
        pass

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
        self.use_maximum_weight = False
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
        print('exp = ', exp)

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
        v_s = []
        imgs = [np.zeros((self.image.shape[0],self.image.shape[0],3))]*len(exp)
        rgb_values = []

        all_obstacles_exp = np.zeros((self.image.shape[0],self.image.shape[0],3))
        # color free space with gray
        all_obstacles_exp[self.semantic_map == 0, 0] = self.free_space_shade
        all_obstacles_exp[self.semantic_map == 0, 1] = self.free_space_shade
        all_obstacles_exp[self.semantic_map == 0, 2] = self.free_space_shade

        for f, w in exp:
            #print('(f, w): ', (f, w))
            print(self.object_affordance_pairs[f][1] + '_' + self.object_affordance_pairs[f][2] + ' has coefficient ' + str(w))
            w_s[f] = w

            temp = np.zeros((self.image.shape[0],self.image.shape[0],3))
            #temp = copy.deepcopy(self.image[0])
            
            v = self.object_affordance_pairs[f][0]
            #print('temp.shape = ', temp.shape)
            #print('v = ', v)
            #v_s.append(v)

            # color free space with gray
            temp[self.semantic_map == 0, 0] = self.free_space_shade
            temp[self.semantic_map == 0, 1] = self.free_space_shade
            temp[self.semantic_map == 0, 2] = self.free_space_shade

            # color the obstacle-affordance pair with green or red
            if w > 0:
                if self.use_maximum_weight:
                    temp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * w / max_w_abs)
                    all_obstacles_exp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                else:
                    temp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
                    rgb_values.append(self.val_high * w / w_sum_abs)
                    all_obstacles_exp[self.semantic_map == v, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
            elif w < 0:
                if self.use_maximum_weight:
                    temp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                    rgb_values.append(self.val_high * abs(w) / max_w_abs)
                    all_obstacles_exp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w_abs
                else:
                    temp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
                    rgb_values.append(self.val_high * abs(w) / w_sum_abs)
                    all_obstacles_exp[self.semantic_map == v, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum_abs
            #elif w == 0:
            #    temp[self.semantic_map == v, 0] = self.val_low
            #    rgb_values.append(self.val_low * abs(w) / w_sum_abs)

            imgs[f] = temp

        imgs.append(all_obstacles_exp)

        #print('weights = ', w_s)
        #print('get image and mask ending')

        return imgs, exp, w_s, rgb_values

def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

# hixron_subscriber class
class hixron(object):
    # constructor
    def __init__(self):
        self.robot_offset = 9.0

        self.simulation = True

        # hri vars
        self.use_lime = False
        self.last_object_moved_ID = -1
        self.old_plan = Path()
        self.old_plan_bool = False
        self.red_object_countdown = -1
        self.red_object_value = -1
        self.humans = []
        self.human_blinking = False
        self.object_arrow_blinking = False

        # icsr vars
        self.extrovert = 1.0
        self.human_detected = False
        self.navigating = True
        self.stopping = False
        self.deviating = False

        # whether to plot
        self.plot_local_costmap_bool = False
        self.plot_global_costmap_bool = False
        self.plot_local_semantic_map_bool = False
        self.plot_global_semantic_map_bool = False
        self.plot_perturbations = False

        # global counter for plotting
        self.counter_global = 0
        self.local_plan_counter = 0
        self.global_costmap_counter = 0

        # use local and/or global costmap
        self.use_local_costmap = False
        self.use_global_costmap = False

        # use local and/or global (semantic) map
        self.use_local_semantic_map = False
        self.use_global_semantic_map = True
        
        # use camera
        self.use_camera = False

        # inflation
        self.inflation_radius = 0.275

        # explanation layer
        self.explanation_layer_bool = True
        
        # data directories
        self.dirCurr = os.getcwd()
        self.yoloDir = self.dirCurr + '/yolo_data/'

        self.dirMain = 'hixron_data'
        try:
            os.mkdir(self.dirMain)
        except FileExistsError:
            pass

        self.dirData = self.dirMain + '/data'
        try:
            os.mkdir(self.dirData)
        except FileExistsError:
            pass

        if self.plot_local_semantic_map_bool == True:
            self.local_semantic_map_dir = self.dirMain + '/local_semantic_map_images'
            try:
                os.mkdir(self.local_semantic_map_dir)
            except FileExistsError:
                pass

        if self.plot_global_semantic_map_bool == True:
            self.global_semantic_map_dir = self.dirMain + '/global_semantic_map_images'
            try:
                os.mkdir(self.global_semantic_map_dir)
            except FileExistsError:
                pass

        if self.plot_local_costmap_bool == True and self.use_local_costmap == True:
            self.local_costmap_dir = self.dirMain + '/local_costmap_images'
            try:
                os.mkdir(self.local_costmap_dir)
            except FileExistsError:
                pass

        if self.plot_global_costmap_bool == True and self.use_global_costmap == True:
            self.global_costmap_dir = self.dirMain + '/global_costmap_images'
            try:
                os.mkdir(self.global_costmap_dir)
            except FileExistsError:
                pass

        if self.plot_perturbations == True:        
            self.global_perturbation_dir = self.dirMain + '/perturbations'
            try:
                os.mkdir(self.global_perturbation_dir)
            except FileExistsError:
                pass

        # simulation variables
        if self.simulation:
            # gazebo vars
            self.gazebo_names = []
            self.gazebo_poses = []
            self.gazebo_labels = []

        # tf vars
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_odom_map = [] 
        self.tf_map_odom = []

        # robot variables
        #self.robot_position_map = Point(0.0,0.0,0.0)        
        #self.robot_orientation_map = Quaternion(0.0,0.0,0.0,1.0)
        #self.robot_position_odom = Point(0.0,0.0,0.0)        
        #self.robot_orientation_odom = Quaternion(0.0,0.0,0.0,1.0)
        self.robot_pose_map = Pose()
        self.robot_pose_odom = Pose()
        self.footprint = []  

        # plans' variables
        self.local_plan = []
        self.global_plan_current = Path() 
        self.global_plan_history = []
        self.globalPlan_goalPose_indices_history = []

        # deviation & failure variables
        self.hard_obstacle = 99

        # goal pose
        self.goal_pose_current = Pose()
        self.goal_pose_history = []
                
        # local semantic map vars
        self.local_semantic_map_origin_x = 0 
        self.local_semantic_map_origin_y = 0 
        self.local_semantic_map_resolution = 0.025
        self.local_semantic_map_size = 160
        self.local_semantic_map_info = []
        self.local_semantic_map = np.zeros((self.local_semantic_map_size, self.local_semantic_map_size))

        # ontology part
        self.scenario_name = 'library_3' #'scenario1', 'library', 'library_2', 'library_3'
        # load ontology
        self.ontology = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'ontology.csv')
        #cols = ['c_map_x', 'c_map_y', 'd_map_x', 'd_map_y']
        #self.ontology[cols] = self.ontology[cols].astype(float)
        self.ontology = np.array(self.ontology)
        #print(self.ontology)
        for i in range(self.ontology.shape[0]):
            self.ontology[i, 3] += self.robot_offset
            self.ontology[i, 4] -= self.robot_offset

            self.ontology[i, 12] += self.robot_offset
            self.ontology[i, 13] -= self.robot_offset

        # load global semantic map info
        self.global_semantic_map_info = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'map_info.csv')) 
        # global semantic map vars
        self.global_semantic_map_origin_x = float(self.global_semantic_map_info[0,4])
        self.global_semantic_map_origin_x += self.robot_offset  
        self.global_semantic_map_origin_y = float(self.global_semantic_map_info[0,5])
        self.global_semantic_map_origin_y -= self.robot_offset 
        self.global_semantic_map_resolution = float(self.global_semantic_map_info[0,1])
        self.global_semantic_map_size = [int(self.global_semantic_map_info[0,3]), int(self.global_semantic_map_info[0,2])]
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #print(self.global_semantic_map_origin_x, self.global_semantic_map_origin_y, self.global_semantic_map_resolution, self.global_semantic_map_size)
        self.global_semantic_map_complete = []

        # camera variables
        self.camera_image = np.array([])
        self.depth_image = np.array([])
        # camera projection matrix 
        self.P = np.array([])

        self.semantic_labels_marker_array = MarkerArray()

    # declare subscribers
    def main_(self):
        self.pub_move = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.chair_8_moved = False
        self.chair_9_moved = False

        self.chair_8_state = ModelState()
        self.chair_8_state.model_name = 'chair_8'
        self.chair_8_state.reference_frame = 'world'  # ''ground_plane'
        # pose
        self.chair_8_state.pose.position.x = -7.73
        self.chair_8_state.pose.position.y = 5.84
        self.chair_8_state.pose.position.z = 0
        #quaternion = tf.transformations.quaternion_from_euler(0, 0, 3.122560)
        quaternion = euler_to_quaternion(3.122560, 0, 0)
        self.chair_8_state.pose.orientation.x = quaternion[0]
        self.chair_8_state.pose.orientation.y = quaternion[1]
        self.chair_8_state.pose.orientation.z = quaternion[2]
        self.chair_8_state.pose.orientation.w = quaternion[3]
        # twist
        self.chair_8_state.twist.linear.x = 0
        self.chair_8_state.twist.linear.y = 0
        self.chair_8_state.twist.linear.z = 0
        self.chair_8_state.twist.angular.x = 0
        self.chair_8_state.twist.angular.y = 0
        self.chair_8_state.twist.angular.z = 0


        self.chair_9_state = ModelState()
        self.chair_9_state.model_name = 'chair_9'
        self.chair_9_state.reference_frame = 'world'  # ''ground_plane'
        # pose
        self.chair_9_state.pose.position.x = -6.16
        self.chair_9_state.pose.position.y = 3.76
        self.chair_9_state.pose.position.z = 0
        #quaternion = tf.transformations.quaternion_from_euler(0, 0, 3.122560)
        quaternion = euler_to_quaternion(3.122560, 0, 0)
        self.chair_9_state.pose.orientation.x = quaternion[0]
        self.chair_9_state.pose.orientation.y = quaternion[1]
        self.chair_9_state.pose.orientation.z = quaternion[2]
        self.chair_9_state.pose.orientation.w = quaternion[3]
        # twist
        self.chair_9_state.twist.linear.x = 0
        self.chair_9_state.twist.linear.y = 0
        self.chair_9_state.twist.linear.z = 0
        self.chair_9_state.twist.angular.x = 0
        self.chair_9_state.twist.angular.y = 0
        self.chair_9_state.twist.angular.z = 0

        
        # create the base plot structure
        if self.plot_local_costmap_bool == True or self.plot_global_costmap_bool == True or self.plot_local_semantic_map_bool == True or self.plot_global_semantic_map_bool == True:
            self.fig = plt.figure(frameon=False)
            #self.w = 1.6 * 3
            #self.h = 1.6 * 3
            #self.fig.set_size_inches(self.w, self.h)
            self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
            self.ax.set_axis_off()
            self.fig.add_axes(self.ax)

        # subscribers
        # local plan subscriber
        #self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        # global plan subscriber 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        # goal pose subscriber 
        self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)

        # robot footprint subscriber
        #self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        # odometry subscriber
        #self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=1)
        self.pub_current_path = rospy.Publisher('/path_markers', MarkerArray, queue_size=1)
        self.pub_old_path = rospy.Publisher('/old_path_markers', MarkerArray, queue_size=1)

        self.semantic_labels_marker_array = MarkerArray()
        self.current_path_marker_array = MarkerArray()
        self.old_path_marker_array = MarkerArray()

        # global-amcl pose subscriber
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        # local costmap subscriber
        if self.use_local_costmap == True:
            self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        # global costmap subscriber
        if self.use_global_costmap == True:
            self.sub_global_costmap = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_callback)

        # explanation layer
        if self.explanation_layer_bool:
            self.pub_explanation_layer = rospy.Publisher("/explanation_layer", PointCloud2, queue_size=1)
        
            # point_cloud variables
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ]

            # header
            self.header = Header()

        # CV part
        # robot camera subscribers
        if self.use_camera == True:
            self.camera_image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
            self.depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image) #"32FC1"
            self.camera_info_sub = message_filters.Subscriber('/xtion/rgb/camera_info', CameraInfo)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.camera_image_sub, self.depth_sub, self.camera_info_sub], 10, 1.0)
            self.ts.registerCallback(self.camera_feed_callback)
            # Load YOLO model
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom, yolov5n(6)-yolov5s(6)-yolov5m(6)-yolov5l(6)-yolov5x(6)
            self.model = torch.hub.load(self.path_prefix + '/yolov5_master/', 'custom', self.path_prefix + '/models/yolov5s.pt', source='local')  # custom trained model

        # gazebo vars
        if self.simulation:
            pass
            # gazebo model states subscriber
            self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_callback)

            # load gazebo tags
            self.gazebo_labels = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'gazebo_tags.csv')) 
            
    # camera feed callback
    def camera_feed_callback(self, img, depth_img, info):
        #print('\ncamera_feed_callback')

        # RGB IMAGE
        # convert rgb image from robot's camera to np.array
        self.camera_image = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
        # Get image dimensions
        #(height, width) = image.shape[:2]

        # DEPTH IMAGE
        # convert depth image to np array
        self.depth_image = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_img.height, depth_img.width, -1)    
        # fill missing values with negative distance (-1.0)
        self.depth_image = np.nan_to_num(self.depth_image, nan=-1.0)
        
        # get projection matrix
        self.P = info.P

        # potential place to make semantic map, if local costmap is not used
        if self.use_local_costmap == False:
            # create semantic data
            self.create_semantic_data()

            # increase the global counter
            self.counter_global += 1

    # send goal pose
    def send_goal_pose(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
  
        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()
   
        # Creates a goal to send to the action server.
        goal = MoveBaseGoal()
        goal.target_pose.header.seq = 0
        goal.target_pose.header.stamp.secs = 0
        goal.target_pose.header.stamp.nsecs = 0
        goal.target_pose.header.frame_id = "map"

        goal.target_pose.pose.position.x = -7.65 + self.robot_offset
        goal.target_pose.pose.position.y = 2.5 - self.robot_offset
        goal.target_pose.pose.position.z = 0.0

        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = -0.71
        goal.target_pose.pose.orientation.w = 0.70
   
        # Sends the goal to the action server.
        client.send_goal(goal)
        
    # odom callback
    def odom_callback(self, msg):
        #print('odom_callback')
        
        #self.robot_position_odom = msg.pose.pose.position
        #self.robot_orientation_odom = msg.pose.pose.orientation
        self.robot_pose_odom = msg.pose
        self.robot_twist_linear = msg.twist.twist.linear
        self.robot_twist_angular = msg.twist.twist.angular

    # amcl (global) pose callback
    def amcl_callback(self, msg):
        #print('amcl_callback')

        #self.robot_position_map = msg.pose.pose.position
        #self.robot_orientation_map = msg.pose.pose.orientation
        self.robot_pose_map = msg.pose.pose
        #self.robot_pose_map.position.x -= self.robot_offset
        #self.robot_pose_map.position.y += self.robot_offset

        if self.chair_8_moved == False:
            x = -7.73 + self.robot_offset
            y = 5.84 + 1.6 - self.robot_offset

            dx = self.robot_pose_map.position.x - x
            dy = self.robot_pose_map.position.y - y

            dist = math.sqrt(dx**2 + dy**2)

            if dist < 0.5:
                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)

                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)
                self.pub_move.publish(self.chair_8_state)

                self.chair_8_moved = True
        else:
            self.pub_move.publish(self.chair_8_state)

        if self.chair_9_moved == False:
            x = -6.16 + self.robot_offset
            y = 3.76 + 1.8 - self.robot_offset

            dx = self.robot_pose_map.position.x - x
            dy = self.robot_pose_map.position.y - y

            dist = math.sqrt(dx**2 + dy**2)

            if dist < 0.5:
                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)

                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)
                self.pub_move.publish(self.chair_9_state)
                
                self.chair_9_moved = True

        else:
            self.pub_move.publish(self.chair_9_state)
        
    # goal pose callback
    def goal_pose_callback(self, msg):
        print('goal_pose_callback')

        self.goal_pose_current = msg.pose
        self.goal_pose_current.position.x -= self.robot_offset
        self.goal_pose_current.position.y += self.robot_offset
        self.goal_pose_history.append(msg.pose)

    # robot footprint callback
    def footprint_callback(self, msg):
        #print('local_plan_callback')  
        self.footprint = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        
    # Gazebo callback
    def gazebo_callback(self, states_msg):
        #print('gazebo_callback')  
        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

        #self.create_semantic_data()

    # global plan callback
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')
        
        # save global plan to class vars
        self.global_plan_current = copy.deepcopy(msg)    
        self.global_plan_history.append(self.global_plan_current)
        self.globalPlan_goalPose_indices_history.append([len(self.global_plan_history), len(self.goal_pose_history)])
        
        if self.use_global_semantic_map:
            
            # create semantic data
            self.create_semantic_data()

            #if self.global_plans_deviation:
            #self.test_explain()

            # increase the global counter (needed for plotting numeration)
            self.counter_global += 1     

        # potential place to make a local semantic map, if local costmap is not used
        elif self.use_local_costmap == False and self.use_local_semantic_map:

            # update local_map params (origin cordinates)
            self.local_semantic_map_origin_x = self.robot_position_map.x - self.local_semantic_map_resolution * self.local_semantic_map_size * 0.5 
            self.local_semantic_map_origin_y = self.robot_position_map.y - self.local_semantic_map_resolution * self.local_semantic_map_size * 0.5
            self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y]
            
            # create semantic data
            self.create_semantic_data()

            # increase the global counter (needed for plotting numeration)
            self.counter_global += 1   
        
    # local plan callback
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        
        try:
            self.odom_copy = copy.deepcopy(self.odom)
            self.amcl_pose_copy = copy.deepcopy(self.amcl_pose)
            self.footprint_copy = copy.deepcopy(self.footprint)

            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())
            self.tf_map_odom = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]

            # get the local plan
            self.local_plan = []
            for i in range(0,len(msg.poses)):
                self.local_plan.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

            # save the vars for the publisher/explainer
            pd.DataFrame(self.odom_copy).to_csv(self.dirCurr + '/' + self.dirData + '/odom.csv', index=False)#, header=False)
            pd.DataFrame(self.amcl_pose_copy).to_csv(self.dirCurr + '/' + self.dirData + '/amcl_pose.csv', index=False)#, header=False)
            pd.DataFrame(self.footprint_copy).to_csv(self.dirCurr + '/' + self.dirData + '/footprint.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_odom_map).to_csv(self.dirCurr + '/' + self.dirData + '/tf_odom_map.csv', index=False)#, header=False)
            pd.DataFrame(self.tf_map_odom).to_csv(self.dirCurr + '/' + self.dirData + '/tf_map_odom.csv', index=False)#, header=False)            
            pd.DataFrame(self.local_plan).to_csv(self.dirCurr + '/' + self.dirData + '/local_plan.csv', index=False)#, header=False)

            # potential place to make semantic map, if local costmap is not used
            if self.use_local_costmap == False and self.simulation == True:
                # increase the local planner counter
                self.local_plan_counter += 1

                if self.local_plan_counter == 20:
                    # update local_map (costmap) data
                    self.local_semantic_map_origin_x = self.robot_position_map.x - 0.5 * self.local_semantic_map_size * self.local_semantic_map_resolution
                    self.local_semantic_map_origin_y = self.robot_position_map.y - 0.5 * self.local_semantic_map_size * self.local_semantic_map_resolution
                    #self.local_semantic_map_resolution = msg.info.resolution
                    #self.local_semantic_map_size = self.local_semantic_map_size
                    self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y]#, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

                    # create np.array local_map object
                    #self.local_semantic_map = np.zeros((self.local_semantic_map_size,self.local_semantic_map_size))

                    # create semantic data
                    self.create_semantic_data()

                    # increase the global counter
                    self.counter_global += 1
                    # reset the local planner counter
                    self.local_plan_counter = 0

        except:
            #print('exception = ', e) # local plan frame rate is too high to print possible exceptions
            return

    # local costmap callback
    def local_costmap_callback(self, msg):
        print('\nlocal_costmap_callback')
        
        try:          
            # update local_map data
            self.local_semantic_map_origin_x = msg.info.origin.position.x
            self.local_semantic_map_origin_y = msg.info.origin.position.y
            #self.local_semantic_map_resolution = msg.info.resolution
            #self.local_semantic_map_size = self.local_semantic_map_size
            self.local_semantic_map_info = [self.local_semantic_map_resolution, self.local_semantic_map_size, self.local_semantic_map_size, self.local_semantic_map_origin_x, self.local_semantic_map_origin_y, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

            # create np.array local_map object
            self.local_semantic_map = np.asarray(msg.data)
            self.local_semantic_map.resize((self.local_semantic_map_size,self.local_semantic_map_size))

            if self.plot_costmaps_bool == True:
                self.plot_costmaps()
                
            # Turn inflated area to free space and 100s to 99s
            self.local_semantic_map[self.local_semantic_map == 100] = 99
            self.local_semantic_map[self.local_semantic_map <= 98] = 0

            # create semantic map
            self.create_semantic_data()

            # increase the global counter
            self.counter_global += 1

        except Exception as e:
            print('exception = ', e)
            return

    # global costmap callback
    def global_costmap_callback(self, msg):
        print('\nglobal_costmap_callback')
        
        try:
            # create np.array global_map object
            self.global_costmap = np.asarray(msg.data)
            self.global_costmap.resize((msg.info.width,msg.info.height))

            if self.plot_global_costmap_bool == True:
                self.plot_global_costmap()

            # increase the global costmap counter
            self.counter_global_costmap += 1

        except Exception as e:
            print('exception = ', e)
            return

    # plot local costmap
    def plot_local_costmap(self):
        start = time.time()

        dirCurr = self.local_costmap_dir + '/' + str(self.counter_global)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass
        
        local_map_99s_100s = copy.deepcopy(self.local_semantic_map)
        local_map_99s_100s[local_map_99s_100s < 99] = 0        
        #self.fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        local_map_99s_100s = np.flip(local_map_99s_100s, 0)
        self.ax.imshow(local_map_99s_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s_100s.png', transparent=False)
        #self.fig.clf()
        
        local_map_original = copy.deepcopy(self.local_semantic_map)
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_map_original = np.flip(local_map_original, 0)
        self.ax.imshow(local_map_original.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_original.png', transparent=False)
        #self.fig.clf()
        
        local_map_100s = copy.deepcopy(self.local_semantic_map)
        local_map_100s[local_map_100s != 100] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(ax)
        local_map_100s = np.flip(local_map_100s, 0)
        self.ax.imshow(local_map_100s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_100s.png', transparent=False)
        #self.fig.clf()
        
        local_map_99s = copy.deepcopy(self.local_semantic_map)
        local_map_99s[local_map_99s != 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_map_99s = np.flip(local_map_99s, 0)
        self.ax.imshow(local_map_99s.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_99s.png', transparent=False)
        #self.fig.clf()
        
        local_map_less_than_99 = copy.deepcopy(self.local_semantic_map)
        local_map_less_than_99[local_map_less_than_99 >= 99] = 0
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #self.fig.set_size_inches(w, h)
        #ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        local_map_less_than_99 = np.flip(local_map_less_than_99, 0)
        self.ax.imshow(local_map_less_than_99.astype('float64'), aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'local_costmap_less_than_99.png', transparent=False)
        self.fig.clf()
        
        end = time.time()
        print('costmaps plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # plot global costmap
    def plot_global_costmap(self):
        start = time.time()

        dirCurr = self.global_costmap_dir + '/' + str(self.counter_global_costmap)
        try:
            os.mkdir(dirCurr)
        except FileExistsError:
            pass

        self.ax.imshow(self.global_costmap, aspect='auto')
        self.fig.savefig(dirCurr + '/' + 'global_costmap_original.png', transparent=False)
        self.fig.clf()
        
        end = time.time()
        print('global costmap plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # create semantic data
    def create_semantic_data(self):
        # update ontology
        self.update_ontology()

        # create semantic map
        if self.use_local_semantic_map == True:
            self.create_local_semantic_map()
        if self.use_global_semantic_map ==  True:
            self.create_global_semantic_map()

        # plot semantic_map
        if self.plot_local_semantic_map_bool == True:
            self.plot_local_semantic_map()
        if self.plot_global_semantic_map_bool == True:
            self.plot_global_semantic_map()

        # create interpretable features
        #self.create_interpretable_features()

        #if self.explanation_layer_bool:
        #    self.publish_explanation_layer()
        
    # update ontology
    def update_ontology(self):
        # check if any object changed its position from simulation or from object detection (and tracking)

        gazebo_names = copy.deepcopy(self.gazebo_names)
        gazebo_poses = copy.deepcopy(self.gazebo_poses)

        self.humans = []
        for i in range(0, len(gazebo_names)):
            gazebo_poses[i].position.x += self.robot_offset
            gazebo_poses[i].position.y -= self.robot_offset 
            if 'citizen' in gazebo_names[i] or 'human' in gazebo_names[i]:
                self.humans.append(gazebo_poses[i])

        if gazebo_names == [] or gazebo_poses == []:
            return

        # simulation relying on Gazebo
        if self.simulation:
            multiplication_factor = 0.75
            for i in range(7, 9): #(0, self.ontology.shape[0]):
                # if the object has some affordance (etc. movability, openability), then it may have changed its position 
                if self.ontology[i][7] == 1:
                    # get the object's new position from Gazebo
                    obj_gazebo_name = self.ontology[i][1]
                    obj_gazebo_name_idx = gazebo_names.index(obj_gazebo_name)
                    
                    obj_x_new = gazebo_poses[obj_gazebo_name_idx].position.x
                    obj_y_new = gazebo_poses[obj_gazebo_name_idx].position.y

                    obj_x_size = self.ontology[i][5]
                    obj_y_size = self.ontology[i][6]

                    mass_centre = self.ontology[i][9]

                    obj_x_current = copy.deepcopy(self.ontology[i][3])
                    obj_y_current = copy.deepcopy(self.ontology[i][4])

                    # update ontology
                    # almost every object type in Gazebo has a different center of mass
                    if mass_centre == 'tr':
                        # top-right is the mass centre
                        obj_x_new -= 0.5*obj_x_size
                        obj_y_new -= 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 'r':
                        # right is the mass centre
                        obj_x_new -= 0.5*obj_x_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 'br':
                        # bottom-right is the mass centre
                        obj_x_new -= 0.5*obj_x_size
                        obj_y_new += 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 'b':
                        # bottom is the mass centre
                        obj_y_new += 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]#
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 'bl':
                        # bottom-left is the mass centre
                        obj_x_new += 0.5*obj_x_size
                        obj_y_new += 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 'l':
                        # left is the mass centre
                        obj_x_new += 0.5*obj_x_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'
                    
                    elif mass_centre == 'tl':
                        # top-left is the mass centre
                        obj_x_new += 0.5*obj_x_size
                        obj_y_new -= 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][12] += obj_x_new - obj_x_current
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

                    elif mass_centre == 't':
                        # top is the mass centre
                        obj_y_new -= 0.5*obj_y_size
                        # check whether the (centroid) coordinates of the object are changed (enough)
                        diff_x = abs(obj_x_new - obj_x_current)
                        diff_y = abs(obj_y_new - obj_y_current)
                        if diff_x > multiplication_factor*obj_x_size or diff_y > multiplication_factor*obj_y_size:
                            self.ontology[i][3] = obj_x_new
                            self.ontology[i][4] = obj_y_new
                            self.last_object_moved_ID = self.ontology[i][0]
                            self.ontology[i][13] += obj_y_new - obj_y_current
                            if self.ontology[i][11] == 'y':
                                self.ontology[i][11] = 'n'

        # real world or simulation relying on object detection
        elif self.simulation == False:
            # in object detection the center of mass is always the object's centroid
            start = time.time()
    
            # Inference
            results = self.model(self.camera_image)
            end = time.time()
            print('yolov5 inference runtime: ' + str(round(1000*(end-start), 2)) + ' ms')
            
            # Results
            res = np.array(results.pandas().xyxy[0])
            
            # labels, confidences and bounding boxes
            labels = list(res[:,-1])
            #print('original_labels: ', labels)
            confidences = list((res[:,-3]))
            #print('confidences: ', confidences)
            # boundix boxes coordinates
            x_y_min_max = np.array(res[:,0:4])
            
            confidence_threshold = 0.0
            if confidence_threshold > 0.0:
                labels_ = []
                for i in range(len(labels)):
                    if confidences[i] < confidence_threshold:
                        np.delete(x_y_min_max, i, 0)
                    else:
                        labels_.append(labels[i])
                labels = labels_
                #print('filtered_labels: ', labels)

            # get the 3D coordinates of the detected objects in the /map dataframe 
            objects_coordinates_3d = []

            # camera intrinsic parameters
            fx = self.P[0]
            cx = self.P[2]
            #fy = self.P[5]
            #cy = self.P[6]
            for i in range(0, len(labels)):
                # get the depth of the detected object's centroid
                u = int((x_y_min_max[i,0]+x_y_min_max[i,2])/2)
                v = int((x_y_min_max[i,1]+x_y_min_max[i,3])/2)
                depth = self.depth_image[v, u][0]
                # get the 3D positions relative to the robot
                x = depth
                y = (u - cx) * z / fx
                z = 0 #(v - cy) * z / fy
                t_ro_R = [x,y,z]
                
                t_wr_W = [self.robot_position_map.x, self.robot_position_map.y, self.robot_position_map.z]
                r_RW = R.from_quat([self.robot_orientation_map.x, self.robot_orientation_map.y, self.robot_orientation_map.z, self.robot_orientation_map.w]).inv

                t_ro_W = r_RW * t_ro_R
                t_wo_W = t_wr_W + t_ro_W
                
                x_o_new = t_wo_W[0]
                y_o_new = t_wo_W[1]

                for j in range(0, self.ontology.shape[0]):
                    if labels[i] == self.ontology[j][1]:

                        x_o_curr = self.ontology[i][2]
                        y_o_curr = self.ontology[i][3]

                        if abs(x_o_new - x_o_curr) > 0.1 and abs(y_o_new - y_o_curr) > 0.1:
                            self.ontology[i][2] = x_o_new
                            self.ontology[i][3] = y_o_new
                            print('\nThe object ' + labels[i] + ' has changed its position!')
                            print('\nOld position: x = ' + str(x_o_curr) + ', y = ' + str(y_o_curr))
                            print('\nNew position: x = ' + str(x_o_new) + ', y = ' + str(y_o_new))                                

        # save the updated ontology for the publisher
        #pd.DataFrame(self.ontology).to_csv(self.dirCurr + '/' + self.dirData + '/ontology.csv', index=False)#, header=False)
    
    # create local semantic map
    def create_local_semantic_map(self):
        # GET transformations between coordinate frames
        # tf from map to odom
        t_mo = np.asarray([self.tf_map_odom[0],self.tf_map_odom[1],self.tf_map_odom[2]])
        r_mo = R.from_quat([self.tf_map_odom[3],self.tf_map_odom[4],self.tf_map_odom[5],self.tf_map_odom[6]])
        r_mo = np.asarray(r_mo.as_matrix())
        #print('r_mo = ', r_mo)
        #print('t_mo = ', t_mo)

        # tf from odom to map
        t_om = np.asarray([self.tf_odom_map[0],self.tf_odom_map[1],self.tf_odom_map[2]])
        r_om = R.from_quat([self.tf_odom_map[3],self.tf_odom_map[4],self.tf_odom_map[5],self.tf_odom_map[6]])
        r_om = np.asarray(r_om.as_matrix())
        #print('r_om = ', r_om)
        #print('t_om = ', t_om)

        # convert LC points from /odom to /map
        # LC origin is a bottom-left point
        lc_bl_odom_x = self.local_semantic_map_origin_x
        lc_bl_odom_y = self.local_semantic_map_origin_y
        lc_p_odom = np.array([lc_bl_odom_x, lc_bl_odom_y, 0.0])#
        lc_p_map = lc_p_odom.dot(r_om) + t_om#
        lc_map_bl_x = lc_p_map[0]
        lc_map_bl_y = lc_p_map[1]
        
        # LC's top-right point
        lc_tr_odom_x = self.local_semantic_map_origin_x + self.local_semantic_map_size * self.local_semantic_map_resolution
        lc_tr_odom_y = self.local_semantic_map_origin_y + self.local_semantic_map_size * self.local_semantic_map_resolution
        lc_p_odom = np.array([lc_tr_odom_x, lc_tr_odom_y, 0.0])#
        lc_p_map = lc_p_odom.dot(r_om) + t_om#
        lc_map_tr_x = lc_p_map[0]
        lc_map_tr_y = lc_p_map[1]
        
        # LC sides coordinates in the /map frame
        lc_map_left = lc_map_bl_x
        lc_map_right = lc_map_tr_x
        lc_map_bottom = lc_map_bl_y
        lc_map_top = lc_map_tr_y
        #print('(lc_map_left, lc_map_right, lc_map_bottom, lc_map_top) = ', (lc_map_left, lc_map_right, lc_map_bottom, lc_map_top))

        start = time.time()
        self.semantic_map = np.zeros(self.local_semantic_map.shape)
        self.semantic_map_inflated = np.zeros(self.local_semantic_map.shape)
        inflation_factor = 0
        for i in range(0, self.ontology.shape[0]):
            # object's vertices from /map to /odom and /lc
            # top left vertex
            x_size = self.ontology[i][4]
            y_size = self.ontology[i][5]
            c_map_x = self.ontology[i][2]
            c_map_y = self.ontology[i][3]

            # top left vertex
            tl_map_x = c_map_x - 0.5*x_size
            tl_map_y = c_map_y + 0.5*y_size

            # bottom right vertex
            br_map_x = c_map_x + 0.5*x_size
            br_map_y = c_map_y - 0.5*y_size
            
            # top right vertex
            tr_map_x = c_map_x + 0.5*x_size
            tr_map_y = c_map_y + 0.5*y_size
            p_map = np.array([tr_map_x, tr_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo#
            tr_odom_x = p_odom[0]#
            tr_odom_y = p_odom[1]#
            tr_pixel_x = int((tr_odom_x - self.local_semantic_map_origin_x) / self.local_semantic_map_resolution)
            tr_pixel_y = int((tr_odom_y - self.local_semantic_map_origin_y) / self.local_semantic_map_resolution)

            # bottom left vertex
            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y - 0.5*y_size
            p_map = np.array([bl_map_x, bl_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo#
            bl_odom_x = p_odom[0]#
            bl_odom_y = p_odom[1]#
            bl_pixel_x = int((bl_odom_x - self.local_semantic_map_origin_x) / self.local_semantic_map_resolution)
            bl_pixel_y = int((bl_odom_y - self.local_semantic_map_origin_y) / self.local_semantic_map_resolution)

            # object's sides coordinates
            object_left = bl_pixel_x
            object_top = tr_pixel_y
            object_right = tr_pixel_x
            object_bottom = bl_pixel_y

            obstacle_in_neighborhood = False 
            x_1 = 0
            x_2 = 0
            y_1 = 0
            y_2 = 0

            # centroid in LC
            if lc_map_left < c_map_x < lc_map_right and lc_map_bottom < c_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            # top-left(tl) in LC
            elif lc_map_left < tl_map_x < lc_map_right and lc_map_bottom < tl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            # bottom-left(bl) in LC
            elif lc_map_left < bl_map_x < lc_map_right and lc_map_bottom < bl_map_y < lc_map_top:
                x_1 = object_left
                x_2 = min(self.local_semantic_map_size-1,object_right)
                y_1 = object_bottom
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True
                
            # bottom-right(br) in LC
            elif lc_map_left < br_map_x < lc_map_right and lc_map_bottom < br_map_y < lc_map_top:            
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = object_bottom
                y_2 = min(self.local_semantic_map_size-1,object_top)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True
                
            # top-right(tr) in LC
            elif lc_map_left < tr_map_x < lc_map_right and lc_map_bottom < tr_map_y < lc_map_top:
                x_1 = max(0,object_left)
                x_2 = object_right
                y_1 = max(0,object_bottom)
                y_2 = object_top
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))

                obstacle_in_neighborhood = True

            if obstacle_in_neighborhood == True:
                # semantic map
                self.semantic_map[max(0, y_1-inflation_factor):min(self.local_semantic_map_size-1, y_2+inflation_factor), max(0,x_1-inflation_factor):min(self.local_semantic_map_size-1, x_2+inflation_factor)] = i+1
                
                # inflate semantic map using heuristics
                inflation_x = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2) #int(0.33 * (object_right - object_left)) #int((1.66 * (object_right - object_left) - (object_right - object_left)) / 2) 
                inflation_y = int((max(23, abs(object_bottom - object_top)) - abs(object_bottom - object_top)) / 2)                 
                self.semantic_map_inflated[max(0, y_1-inflation_y):min(self.local_semantic_map_size-1, y_2+inflation_y), max(0,x_1-inflation_x):min(self.local_semantic_map_size-1, x_2+inflation_x)] = i+1
       
        end = time.time()
        print('semantic map creation runtime = ' + str(round(end-start,3)) + ' seconds!')


        # find centroids of the objects in the semantic map
        lc_regions = regionprops(self.semantic_map.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        self.centroids_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            self.centroids_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        # inflate using the remaining obstacle points of the local costmap            
        if self.use_local_costmap == True:
            for i in range(self.semantic_map.shape[0]):
                for j in range(0, self.semantic_map.shape[1]):
                    if self.local_semantic_map[i, j] > 98 and self.semantic_map_inflated[i, j] == 0:
                        distances_to_centroids = []
                        distances_indices = []
                        for k in range(0, len(self.centroids_semantic_map)):
                            dx = abs(j - self.centroids_semantic_map[k][1])
                            dy = abs(i - self.centroids_semantic_map[k][2])
                            distances_to_centroids.append(dx + dy) # L1
                            #distances_to_centroids.append(math.sqrt(dx**2 + dy**2)) # L2
                            distances_indices.append(k)
                        idx = distances_to_centroids.index(min(distances_to_centroids))
                        self.semantic_map_inflated[i, j] = self.centroids_semantic_map[idx][0]

            # turn pixels in the inflated semantic_map, which are zero in the local costmap, to zero
            self.semantic_map_inflated[self.local_semantic_map == 0] = 0

        # save local and semantic maps data
        pd.DataFrame(self.local_semantic_map_info).to_csv(self.dirCurr + '/' + self.dirData + '/local_map_info.csv', index=False)#, header=False)
        pd.DataFrame(self.local_semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/local_map.csv', index=False) #, header=False)
        pd.DataFrame(self.semantic_map).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map.csv', index=False)#, header=False)
        pd.DataFrame(self.semantic_map_inflated).to_csv(self.dirCurr + '/' + self.dirData + '/semantic_map_inflated.csv', index=False)#, header=False)

    # create global semantic map
    def create_global_semantic_map(self):
        #start = time.time()

        # do not update the map, if
        if self.last_object_moved_ID == -1 and len(np.unique(self.global_semantic_map)) != 1:
            return
        
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        self.global_semantic_map_inflated = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #print('(self.global_semantic_map_size[0],self.global_semantic_map_size[1]) = ', (self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        
        for i in range(0, self.ontology.shape[0]):
            # IMPORTANT OBJECT'S POINTS
            # centroid and size
            c_map_x = float(self.ontology[i][3])
            c_map_y = float(self.ontology[i][4])
            #print('(i, name) = ', (i, self.ontology[i][2]))
            x_size = float(self.ontology[i][5])
            y_size = float(self.ontology[i][6])
            
            # top left vertex
            #tl_map_x = c_map_x - 0.5*x_size
            #tl_map_y = c_map_y - 0.5*y_size
            #tl_pixel_x = int((tl_map_x - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
            #tl_pixel_y = int((tl_map_y - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

            # bottom right vertex
            #br_map_x = c_map_x + 0.5*x_size
            #br_map_y = c_map_y + 0.5*y_size
            #br_pixel_x = int((br_map_x - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
            #br_pixel_y = int((br_map_y - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
            
            # top right vertex
            tr_map_x = c_map_x + 0.5*x_size
            tr_map_y = c_map_y - 0.5*y_size
            tr_pixel_x = int((tr_map_x - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
            tr_pixel_y = int((tr_map_y - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

            # bottom left vertex
            bl_map_x = c_map_x - 0.5*x_size
            bl_map_y = c_map_y + 0.5*y_size
            bl_pixel_x = int((bl_map_x - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
            bl_pixel_y = int((bl_map_y - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

            # object's sides coordinates
            object_left = bl_pixel_x
            object_top = tr_pixel_y
            object_right = tr_pixel_x
            object_bottom = bl_pixel_y
            #print('\n(i, name) = ', (i, self.ontology[i][1]))
            #print('(object_left,object_right,object_top,object_bottom) = ', (object_left,object_right,object_top,object_bottom))
            #print('(c_map_x,c_map_y,x_size,y_size) = ', (c_map_x,c_map_y,x_size,y_size))

            # global semantic map
            self.global_semantic_map[max(0, object_top):min(self.global_semantic_map_size[0], object_bottom), max(0, object_left):min(self.global_semantic_map_size[1], object_right)] = self.ontology[i][0]

            # inflate global semantic map
            #inflation_x = int(self.inflation_radius / self.global_semantic_map_resolution) 
            #inflation_y = int(self.inflation_radius / self.global_semantic_map_resolution)
            #self.global_semantic_map_inflated[max(0, object_top-inflation_y):min(self.global_semantic_map_size[0], object_bottom+inflation_y), max(0, object_left-inflation_x):min(self.global_semantic_map_size[1], object_right+inflation_x)] = i+1

        #end = time.time()
        #print('global semantic map creation runtime = ' + str(round(end-start,3)) + ' seconds!')

        self.global_semantic_map_complete = copy.deepcopy(self.global_semantic_map)

        # find centroids of the objects in the semantic map
        #lc_regions = regionprops(self.global_semantic_map.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        #self.centroids_global_semantic_map = []
        #for lc_region in lc_regions:
        #    v = lc_region.label
        #    cy, cx = lc_region.centroid
        #    self.centroids_global_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

    # plot local semantic_map
    def plot_local_semantic_map(self):
        start = time.time()

        dirCurr = self.local_semantic_map_dir + '/' + str(self.counter_global)            
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
        segs = np.flip(self.semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/semantic_map.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_inflated_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/semantic_map_inflated.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_semantic_map.append([v,cx,cy,self.ontology[v-1][2]])

        for i in range(0, len(centroids_semantic_map)):
            self.ax.scatter(centroids_semantic_map[i][1], centroids_semantic_map[i][2], c='white', marker='o')   
            self.ax.text(centroids_semantic_map[i][1], centroids_semantic_map[i][2], centroids_semantic_map[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_semantic_map).to_csv(dirCurr + '/centroids_semantic_map.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_semantic_map_inflated = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_semantic_map_inflated.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_semantic_map_inflated)):
            self.ax.scatter(centroids_semantic_map_inflated[i][1], centroids_semantic_map_inflated[i][2], c='white', marker='o')   
            self.ax.text(centroids_semantic_map_inflated[i][1], centroids_semantic_map_inflated[i][2], centroids_semantic_map_inflated[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'semantic_map_inflated_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_semantic_map_inflated).to_csv(dirCurr + '/centroids_semantic_map_inflated.csv', index=False)#, header=False)

        end = time.time()
        print('semantic map plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # plot global semantic_map
    def plot_global_semantic_map(self):
        #start = time.time()

        dirCurr = self.global_semantic_map_dir + '/' + str(self.counter_global)            
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
        segs = np.flip(self.global_semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/global_semantic_map.csv', index=False)#, header=False)

        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_inflated_without_labels' + '.png', transparent=False)
        #self.fig.clf()
        pd.DataFrame(segs).to_csv(dirCurr + '/global_semantic_map_inflated.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        #self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        #self.ax.set_axis_off()
        #self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_global_semantic_map = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_global_semantic_map.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_global_semantic_map)):
            self.ax.scatter(centroids_global_semantic_map[i][1], centroids_global_semantic_map[i][2], c='white', marker='o')   
            self.ax.text(centroids_global_semantic_map[i][1], centroids_global_semantic_map[i][2], centroids_global_semantic_map[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_global_semantic_map).to_csv(dirCurr + '/global_centroids_semantic_map.csv', index=False)#, header=False)
        
        #fig = plt.figure(frameon=False)
        #w = 1.6 * 3
        #h = 1.6 * 3
        #fig.set_size_inches(w, h)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)
        segs = np.flip(self.global_semantic_map_inflated, axis=0)
        self.ax.imshow(segs.astype('float64'), aspect='auto')

        # find centroids_in_LC of the objects' areas
        lc_regions = regionprops(segs.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        centroids_global_semantic_map_inflated = []
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            centroids_global_semantic_map_inflated.append([v,cx,cy,self.ontology[v-1][1]])

        for i in range(0, len(centroids_global_semantic_map_inflated)):
            self.ax.scatter(centroids_global_semantic_map_inflated[i][1], centroids_global_semantic_map_inflated[i][2], c='white', marker='o')   
            self.ax.text(centroids_global_semantic_map_inflated[i][1], centroids_global_semantic_map_inflated[i][2], centroids_global_semantic_map_inflated[i][3], c='white')

        self.fig.savefig(dirCurr + '/' + 'global_semantic_map_inflated_with_labels' + '.png', transparent=False)
        self.fig.clf()

        pd.DataFrame(centroids_global_semantic_map_inflated).to_csv(dirCurr + '/global_centroids_semantic_map_inflated.csv', index=False)#, header=False)

        #end = time.time()
        #print('semantic map plotting runtime = ' + str(round(end-start,3)) + ' seconds')

    # create interpretable features
    def create_interpretable_features(self):
        if self.use_global_semantic_map:
            # list of IDs of objects in global semantic map
            IDs = np.unique(self.global_semantic_map)
            
            # get object-affordance pairs in the current global semantic map
            self.object_affordance_pairs_global = [] # ID, object, affordance]
            for i in range(0, len(self.objects_of_interest)):
                if self.objects_of_interest[i][0] in IDs:
                    if self.ontology[i][7] == 1:
                        self.object_affordance_pairs_global.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                    if self.ontology[i][8] == 1:
                        self.object_affordance_pairs_global.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

            #print('object_affordance_pairs_global: ', self.object_affordance_pairs_global)

        if self.use_local_semantic_map:
            # list of labels of objects in local semantic map
            labels = np.unique(self.local_semantic_map)
            object_affordance_pairs_local = [] # [label, object, affordance]
            # get object-affordance pairs in the current local semantic map
            for i in range(0, self.ontology.shape[0]):
                if self.ontology[i][0] in labels:
                    if self.ontology[i][6] == 1:
                        object_affordance_pairs_local.append([self.ontology[i][0], self.ontology[i][1], 'movability'])

                    if self.ontology[i][7] == 1:
                        object_affordance_pairs_local.append([self.ontology[i][0], self.ontology[i][1], 'openability'])

    # publish semantic layer
    def publish_explanation_layer(self, output):
        #points_start = time.time()
            
        z = 0.0
        a = 255                    
        points = []

        # define output
        #output = self.global_semantic_map * 255.0
        #output = self.global_semantic_map_inflated * 255.0
        #output = output[:, :, [2, 1, 0]] * 255.0
        #output = output.astype(np.uint8)

        # draw layer
        size_1 = int(self.global_semantic_map_size[1])
        size_0 = int(self.global_semantic_map_size[0])
        for i in range(0, size_1):
            for j in range(0, size_0):
                x = self.global_semantic_map_origin_x + (size_1-i) * self.global_semantic_map_resolution
                y = self.global_semantic_map_origin_y + j * self.global_semantic_map_resolution
                r = int(output[j, i, 0])
                g = int(output[j, i, 1])
                b = int(output[j, i, 2])
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [x, y, z, rgb]
                points.append(pt)

        #points_end = time.time()
        #print('explanation layer runtime = ', round(points_end - points_start,3))
        
        # publish
        self.header.frame_id = 'map'
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        #print('PUBLISHED!')
        
        self.pub_semantic_labels.publish(self.semantic_labels_marker_array)
        self.pub_current_path.publish(self.current_path_marker_array)
        self.pub_old_path.publish(self.old_path_marker_array)
        self.pub_explanation_layer.publish(pc2)
        
    # decide on explanation variables
    def set_explanation_vars(self):
        explanaton_timing = ['always', 'when_change_in_behavior', '']
        explanaton_duration = ['always', 'when_change_in_behavior', '']
        explanaton_representation = ['visual', 'textual', 'verbal', 'visual-textual', 'visual-verbal', 'textual-verbal', 'visual-textual-verbal']
        if self.extrovert == 1.0:
            pass

    # test whether explanation is needed
    def test_explain(self):
        #print('test_explain!')

        if self.first_call:
            self.create_semantic_data()

        #if self.use_lime == False:
        #self.set_explanation_vars()
        self.explain_global_without_lime()

        #elif self.deviation_between_global_plans and self.use_lime:
        #    self.explain_global_deviation()

    # explain without using lime, rely only on ontology and perception
    def explain_global_without_lime(self):
        # STATIC PART        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)

        if len(np.unique(global_semantic_map_complete_copy)) != self.ontology.shape[0]+1:
            return

        color_shape_path_combination = [2,1,1]
        
        color_schemes = ['only_red', 'red_nuanced', 'green_yellow_red']
        color_scheme = color_schemes[color_shape_path_combination[0]]
        color_whole_objects = False

        shape_schemes = ['wo_text', 'with_text']
        shape_scheme = shape_schemes[color_shape_path_combination[1]]

        path_schemes = ['full_line', 'arrows']
        path_scheme = path_schemes[color_shape_path_combination[2]]

        # define local explanation window around robot
        around_robot_size_x = 2.5
        around_robot_size_y = 2.5

        # create the RGB explanation matrix of the same size as semantic map
        self.explanation_size_y = self.global_semantic_map_size[0]
        self.explanation_size_x = self.global_semantic_map_size[1]
        #print('(self.explanation_size_x,self.explanation_size_y)',(self.explanation_size_y,self.explanation_size_x))
        explanation_R = np.zeros((self.explanation_size_y, self.explanation_size_x))
        explanation_R[:,:] = 120 # free space
        explanation_R[global_semantic_map_complete_copy > 0] = 180.0 # obstacle
        explanation_G = copy.deepcopy(explanation_R)
        explanation_B = copy.deepcopy(explanation_R)

        # find the objects/obstacles in the robot's local neighbourhood
        robot_pose = self.robot_pose_map
        x_min = robot_pose.position.x - around_robot_size_x
        y_min = robot_pose.position.y - around_robot_size_y
        x_max = robot_pose.position.x + around_robot_size_x
        y_max = robot_pose.position.y + around_robot_size_y
        #print('(x_min,x_max,y_min,y_max) = ', (x_min,x_max,y_min,y_max))

        x_min_pixel = int((x_min - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)        
        x_max_pixel = int((x_max - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
        y_min_pixel = int((y_min - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        y_max_pixel = int((y_max - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        x_min_pixel = max(0, x_min_pixel)
        x_max_pixel = min(self.explanation_size_x - 1, x_max_pixel)
        y_min_pixel = max(0, y_min_pixel)
        y_max_pixel = min(self.explanation_size_y - 1, y_max_pixel)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        neighborhood_objects_IDs = np.unique(global_semantic_map_complete_copy[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel])
        if 0 in neighborhood_objects_IDs:
            neighborhood_objects_IDs = neighborhood_objects_IDs[1:]
        neighborhood_objects_IDs = [int(item) for item in neighborhood_objects_IDs]
        #print('neighborhood_objects_IDs =', neighborhood_objects_IDs)

        # OBSTACLE COLORING
        if color_scheme == color_schemes[1]:
            c_x_pixel = int(0.5*(x_min_pixel + x_max_pixel)+1)
            c_y_pixel = int(0.5*(y_min_pixel + y_max_pixel)+1)
            d_x = x_max_pixel - x_min_pixel
            d_y = y_max_pixel - y_min_pixel
            #print('(d_x, d_y) = ', (d_x, d_y))

            R_temp = copy.deepcopy(explanation_R)

            bgr_temp = (np.dstack((explanation_B,explanation_G,explanation_R))).astype(np.uint8)
            #hsv_temp = matplotlib.colors.rgb_to_hsv(bgr_temp)
            hsv_temp = cv2.cvtColor(bgr_temp, cv2.COLOR_BGR2HSV)
            #print(type(hsv_temp))
            #print(hsv_temp.shape)

            hsv_temp[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x), 0] = 0
            hsv_temp[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x), 2] = 255
            

            hsv_temp[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel+int(0.2*d_x), 1] = 160
            
            hsv_temp[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel-int(0.2*d_x), 1] = 135
            hsv_temp[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel+int(0.2*d_x):c_x_pixel+int(0.3*d_x), 1] = 135
            hsv_temp[c_y_pixel-int(0.3*d_y):c_y_pixel-int(0.2*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x), 1] = 135
            hsv_temp[c_y_pixel+int(0.2*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x), 1] = 135
            
            hsv_temp[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel-int(0.3*d_x), 1] = 95
            hsv_temp[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel+int(0.3*d_x):c_x_pixel+int(0.4*d_x), 1] = 95
            hsv_temp[c_y_pixel-int(0.4*d_y):c_y_pixel-int(0.3*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel+int(0.3*d_x), 1] = 95
            hsv_temp[c_y_pixel+int(0.3*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.4*d_x), 1] = 95
            
            hsv_temp[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel-int(0.4*d_x), 1] = 50
            hsv_temp[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel+int(0.4*d_x):c_x_pixel+int(0.5*d_x), 1] = 50
            hsv_temp[c_y_pixel-int(0.5*d_y):c_y_pixel-int(0.4*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x), 1] = 50
            hsv_temp[c_y_pixel+int(0.4*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x), 1] = 50
            

            bgr_temp = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR)
            explanation_R = bgr_temp[:,:,2]
            explanation_G = bgr_temp[:,:,1]
            explanation_B = bgr_temp[:,:,0]

            explanation_R[R_temp == 120] = 120 # return free space to original values
            explanation_G[R_temp == 120] = 120 # return free space to original values
            explanation_B[R_temp == 120] = 120 # return free space to original value

            for row in self.ontology[-4:,:]:
                wall_ID = row[0]
                if wall_ID in neighborhood_objects_IDs:
                    explanation_R[global_semantic_map_complete_copy == wall_ID] = 180
                    explanation_G[global_semantic_map_complete_copy == wall_ID] = 180
                    explanation_B[global_semantic_map_complete_copy == wall_ID] = 180 

        elif color_scheme == color_schemes[2]:
            R_temp = copy.deepcopy(explanation_R)

            c_x_pixel = int(0.5*(x_min_pixel + x_max_pixel)+1)
            c_y_pixel = int(0.5*(y_min_pixel + y_max_pixel)+1)
            d_x = x_max_pixel - x_min_pixel
            d_y = y_max_pixel - y_min_pixel
            #print('(d_x, d_y) = ', (d_x, d_y))

            
            explanation_R[c_y_pixel-int(0.1*d_y):c_y_pixel+int(0.1*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.1*d_x)] = 227
            explanation_G[c_y_pixel-int(0.1*d_y):c_y_pixel+int(0.1*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.1*d_x)] = 242
            explanation_B[c_y_pixel-int(0.1*d_y):c_y_pixel+int(0.1*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.1*d_x)] = 19

            explanation_R[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel-int(0.1*d_x)] = 206
            explanation_R[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel+int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 206
            explanation_R[c_y_pixel-int(0.2*d_y):c_y_pixel-int(0.1*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel+int(0.2*d_x)] = 206
            explanation_R[c_y_pixel+int(0.1*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 206
            explanation_G[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel-int(0.1*d_x)] = 215
            explanation_G[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel+int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 215
            explanation_G[c_y_pixel-int(0.2*d_y):c_y_pixel-int(0.1*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel+int(0.2*d_x)] = 215
            explanation_G[c_y_pixel+int(0.1*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 215
            explanation_B[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel-int(0.1*d_x)] = 15
            explanation_B[c_y_pixel-int(0.2*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel+int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 15
            explanation_B[c_y_pixel-int(0.2*d_y):c_y_pixel-int(0.1*d_y), c_x_pixel-int(0.2*d_x):c_x_pixel+int(0.2*d_x)] = 15
            explanation_B[c_y_pixel+int(0.1*d_y):c_y_pixel+int(0.2*d_y), c_x_pixel-int(0.1*d_x):c_x_pixel+int(0.2*d_x)] = 15
            
            explanation_R[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel-int(0.2*d_x)] = 124
            explanation_R[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel+int(0.2*d_x):c_x_pixel+int(0.3*d_x)] = 124
            explanation_R[c_y_pixel-int(0.3*d_y):c_y_pixel-int(0.2*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 124
            explanation_R[c_y_pixel+int(0.2*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 124
            explanation_G[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel-int(0.2*d_x)] = 220
            explanation_G[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel+int(0.2*d_x):c_x_pixel+int(0.3*d_x)] = 220
            explanation_G[c_y_pixel-int(0.3*d_y):c_y_pixel-int(0.2*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 220
            explanation_G[c_y_pixel+int(0.2*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 220
            explanation_B[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel-int(0.2*d_x)] = 15
            explanation_B[c_y_pixel-int(0.3*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel+int(0.2*d_x):c_x_pixel+int(0.3*d_x)] = 15
            explanation_B[c_y_pixel-int(0.3*d_y):c_y_pixel-int(0.2*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 15
            explanation_B[c_y_pixel+int(0.2*d_y):c_y_pixel+int(0.3*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.3*d_x)] = 15
            
            explanation_R[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel-int(0.3*d_x)] = 108
            explanation_R[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel+int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 108
            explanation_R[c_y_pixel-int(0.4*d_y):c_y_pixel-int(0.3*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel+int(0.3*d_x)] = 108
            explanation_R[c_y_pixel+int(0.3*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 108
            explanation_G[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel-int(0.3*d_x)] = 196
            explanation_G[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel+int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 196
            explanation_G[c_y_pixel-int(0.4*d_y):c_y_pixel-int(0.3*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel+int(0.3*d_x)] = 196
            explanation_G[c_y_pixel+int(0.3*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 196
            explanation_B[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel-int(0.3*d_x)] = 8
            explanation_B[c_y_pixel-int(0.4*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel+int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 8
            explanation_B[c_y_pixel-int(0.4*d_y):c_y_pixel-int(0.3*d_y), c_x_pixel-int(0.4*d_x):c_x_pixel+int(0.3*d_x)] = 8
            explanation_B[c_y_pixel+int(0.3*d_y):c_y_pixel+int(0.4*d_y), c_x_pixel-int(0.3*d_x):c_x_pixel+int(0.4*d_x)] = 8
            
            explanation_R[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel-int(0.4*d_x)] = 98
            explanation_R[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel+int(0.4*d_x):c_x_pixel+int(0.5*d_x)] = 98
            explanation_R[c_y_pixel-int(0.5*d_y):c_y_pixel-int(0.4*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 98
            explanation_R[c_y_pixel+int(0.4*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 98
            explanation_G[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel-int(0.4*d_x)] = 176
            explanation_G[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel+int(0.4*d_x):c_x_pixel+int(0.5*d_x)] = 176
            explanation_G[c_y_pixel-int(0.5*d_y):c_y_pixel-int(0.4*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 176
            explanation_G[c_y_pixel+int(0.4*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 176
            explanation_B[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel-int(0.4*d_x)] = 9
            explanation_B[c_y_pixel-int(0.5*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel+int(0.4*d_x):c_x_pixel+int(0.5*d_x)] = 9
            explanation_B[c_y_pixel-int(0.5*d_y):c_y_pixel-int(0.4*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 9
            explanation_B[c_y_pixel+int(0.4*d_y):c_y_pixel+int(0.5*d_y), c_x_pixel-int(0.5*d_x):c_x_pixel+int(0.5*d_x)] = 9
            
            explanation_R[R_temp == 120] = 120 # return free space to original values
            explanation_G[R_temp == 120] = 120 # return free space to original values
            explanation_B[R_temp == 120] = 120 # return free space to original value

            '''
            if color_whole_objects == True:
                for ID in neighborhood_objects_IDs:
                        if self.ontology[ID-1][7] == 0:
                            explanation_R[global_semantic_map_complete_copy == ID] = 3
                            explanation_G[global_semantic_map_complete_copy == ID] = 144
                            explanation_B[global_semantic_map_complete_copy == ID] = 31
                        elif self.ontology[ID-1][7] == 1:
                            explanation_R[global_semantic_map_complete_copy == ID] = 230
                            explanation_G[global_semantic_map_complete_copy == ID] = 217
                            explanation_B[global_semantic_map_complete_copy == ID] = 26
            else:
                N_objects = self.ontology.shape[0]
                neighborhood_mask = copy.deepcopy(global_semantic_map_complete_copy)
                neighborhood_mask[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel] += N_objects
                for ID in neighborhood_objects_IDs:
                    if self.ontology[ID-1][7] == 1:
                        explanation_R[neighborhood_mask == ID + N_objects] = 230
                        explanation_G[neighborhood_mask == ID + N_objects] = 217
                        explanation_B[neighborhood_mask == ID + N_objects] = 26  
                    if self.ontology[ID-1][7] == 0:
                        explanation_R[neighborhood_mask == ID + N_objects] = 3
                        explanation_G[neighborhood_mask == ID + N_objects] = 144
                        explanation_B[neighborhood_mask == ID + N_objects] = 31
            '''

            for row in self.ontology[-4:,:]:
                wall_ID = row[0]
                if wall_ID in neighborhood_objects_IDs:
                    explanation_R[global_semantic_map_complete_copy == wall_ID] = 180
                    explanation_G[global_semantic_map_complete_copy == wall_ID] = 180
                    explanation_B[global_semantic_map_complete_copy == wall_ID] = 180


                self.dynamic_explanation = False
                        
        # VISUALIZE OBSTACLE NAMES USING PC2
        if shape_scheme == shape_schemes[1]:
            self.semantic_labels_marker_array.markers = []
            for i in range(0, self.ontology.shape[0] - 4):                
                x_map = self.ontology[i][12]
                y_map = self.ontology[i][13]
                
                # visualize orientations and semantic labels of known objects
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = i
                marker.type = marker.TEXT_VIEW_FACING
                if self.ontology[i][0] in neighborhood_objects_IDs:
                    marker.action = marker.ADD
                else:
                    marker.action = marker.DELETE
                marker.pose = Pose()
                marker.pose.position.x = x_map
                marker.pose.position.y = y_map
                marker.pose.position.z = 0.5
                marker.pose.orientation.x = 0.0#qx
                marker.pose.orientation.y = 0.0#qy
                marker.pose.orientation.z = 0.0#qz
                marker.pose.orientation.w = 0.0#qw
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.scale.x = 0.35
                marker.scale.y = 0.35
                marker.scale.z = 0.35
                #marker.frame_locked = False
                marker.text = self.ontology[i][2]
                marker.ns = "my_namespace"
                self.semantic_labels_marker_array.markers.append(marker)
                                         
        # DYNAMIC PART
        if len(self.global_plan_history) > 1:
            # test if there is deviation between current and previous
            self.deviation_between_global_plans = False
            deviation_threshold = 10.0
            
            self.globalPlan_goalPose_indices_history_hold = copy.deepcopy(self.globalPlan_goalPose_indices_history[-2:])
            #self.global_plan_history_hold = copy.deepcopy(self.global_plan_history)
            self.global_plan_current_hold = copy.deepcopy(self.global_plan_history[-1])
            if len(self.global_plan_history) > 1:
                self.global_plan_previous_hold = copy.deepcopy(self.global_plan_history[-2])
            
                min_plan_length = min(len(self.global_plan_current_hold.poses), len(self.global_plan_previous_hold.poses))
                
                # calculate deivation
                global_dev = 0
                for i in range(0, min_plan_length):
                    dev_x = self.global_plan_current_hold.poses[i].pose.position.x - self.global_plan_previous_hold.poses[i].pose.position.x
                    dev_y = self.global_plan_current_hold.poses[i].pose.position.y - self.global_plan_previous_hold.poses[i].pose.position.y
                    local_dev = dev_x**2 + dev_y**2
                    global_dev += local_dev
                global_dev = math.sqrt(global_dev)
                #print('\nDEVIATION BETWEEN GLOBAL PLANS!!! = ', global_dev)
                #print('OBJECT_MOVED_ID = ', self.last_object_moved_ID)
            
                if global_dev > deviation_threshold:
                    #print('DEVIATION BETWEEN GLOBAL PLANS!!! = ', global_dev)
                    self.deviation_between_global_plans = True
                    self.deviating = True

            # check if the last two global plans have the same goal pose
            same_goal_pose = False
            if len(self.globalPlan_goalPose_indices_history_hold) > 1:
                if self.globalPlan_goalPose_indices_history_hold[-1][1] == self.globalPlan_goalPose_indices_history_hold[-2][1]:
                    same_goal_pose = True

            if same_goal_pose == False:
                print('New goal chosen!!!')
                self.old_plan_bool = False

            # if deviation happened and some object was moved
            if self.deviation_between_global_plans and same_goal_pose:
                #print('TESTIRA se moguca devijacija')
                if self.last_object_moved_ID > 0 and self.red_object_countdown == -1: #self.last_object_moved_ID in neighborhood_objects_IDs
                    # define the red object
                    self.red_object_value = copy.deepcopy(self.last_object_moved_ID)
                    self.red_object_countdown = 7
                    self.last_object_moved_ID = -1

                    # save the previous plan
                    self.old_plan = copy.deepcopy(self.global_plan_previous_hold)
                    self.old_plan_bool = True

            # VISUALIZE OLD PATH
            if self.old_plan_bool == True:
                self.old_path_marker_array.markers = []

                #print('len(self.old_plan.poses) = ', len(self.old_plan.poses))
                for i in range(0, len(self.old_plan.poses)):
                    x_map = self.old_plan.poses[i].pose.position.x
                    y_map = self.old_plan.poses[i].pose.position.y

                    # visualize path
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.SPHERE
                    marker.action = marker.ADD
                    marker.pose = Pose()
                    marker.pose.position.x = self.old_plan.poses[i].pose.position.x
                    marker.pose.position.y = self.old_plan.poses[i].pose.position.y
                    marker.pose.position.z = 0.8
                    marker.pose.orientation.x = self.old_plan.poses[i].pose.orientation.x
                    marker.pose.orientation.y = self.old_plan.poses[i].pose.orientation.y
                    marker.pose.orientation.z = self.old_plan.poses[i].pose.orientation.z
                    marker.pose.orientation.w = self.old_plan.poses[i].pose.orientation.w
                    marker.color.r = 0.85
                    marker.color.g = 0.85
                    marker.color.b = 0.85
                    marker.color.a = 0.8
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    #marker.frame_locked = False
                    marker.ns = "my_namespace"
                    self.old_path_marker_array.markers.append(marker)

                for i in range(len(self.old_plan.poses), 600):
                    x_map = 0
                    y_map = 0

                    # visualize path
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.SPHERE
                    marker.action = marker.DELETE
                    marker.pose = Pose()
                    marker.pose.position.x = 0
                    marker.pose.position.y = 0
                    marker.pose.position.z = 0.8
                    marker.pose.orientation.x = 0
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                    marker.pose.orientation.w = 0
                    marker.color.r = 0.85
                    marker.color.g = 0.85
                    marker.color.b = 0.85
                    marker.color.a = 0.8
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    #marker.frame_locked = False
                    marker.ns = "my_namespace"
                    self.old_path_marker_array.markers.append(marker)

            # VISUALIZE CURRENT PATH
            if path_scheme == path_schemes[0]:         
                previous_marker_array_length = len(self.current_path_marker_array.markers)
                current_marker_array_length = len(self.global_plan_current_hold.poses)-25
                delete_needed = False
                if current_marker_array_length < previous_marker_array_length:
                    delete_needed = True
                len_max = max(previous_marker_array_length, current_marker_array_length)
                self.current_path_marker_array.markers = []        
                
                for i in range(25, len_max+25):
                    # visualize path
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.SPHERE
                    marker.action = marker.ADD #DELETEALL #ADD
                    if delete_needed and i >= current_marker_array_length+25:
                        marker.action = marker.DELETE
                        #marker.lifetime = 1.0
                        marker.pose = Pose()
                        marker.pose.position.x = 0.0
                        marker.pose.position.y = 0.0
                        marker.pose.position.z = 0.95
                        marker.pose.orientation.x = 0.0
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 1.0
                        marker.color.r = 0.043
                        marker.color.g = 0.941
                        marker.color.b = 1.0
                        marker.color.a = 0.5        
                        marker.scale.x = 0.1
                        marker.scale.y = 0.1
                        marker.scale.z = 0.1
                        #marker.frame_locked = False
                        marker.ns = "my_namespace"
                        self.current_path_marker_array.markers.append(marker)
                        continue
                    #marker.lifetime = 1.0
                    marker.pose = Pose()
                    marker.pose.position.x = self.global_plan_current_hold.poses[i].pose.position.x
                    marker.pose.position.y = self.global_plan_current_hold.poses[i].pose.position.y
                    marker.pose.position.z = 0.95
                    marker.pose.orientation.x = self.global_plan_current_hold.poses[i].pose.orientation.x
                    marker.pose.orientation.y = self.global_plan_current_hold.poses[i].pose.orientation.y
                    marker.pose.orientation.z = self.global_plan_current_hold.poses[i].pose.orientation.z
                    marker.pose.orientation.w = self.global_plan_current_hold.poses[i].pose.orientation.w
                    marker.color.r = 0.043
                    marker.color.g = 0.941
                    marker.color.b = 1.0
                    marker.color.a = 0.5        
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    #marker.frame_locked = False
                    marker.ns = "my_namespace"
                    self.current_path_marker_array.markers.append(marker)
            elif path_scheme == path_schemes[1]:
                # ARROWS
                previous_marker_array_length = len(self.current_path_marker_array.markers)
                current_marker_array_length = int(float(len(self.global_plan_current_hold.poses)-21) / 75) + 1
                delete_needed = False
                if current_marker_array_length < previous_marker_array_length:
                    delete_needed = True
                len_max = max(previous_marker_array_length, current_marker_array_length)
                self.current_path_marker_array.markers = []        
                        
                marker_ID = 0
                for i in range(35, len(self.global_plan_current_hold.poses) - 1 , 75):
                    # visualize path
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = marker_ID
                    marker_ID += 1
                    marker.type = marker.ARROW
                    marker.action = marker.ADD
                    if delete_needed and marker_ID >= current_marker_array_length:
                        marker.action = marker.DELETE
                        #marker.lifetime = 1.0
                        marker.pose = Pose()
                        marker.pose.position.x = 0.0
                        marker.pose.position.y = 0.0
                        marker.pose.position.z = 0.95
                        marker.pose.orientation.x = 0.0
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 1.0
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 1.0
                        marker.color.a = 0.8
                        marker.scale.x = 1.0
                        marker.scale.y = 0.15
                        marker.scale.z = 0.25
                        #marker.frame_locked = False
                        marker.ns = "my_namespace"
                        self.current_path_marker_array.markers.append(marker)
                        continue
                    marker.pose = Pose()
                    marker.pose.position.x = self.global_plan_current_hold.poses[i].pose.position.x
                    marker.pose.position.y = self.global_plan_current_hold.poses[i].pose.position.y
                    marker.pose.position.z = 0.95
                    marker.pose.orientation.x = self.global_plan_current_hold.poses[i].pose.orientation.x
                    marker.pose.orientation.y = self.global_plan_current_hold.poses[i].pose.orientation.y
                    marker.pose.orientation.z = self.global_plan_current_hold.poses[i].pose.orientation.z
                    marker.pose.orientation.w = self.global_plan_current_hold.poses[i].pose.orientation.w
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 1.0
                    marker.color.a = 0.8
                    marker.scale.x = 1.0
                    marker.scale.y = 0.15
                    marker.scale.z = 0.25
                    #marker.frame_locked = False
                    marker.ns = "my_namespace"
                    self.current_path_marker_array.markers.append(marker)

        # COLOR RED OBJECT
        if self.red_object_countdown > 0:
            #print('BOJI STOLICU CRVENO!!!')
            #print('self.red_object_countdown = ', self.red_object_countdown)
            #print('self.red_object_value = ', self.red_object_value)
            #print('self.last_object_moved_ID = ', self.last_object_moved_ID)

            #RGB_val = [201,9,9]
            RGB_val = [255,0,0]
            explanation_R[global_semantic_map_complete_copy == self.red_object_value] = RGB_val[0]
            explanation_G[global_semantic_map_complete_copy == self.red_object_value] = RGB_val[1]
            explanation_B[global_semantic_map_complete_copy == self.red_object_value] = RGB_val[2]
            
            self.red_object_countdown -= 1
        elif self.red_object_countdown == 0:
            self.red_object_countdown = -1
            self.red_object_value = -1


        # FORM THE EXPLANATION IMAGE                  
        explanation = (np.dstack((explanation_R,explanation_G,explanation_B))).astype(np.uint8)

        #font = {'family' : 'fantasy', #{'cursive', 'fantasy', 'monospace', 'sans', 'sans serif', 'sans-serif', 'serif'}
        #'weight' : 'normal', #[ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
        #'size'   : 5}
        #matplotlib.rc('font', **font)

        fig = plt.figure(frameon=True)
        w = 0.01 * self.explanation_size_x
        h = 0.01 * self.explanation_size_y
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(np.fliplr(explanation)) #np.flip(explanation))#.astype(np.uint8))

        # VISUALIZE OBSTACLE ARROWS AROUND MOVABLE OBJECTS USING MATPLOTLIB
        for i in range(0, self.ontology.shape[0] - 4):            
            # if it is a movable object and in the robot's neighborhood
            if self.ontology[i][0] in neighborhood_objects_IDs and self.ontology[i][7] == 1:
                x_map = self.ontology[i][3]
                y_map = self.ontology[i][4]

                x_pixel = int((x_map - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
                y_pixel = int((y_map - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

                dx = int(self.ontology[i][5] / 0.05)
                dy = int(self.ontology[i][6] / 0.05)

                xs_plot = []
                ys_plot = []
                arrows = []

                # if object under table
                if self.ontology[i][11] == 'y':
                    if self.ontology[i][10] == 'r':
                        xs_plot.append(self.explanation_size_x - x_pixel + dx - 1)
                        ys_plot.append(y_pixel - 1)
                        arrows.append('>')
                    elif self.ontology[i][10] == 'l':
                        xs_plot.append(self.explanation_size_x - x_pixel - dx - 1)
                        ys_plot.append(y_pixel - 1)
                        arrows.append('<')
                    elif self.ontology[i][10] == 't':
                        xs_plot.append(self.explanation_size_x - x_pixel - 2)
                        ys_plot.append(y_pixel - dy - 2)
                        arrows.append('^')
                    elif self.ontology[i][10] == 'b':
                        xs_plot.append(self.explanation_size_x - x_pixel - 2)
                        ys_plot.append(y_pixel + dy - 1)
                        arrows.append('v')
                # if object is not under the table
                elif self.ontology[i][11] == 'n' or self.ontology[i][11] == 'na':
                    xs_plot.append(self.explanation_size_x - x_pixel + dx - 1)
                    ys_plot.append(y_pixel - 1)
                    arrows.append('>')
                    xs_plot.append(self.explanation_size_x - x_pixel - dx - 1)
                    ys_plot.append(y_pixel - 1)
                    arrows.append('<')

                    xs_plot.append(self.explanation_size_x - x_pixel - 2)
                    ys_plot.append(y_pixel - dy - 2)
                    arrows.append('^')
                    xs_plot.append(self.explanation_size_x - x_pixel - 2)
                    ys_plot.append(y_pixel + dy - 1)
                    arrows.append('v')
                '''
                elif self.ontology[i][11] == 'na':
                    objects_to_the_left = np.unique(global_semantic_map_complete_copy[int(y_pixel-0.5*dy_):int(y_pixel+0.5*dy_),int(x_pixel-1.5*dx_):int(x_pixel-0.5*dx_)]) 
                    objects_to_the_right = np.unique(global_semantic_map_complete_copy[int(y_pixel-0.5*dy_):int(y_pixel+0.5*dy_),int(x_pixel+0.5*dx_):int(x_pixel+1.5*dx_)])
                    objects_to_the_top = np.unique(global_semantic_map_complete_copy[int(y_pixel-1.5*dy_):int(y_pixel-0.5*dy_),int(x_pixel-0.5*dx_):int(x_pixel+0.5*dx_)])
                    objects_to_the_bottom = np.unique(global_semantic_map_complete_copy[int(y_pixel-1.5*dy_):int(y_pixel-0.9*dy_),int(x_pixel-0.5*dx_):int(x_pixel+0.5*dx_)])

                    if len(objects_to_the_left) == 1 and objects_to_the_left[0] == 0:
                        xs_plot.append(self.explanation_size_x - x_pixel - dx)
                        ys_plot.append(y_pixel)
                        arrows.append('<')

                    if len(objects_to_the_right) == 1 and objects_to_the_right[0] == 0:
                        xs_plot.append(self.explanation_size_x - x_pixel + dx)
                        ys_plot.append(y_pixel)
                        arrows.append('>')

                    if len(objects_to_the_top) == 1 and objects_to_the_top[0] == 0:
                        xs_plot.append(self.explanation_size_x - x_pixel)
                        ys_plot.append(y_pixel - dy)
                        arrows.append('^')

                    if len(objects_to_the_bottom) == 1 and objects_to_the_bottom[0] == 0:
                        xs_plot.append(self.explanation_size_x - x_pixel)
                        ys_plot.append(y_pixel + dy)
                        arrows.append('v')
                '''

                if self.ontology[i][0] == self.red_object_value:
                    for j in range(0, len(arrows)):
                        #C = np.array([201, 9, 9])
                        C = np.array([255, 0, 0])
                        plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)
                elif color_scheme == color_schemes[1] or color_scheme == color_schemes[2]:
                    for j in range(0, len(arrows)):
                        R = explanation[y_pixel][x_pixel][0]
                        G = explanation[y_pixel][x_pixel][1]
                        B = explanation[y_pixel][x_pixel][2]
                        C = np.array([R, G, B])
                        #plt.plot(xs_plot[j] + 0.3, ys_plot[j] - 0.2, marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)
                        plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)
                elif color_scheme == color_schemes[0]:
                    C = np.array([180, 180, 180])   
                    for j in range(0, len(arrows)):
                        plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)

        # PLOT HUMAN AS BLINKING EXCLAMATION MARK
        ID = self.ontology.shape[0]
        for human_pose in self.humans:
            x_map = human_pose.position.x
            y_map = human_pose.position.y
            
            distance_human_robot = math.sqrt((x_map - robot_pose.position.x)**2 + (y_map - robot_pose.position.y)**2)
            
            # visualize orientations and semantic labels of humans
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = ID
            ID += 1
            marker.type = marker.TEXT_VIEW_FACING
            if distance_human_robot > 2.0:
                marker.action = marker.DELETE
            else:
                marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = x_map + 0.25
            marker.pose.position.y = y_map - 0.6
            marker.pose.position.z = 0.5
            marker.pose.orientation.x = 0.0#qx
            marker.pose.orientation.y = 0.0#qy
            marker.pose.orientation.z = 0.0#qz
            marker.pose.orientation.w = 0.0#qw
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.35
            #marker.frame_locked = False
            marker.text = "human"
            marker.ns = "my_namespace"
            self.semantic_labels_marker_array.markers.append(marker)

            if self.human_blinking == True:
                # for nicer plotting
                x_map += 0.2
                y_map += 0.2

                x_pixel = int((x_map + - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
                y_pixel = int((y_map - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

                if distance_human_robot > 2.0:
                    ax.text(self.explanation_size_x - x_pixel, y_pixel, 'i', c='yellow')
                else:
                    C = np.array([255,102,102])
                    ax.text(self.explanation_size_x - x_pixel, y_pixel, 'i', c=C/255.0)
        
        if self.human_blinking == True:
            self.human_blinking = False
        else:
            self.human_blinking = True
        

        # CONVERT IMAGE TO NUMPY ARRAY 
        fig.savefig('explanation' + '.png', transparent=False)
        plt.close()
        explanation = PIL.Image.open(os.getcwd() + '/explanation.png').convert('RGB')        
        explanation = np.array(explanation)[:,:,:3].astype(np.uint8)

        # PUBLISH EXPLANATION LAYER
        self.publish_explanation_layer(explanation)

    # explain deviation between two global plans
    def explain_global_deviation(self):
        # check if the last two global plans have the same goal pose
        same_goal_pose = False
        if len(self.globalPlan_goalPose_indices_history) > 1:
            if self.globalPlan_goalPose_indices_history[-1][1] == self.globalPlan_goalPose_indices_history[-2][1]:
                same_goal_pose = True

        if same_goal_pose:
            # find the objects/obstacles of interest
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
        
            goal_pose = self.goal_pose_current #self.goal_pose_history[-1]
            robot_pose = self.robot_pose_map

            x_min = min(robot_pose.position.x, goal_pose.position.x)
            x_max = max(robot_pose.position.x, goal_pose.position.x)
            y_min = min(robot_pose.position.y, goal_pose.position.y)
            y_max = max(robot_pose.position.y, goal_pose.position.y)
            #print('(x_min,x_max,y_min,y_max) = ', (x_min,x_max,y_min,y_max))
            
            d_x = x_max - x_min
            d_y = y_max - y_min
            
            self.objects_of_interest = []
            if d_x >= d_y:
                for i in range(0, self.ontology.shape[0]):
                    x_obj = self.ontology[i][3]
                    y_obj = self.ontology[i][4]
                    #print('(x_obj, y_obj) = ', (x_obj, y_obj))
                    if x_obj > x_min and x_obj < x_max:
                        self.objects_of_interest.append(self.ontology[i])
      
            else:
                for i in range(0, self.ontology.shape[0]):
                    x_obj = self.ontology[i][3]
                    y_obj = self.ontology[i][4]
                    #print('(x_obj, y_obj) = ', (x_obj, y_obj))
                    if y_obj > y_min and y_obj < y_max:
                        self.objects_of_interest.append(self.ontology[i])
      
            print('There are ' + str(len(self.objects_of_interest)) + ' objects of interest!!!')

            # create interpretable features
            self.create_interpretable_features()

            # get the labels - call the planner
            self.get_labels()

            # create distances between the instance of interest and perturbations
            self.create_distances()

            #self.counter_global += 1

            # Explanation variables
            top_labels=1 #10
            model_regressor = None
            num_features=100000
            feature_selection='auto'

            #start = time.time()
            # find explanation
            ret_exp = ImageExplanation(self.global_semantic_map, self.global_semantic_map, self.object_affordance_pairs_global, self.ontology)
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
            #end = time.time()
            #print('\nMODEL FITTING TIME = ', round(end-start,3))

            #start = time.time()
            # get explanation image
            self.outputs, self.exp, self.weights, self.rgb_values = ret_exp.get_image_and_mask(label=0)
            #end = time.time()
            #print('\nGET EXP PIC TIME = ', round(end-start,3))
            print('exp: ', self.exp)
                
        else:
            print('New goal chosen!!!')

    # get labels for perturbed data
    def get_labels(self):
        #start = time.time()

        # create encoded perturbation data
        self.create_encoded_perturbation_data()
        
        num_samples = self.data.shape[0]
        #n_features = data_width
        
        # get labels
        self.labels = []
        imgs = []

        for i in range(0, num_samples):
            #temp = copy.deepcopy(self.global_semantic_map)
            temp = copy.deepcopy(self.global_semantic_map_inflated)
            
            # find the indices of features (obj.-aff. pairs) which are in their alternative state
            zero_indices = np.where(self.data[i] == 0)[0]
            print('zero_indices = ', zero_indices)
            
            for j in range(0, zero_indices.shape[0]):
                # if feature has 0 in self.data it is in its alternative affordance state
                # --> original semantic map must be modified
                idx = zero_indices[j]
                #print('idx = ', idx)
                temp[temp == idx + 1] = 0

            temp[temp > 0] = self.hard_obstacle

            imgs.append(temp)

        
        plot_perturbations = False
        if plot_perturbations:
            dirCurr = self.global_perturbation_dir + '/' + str(self.counter_global)
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
            preds = self.call_planner(np.array(imgs))
            self.labels.extend(preds)
        self.labels = np.array(self.labels)
        print('labels = ', self.labels)
    
        #end = time.time()
        #print('LABELS CREATION RUNTIME = ', round(end-start,3))
    
    # create encoded perturbation data
    def create_encoded_perturbation_data(self):
        # create data (encoded perturbations)
        # old approach
        #n_features = self.object_affordance_pairs.shape[0]
        #num_samples = 2**n_features
        #lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
        #self.data = np.array(lst).reshape((num_samples, n_features))

        data_height = len(self.object_affordance_pairs_global)
        data_width = self.ontology.shape[0]
        self.data = np.array([[1]*data_width]*data_height)
        
        for i in range(0, data_height):
            self.data[i][self.object_affordance_pairs_global[i][0] - 1] = 0

    # call the planner and get outputs for perturbed inputs
    def call_planner(self, sampled_instances):
        # save perturbations for the local planner
        #start = time.time()
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
        #end = time.time()
        #print('classifier_fn: LOCAL PLANNER DATA PREPARATION RUNTIME = ', round(end-start,3))

        # calling ROS C++ node
        #print('\nC++ node started')

        #start = time.time()
        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        #rospy.wait_for_service("/perturb_node_image/finished")

        time.sleep(0.35)
        
        # kill ROS node
        #Popen(shlex.split('rosnode kill /perturb_node_image'))
        
        #end = time.time()
        #print('classifier_fn: REAL LOCAL PLANNER RUNTIME = ', round(end-start,3))

        #print('\nC++ node ended')

        
        # load local path planner's outputsstart = time.time()
        start = time.time()
        # load command velocities - output from local planner
        #cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('cmd_vel_perturb = ', cmd_vel_perturb)

        # load local plans - output from local planner
        #local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('local_plans = ', local_plans)

        # load transformed global plan to /odom frame
        #transformed_plan = np.array(pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv'))
        #print('transformed_plan = ', transformed_plan)

        #end = time.time()
        #print('classifier_fn: RESULTS LOADING RUNTIME = ', round(end-start,3))

        global_plans_deviation = pd.DataFrame(-1.0, index=np.arange(sample_size), columns=['deviate'])

        start = time.time()
        #transformed_plan = np.array(transformed_plan)

        # fill in deviation dataframe
        # transform transformed_plan to list
        #transformed_plan_xs = []
        #transformed_plan_ys = []
        #for i in range(0, transformed_plan.shape[0]):
        #    transformed_plan_xs.append(transformed_plan[i, 0])
        #    transformed_plan_ys.append(transformed_plan[i, 1])
        
        #for i in range(0, sample_size):
            #print('i = ', i)
            
            # transform the current local_plan to list
        #    local_plan = (local_plans.loc[local_plans['ID'] == i])
        #    local_plan = np.array(local_plan)
            #if i == 0:
            #    local_plan = np.array(self.local_plan)
            #else:
            #    local_plan = np.array(local_plan)
        #    if local_plan.shape[0] == 0:
        #        global_plans_deviation.iloc[i, 0] = 0.0
        #        continue
        #    local_plan_xs = []
        #    local_plan_ys = []
        #    for j in range(0, local_plan.shape[0]):
        #        local_plan_xs.append(local_plan[j, 0])
        #        local_plan_ys.append(local_plan[j, 1])
            
            # find deviation as a sum of minimal point-to-point differences
        #    diff_x = 0
        #    diff_y = 0
        #    devs = []
        #    for j in range(0, local_plan.shape[0]):
        #        local_diffs = []
        #        for k in range(0, len(transformed_plan)):
                    #diff_x = (local_plans_local[j, 0] - transformed_plan[k, 0]) ** 2
                    #diff_y = (local_plans_local[j, 1] - transformed_plan[k, 1]) ** 2
        #            diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
        #            diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
        #            diff = math.sqrt(diff_x + diff_y)
        #            local_diffs.append(diff)                        
        #        devs.append(min(local_diffs))   
        #    global_plans_deviation.iloc[i, 0] = sum(devs)
        #end = time.time()
        #print('classifier_fn: TARGET CALC RUNTIME = ', round(end-start,3))
        
        #if self.publish_explanation_coeffs_bool:
        #    self.original_deviation = global_plans_deviation.iloc[0, 0]
        #    #print('\noriginal_deviation = ', self.original_deviation)

        #cmd_vel_perturb['deviate'] = global_plans_deviation
        
        # return global_plans_deviation
        #return np.array(cmd_vel_perturb.iloc[:, 3:])
        return 0

    # create distances between the instance of interest and perturbations
    def create_distances(self):
        #start = time.time()
        # find distances
        # distance_metric = 'jaccard' - alternative distance metric
        distance_metric='cosine'
        self.distances = sklearn.metrics.pairwise_distances(
            self.data,
            np.array([[1] * self.data.shape[1]]).reshape(1, -1), #self.data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        #end = time.time()
        #print('DISTANCES CREATION RUNTIME = ', round(end-start,3))

def main():
    # ----------main-----------
    rospy.init_node('hixron', anonymous=False)

    # define hixron object
    hixron_obj = hixron()
    
    # call main to initialize subscribers
    hixron_obj.main_()
    
    # call explanation once to establish static map
    hixron_obj.first_call = True
    hixron_obj.test_explain()
    hixron_obj.first_call = False
    hixron_obj.test_explain()
    hixron_obj.test_explain()
    hixron_obj.test_explain()
    hixron_obj.test_explain()
    hixron_obj.test_explain()
    hixron_obj.test_explain()
    #print('BEFORE SLEEP')
    
    # sleep for 10s until Amar starts the video
    d = rospy.Duration(3, 0)
    rospy.sleep(d)
    #print('AFTER SLEEP')
    
    # send the goal pose to start navigation
    hixron_obj.send_goal_pose()
    
    #rate = rospy.Rate(0.15)
    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        #print('spinning')
        #rate.sleep()
        hixron_obj.test_explain()
        #rospy.spin()

main()