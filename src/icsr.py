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
import PIL
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from std_msgs.msg import String

PI = math.pi

def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

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

# get QSR value
def getIntrinsicQsrValue(angle):
    value = ''
    qsr_choice = 0    

    if qsr_choice == 0:
        if -PI/2 <= angle < PI/2:
            value += 'right'
        elif PI/2 <= angle < PI or -PI <= angle < -PI/2:
            value += 'left'

    elif qsr_choice == 1:
        if 0 <= angle < PI/2:
            value += 'left/front'
        elif PI/2 <= angle <= I:
            value += 'left/back'
        elif PI/2 <= angle < 0:
            value += 'right/front'
        elif PI <= angle < PI/2:
            value += 'right/back'

    return value

# hixron_subscriber class
class hixron(object):
    # constructor
    def __init__(self):
        # icsr vars
        self.humans_nearby = False
        self.pub_text_exp = rospy.Publisher('/textual_explanation', String, queue_size=10)
        self.text_exp = ''
        self.extrovert = False
        if self.extrovert:
            self.timing = 'immediately' # 'delayed'
            self.duration = 'short' # 'short', 'long'
            self.representation = 'textual' # 'textual', 'visual', 'textual-visual'
            self.detail_level = 'poor' # 'poor', 'rich'
        else:
            self.timing = 'delayed' # 'delayed'
            self.duration = 'long' # 'short', 'long'
            self.representation = 'textual-visual' # 'textual', 'visual', 'textual-visual'
            self.detail_level = 'rich' # 'poor', 'rich'
            self.introvert_publish_ctr = 4

        self.robot_offset = 9.0
        self.red_object_countdown_textual_only = -1
        self.red_object_value_textual_only = -1

        # hri and icsr and icra vars
        self.last_object_moved_ID = -1
        self.old_plan = Path()
        self.old_plan_bool = False
        self.red_object_countdown = -1
        self.red_object_value = -1
        self.humans = []
        self.human_blinking = False
        self.object_arrow_blinking = False

        # inflation
        self.inflation_radius = 0.275

        self.dirCurr = os.getcwd()

        # gazebo vars
        self.gazebo_names = []
        self.gazebo_poses = []
        self.gazebo_labels = []

        self.robot_pose_map = Pose()

        # plans' variables
        self.global_plan_current = Path() 
        self.global_plan_history = []
        self.globalPlan_goalPose_indices_history = []

        # goal pose
        self.goal_pose_current = Pose()
        self.goal_pose_history = []

        # ontology part
        self.scenario_name = 'icsr' #'scenario1', 'library', 'library_2', 'library_3'
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

        # global plan subscriber 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)
        self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)

        self.pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=1)
        self.pub_current_path = rospy.Publisher('/path_markers', MarkerArray, queue_size=1)
        self.pub_old_path = rospy.Publisher('/old_path_markers', MarkerArray, queue_size=1)
        self.pub_explanation_layer = rospy.Publisher("/explanation_layer", PointCloud2, queue_size=1)
   
        self.semantic_labels_marker_array = MarkerArray()
        self.current_path_marker_array = MarkerArray()
        self.old_path_marker_array = MarkerArray()

        # point_cloud variables
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]

        # header
        self.header = Header()

        # gazebo model states subscriber
        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_callback)

        # load gazebo tags
        self.gazebo_labels = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'gazebo_tags.csv')) 
        
        self.semantic_labels_marker_array = MarkerArray()
        self.semantic_labels_marker_array.markers = []
        for i in range(0, self.ontology.shape[0]):                
            x_map = self.ontology[i][12]
            y_map = self.ontology[i][13]
            
            # visualize orientations and semantic labels of known objects
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
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
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            #marker.frame_locked = False
            marker.text = self.ontology[i][1]
            marker.ns = "my_namespace"
            self.semantic_labels_marker_array.markers.append(marker)
        #self.pub_semantic_labels.publish(self.semantic_labels_marker_array)
  
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
        
        # create semantic data
        self.create_semantic_data()  

    # create semantic data
    def create_semantic_data(self):
        # update ontology
        self.update_ontology()

        # create semantic map
        self.create_global_semantic_map()
                    
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


    # publish common things
    def publish_map_humans_names(self):
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)

        # create the RGB explanation matrix of the same size as semantic map
        explanation_size_y = self.global_semantic_map_size[0]
        explanation_size_x = self.global_semantic_map_size[1]

        explanation_R = np.zeros((explanation_size_y, explanation_size_x))
        explanation_R[:,:] = 120 # free space
        explanation_R[global_semantic_map_complete_copy > 0] = 180.0 # obstacle
        explanation_G = copy.deepcopy(explanation_R)
        explanation_B = copy.deepcopy(explanation_R)
        output = (np.dstack((explanation_R,explanation_G,explanation_B))).astype(np.uint8)
        output = np.fliplr(output)

        z = 0.0
        a = 255                    
        points = []

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

        # publish
        self.header.frame_id = 'map'
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.pub_explanation_layer.publish(pc2)


        self.semantic_labels_marker_array.markers[7].pose.position.x = self.ontology[7][12]
        self.semantic_labels_marker_array.markers[8].pose.position.y = self.ontology[8][13]

        ID = self.ontology.shape[0]
        for human_pose in self.humans:
            x_map = human_pose.position.x
            y_map = human_pose.position.y
            
            # visualize orientations and semantic labels of humans
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = ID
            ID += 1
            marker.type = marker.TEXT_VIEW_FACING
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
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            #marker.frame_locked = False
            marker.text = "human"
            marker.ns = "my_namespace"
            if ID > len(self.semantic_labels_marker_array.markers):
                self.semantic_labels_marker_array.markers.append(marker)
            else:
                self.semantic_labels_marker_array.markers[ID - 1] = marker
        self.pub_semantic_labels.publish(self.semantic_labels_marker_array)

    # test whether explanation is needed
    def test_explain_icsr(self):
        #print('test_explain!')

        if self.first_call:
            self.create_semantic_data()
            self.publish_map_humans_names()
            return
        
        #'''
        if self.extrovert:
            # extrovert
            self.publish_map_humans_names()
            self.explain_textual_only_icsr()
            self.publish_textual_icsr()
        else:
            # introvert
            if self.introvert_publish_ctr == 4:
                self.explain_visual_icsr()
                self.explain_textual_icsr()

                self.publish_map_humans_names()
                self.publish_textual_empty()
            
            elif self.introvert_publish_ctr == 3:
                self.publish_map_humans_names()
                self.publish_textual_empty()

            if self.introvert_publish_ctr == 2:
                self.humans_nearby = False
                for human_pose in self.humans:
                    x_map = human_pose.position.x
                    y_map = human_pose.position.y    
                    distance_human_robot = math.sqrt((x_map - self.robot_pose_map.position.x)**2 + (y_map - self.robot_pose_map.position.y)**2)
                    if distance_human_robot < 2.0:
                        self.humans_nearby = True
                        break

                if self.humans_nearby == True:
                    self.publish_visual_icsr()
                    self.publish_textual_empty()

                else:
                    self.publish_visual_icsr()
                    self.publish_textual_icsr()

            elif self.introvert_publish_ctr == 1:
                self.introvert_publish_ctr = 5
               
            self.introvert_publish_ctr -= 1
        #'''
 

    def explain_visual_icsr(self):
        # STATIC PART        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)

        if len(np.unique(global_semantic_map_complete_copy)) != self.ontology.shape[0]+1:
            return

        color_shape_path_combination = [2,1,0]
        
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
        self.semantic_labels_marker_array.markers[7].pose.position.x = self.ontology[7][12]
        self.semantic_labels_marker_array.markers[8].pose.position.y = self.ontology[8][13]
        '''
        if shape_scheme == shape_schemes[1]:
            self.semantic_labels_marker_array.markers = []
            for i in range(7, 9):                
                x_map = self.ontology[i][12]
                y_map = self.ontology[i][13]
                
                # visualize orientations and semantic labels of known objects
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = i
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                #if self.ontology[i][0] in neighborhood_objects_IDs:
                #    marker.action = marker.ADD
                #else:
                #    marker.action = marker.DELETE
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
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.25
                #marker.frame_locked = False
                marker.text = self.ontology[i][1]
                marker.ns = "my_namespace"
                self.semantic_labels_marker_array.markers.append(marker)
        '''
                                         
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
            #if distance_human_robot > 2.0:
            #    marker.action = marker.DELETE
            #else:
            #    marker.action = marker.ADD
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
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            #marker.frame_locked = False
            marker.text = "human"
            marker.ns = "my_namespace"
            if ID > len(self.semantic_labels_marker_array.markers):
                self.semantic_labels_marker_array.markers.append(marker)
            else:
                self.semantic_labels_marker_array.markers[ID - 1] = marker

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
        self.output = PIL.Image.open(os.getcwd() + '/explanation.png').convert('RGB')        
        self.output = np.array(self.output)[:,:,:3].astype(np.uint8)

    def publish_visual_icsr(self):
        #points_start = time.time()
            
        z = 0.0
        a = 255                    
        points = []

        # draw layer
        size_1 = int(self.global_semantic_map_size[1])
        size_0 = int(self.global_semantic_map_size[0])
        for i in range(0, size_1):
            for j in range(0, size_0):
                x = self.global_semantic_map_origin_x + (size_1-i) * self.global_semantic_map_resolution
                y = self.global_semantic_map_origin_y + j * self.global_semantic_map_resolution
                r = int(self.output[j, i, 0])
                g = int(self.output[j, i, 1])
                b = int(self.output[j, i, 2])
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

    def publish_visual_empty(self):
        #points_start = time.time()
            
        z = 0.0
        a = 255                    
        points = []

        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)

        explanation_size_y = self.global_semantic_map_size[0]
        explanation_size_x = self.global_semantic_map_size[1]
        
        explanation_R = np.zeros((explanation_size_y, explanation_size_x))
        explanation_R[:,:] = 120 # free space
        explanation_R[global_semantic_map_complete_copy > 0] = 180.0 # obstacle
        explanation_G = copy.deepcopy(explanation_R)
        explanation_B = copy.deepcopy(explanation_R)

        output = (np.dstack((explanation_R,explanation_G,explanation_B))).astype(np.uint8)
        output = np.fliplr(output)

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

    def explain_textual_icsr(self):
        if self.red_object_countdown > 0:
            obj_pos_x = self.ontology[self.red_object_value-1][3]
            obj_pos_y = self.ontology[self.red_object_value-1][4]

            d_x = obj_pos_x - self.robot_pose_map.position.x
            d_y = obj_pos_y - self.robot_pose_map.position.y

            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(self.robot_pose_map.orientation.x,self.robot_pose_map.orientation.y,self.robot_pose_map.orientation.z,self.robot_pose_map.orientation.w)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = getIntrinsicQsrValue(angle)

            self.text_exp = 'I am deviating because the ' + self.ontology[self.red_object_value - 1][1] + ', which is to my ' + qsr_value + ', was moved.'
            return

        # define local explanation window around robot
        around_robot_size_x = 1.5
        around_robot_size_y = 1.5
        if self.extrovert == False:
            around_robot_size_x = 2.5
            around_robot_size_y = 2.5

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
        
        explanation_size_y = self.global_semantic_map_size[0]
        explanation_size_x = self.global_semantic_map_size[1]
        #print('(self.explanation_size_x,self.explanation_size_y)',(self.explanation_size_y,self.explanation_size_x))

        x_min_pixel = max(0, x_min_pixel)
        x_max_pixel = min(explanation_size_x - 1, x_max_pixel)
        y_min_pixel = max(0, y_min_pixel)
        y_max_pixel = min(explanation_size_y - 1, y_max_pixel)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)
        neighborhood_objects_IDs = np.unique(global_semantic_map_complete_copy[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel])
        if 0 in neighborhood_objects_IDs:
            neighborhood_objects_IDs = neighborhood_objects_IDs[1:]
        neighborhood_objects_IDs = [int(item) for item in neighborhood_objects_IDs]
        #print('neighborhood_objects_IDs =', neighborhood_objects_IDs)

        neighborhood_objects_distances = []
        neighborhood_objects_spatials = []
        neighborhood_objects_names = []
        for ID in neighborhood_objects_IDs:
            obj_pos_x = self.ontology[ID-1][3]
            obj_pos_y = self.ontology[ID-1][4]

            d_x = obj_pos_x - robot_pose.position.x
            d_y = obj_pos_y - robot_pose.position.y

            dist = math.sqrt(d_x**2 + d_y**2)
            neighborhood_objects_distances.append(dist)

            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(robot_pose.orientation.x,robot_pose.orientation.y,robot_pose.orientation.z,robot_pose.orientation.w)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = getIntrinsicQsrValue(angle)
            neighborhood_objects_spatials.append(qsr_value)
            neighborhood_objects_names.append(self.ontology[ID-1][1])
            #print('tiago passes ' + qsr_value + ' of the ' + self.ontology[ID-1][1])
        #print(len(neighborhood_objects_spatials))
            
        # FORM THE TEXTUAL EXPLANATION
        if len(neighborhood_objects_names) > 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])

            if len(left_objects) != 0 and len(right_objects) != 0:
                self.text_exp = 'I am passing by '

                if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left'
                elif len(left_objects) == 2:
                    self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left'
                elif len(left_objects) == 1:
                    self.text_exp += left_objects[0] + ' to my left'

                self.text_exp += ' and '

                if len(right_objects) > 2:
                    for i in range(0, len(right_objects) - 2):    
                        self.text_exp += right_objects[i] + ', '
                    i += 1
                    self.text_exp += right_objects[i] + ' and '
                    i += 1
                    self.text_exp += right_objects[i] + ' to my right.'
                elif len(right_objects) == 2:
                    self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                elif len(right_objects) == 1:
                    self.text_exp += right_objects[0] + ' to my right.'
            else:
                self.text_exp = 'I am passing by '

                if len(left_objects) != 0:
                    if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left.'
                    elif len(left_objects) == 2:
                        self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
                    elif len(left_objects) == 1:
                        self.text_exp += left_objects[0] + ' to my left.'

                if len(right_objects) != 0:
                    if len(right_objects) > 2:
                        for i in range(0, len(right_objects) - 2):    
                            self.text_exp += right_objects[i] + ', '
                        i += 1
                        self.text_exp += right_objects[i] + ' and '
                        i += 1
                        self.text_exp += right_objects[i] + ' to my right.'
                    elif len(right_objects) == 2:
                        self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                    elif len(right_objects) == 1:
                        self.text_exp += right_objects[0] + ' to my right.'
        
        elif len(neighborhood_objects_names) == 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])
            
            if len(left_objects) != 0 and len(right_objects) != 0:
                self.text_exp = 'I am passing by '
                self.text_exp += left_objects[0] + ' to my left and ' + right_objects[1] + ' to my right.'
            elif len(left_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
            elif len(right_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
        
        elif len(neighborhood_objects_names) == 1:
            self.text_exp = 'I am passing by '    
            self.text_exp += neighborhood_objects_names[0] + ' to my ' + neighborhood_objects_spatials[0] + '.'
        #print(self.text_exp)

    def explain_textual_only_icsr(self):
        if self.red_object_countdown_textual_only > 0:
            self.red_object_countdown_textual_only -= 1
            return
        elif self.red_object_countdown_textual_only == 0:
            self.red_object_countdown_textual_only = -1
            self.red_object_value_textual_only = -1

                # TEST THE DEVIATION
        if len(self.global_plan_history) > 1:
            # test if there is deviation between current and previous
            deviation_between_global_plans_textual = False
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
                    deviation_between_global_plans_textual = True

            # check if the last two global plans have the same goal pose
            same_goal_pose = False
            if len(self.globalPlan_goalPose_indices_history_hold) > 1:
                if self.globalPlan_goalPose_indices_history_hold[-1][1] == self.globalPlan_goalPose_indices_history_hold[-2][1]:
                    same_goal_pose = True

            # if deviation happened and some object was moved
            if deviation_between_global_plans_textual and same_goal_pose:
                #print('TESTIRA se moguca devijacija')
                if self.last_object_moved_ID > 0 and self.red_object_countdown == -1: #self.last_object_moved_ID in neighborhood_objects_IDs
                    # define the red object
                    self.red_object_value_textual_only = copy.deepcopy(self.last_object_moved_ID)
                    self.red_object_countdown_textual_only = 12
                    self.last_object_moved_ID = -1

                    obj_pos_x = self.ontology[self.red_object_value_textual_only-1][3]
                    obj_pos_y = self.ontology[self.red_object_value_textual_only-1][4]

                    d_x = obj_pos_x - self.robot_pose_map.position.x
                    d_y = obj_pos_y - self.robot_pose_map.position.y

                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(self.robot_pose_map.orientation.x,self.robot_pose_map.orientation.y,self.robot_pose_map.orientation.z,self.robot_pose_map.orientation.w)
                    angle = angle - angle_ref
                    if angle >= PI:
                        angle -= 2*PI
                    elif angle < -PI:
                        angle += 2*PI
                    qsr_value = getIntrinsicQsrValue(angle)

                    self.text_exp = 'I am deviating because the ' + self.ontology[self.red_object_value_textual_only - 1][1] + ', which is to my ' + qsr_value + ', was moved.'
                    return    

        # define local explanation window around robot
        around_robot_size_x = 1.5
        around_robot_size_y = 1.5
        if self.extrovert == False:
            around_robot_size_x = 2.5
            around_robot_size_y = 2.5

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
        
        explanation_size_y = self.global_semantic_map_size[0]
        explanation_size_x = self.global_semantic_map_size[1]
        #print('(self.explanation_size_x,self.explanation_size_y)',(self.explanation_size_y,self.explanation_size_x))

        x_min_pixel = max(0, x_min_pixel)
        x_max_pixel = min(explanation_size_x - 1, x_max_pixel)
        y_min_pixel = max(0, y_min_pixel)
        y_max_pixel = min(explanation_size_y - 1, y_max_pixel)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)
        neighborhood_objects_IDs = np.unique(global_semantic_map_complete_copy[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel])
        if 0 in neighborhood_objects_IDs:
            neighborhood_objects_IDs = neighborhood_objects_IDs[1:]
        neighborhood_objects_IDs = [int(item) for item in neighborhood_objects_IDs]
        #print('neighborhood_objects_IDs =', neighborhood_objects_IDs)

        neighborhood_objects_distances = []
        neighborhood_objects_spatials = []
        neighborhood_objects_names = []
        for ID in neighborhood_objects_IDs:
            obj_pos_x = self.ontology[ID-1][3]
            obj_pos_y = self.ontology[ID-1][4]

            d_x = obj_pos_x - robot_pose.position.x
            d_y = obj_pos_y - robot_pose.position.y

            dist = math.sqrt(d_x**2 + d_y**2)
            neighborhood_objects_distances.append(dist)

            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(robot_pose.orientation.x,robot_pose.orientation.y,robot_pose.orientation.z,robot_pose.orientation.w)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = getIntrinsicQsrValue(angle)
            neighborhood_objects_spatials.append(qsr_value)
            neighborhood_objects_names.append(self.ontology[ID-1][1])
            #print('tiago passes ' + qsr_value + ' of the ' + self.ontology[ID-1][1])
        #print(len(neighborhood_objects_spatials))
            
        # FORM THE TEXTUAL EXPLANATION
        if len(neighborhood_objects_names) > 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])

            if len(left_objects) != 0 and len(right_objects) != 0:
                self.text_exp = 'I am passing by '

                if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left'
                elif len(left_objects) == 2:
                    self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left'
                elif len(left_objects) == 1:
                    self.text_exp += left_objects[0] + ' to my left'

                self.text_exp += ' and '

                if len(right_objects) > 2:
                    for i in range(0, len(right_objects) - 2):    
                        self.text_exp += right_objects[i] + ', '
                    i += 1
                    self.text_exp += right_objects[i] + ' and '
                    i += 1
                    self.text_exp += right_objects[i] + ' to my right.'
                elif len(right_objects) == 2:
                    self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                elif len(right_objects) == 1:
                    self.text_exp += right_objects[0] + ' to my right.'
            else:
                self.text_exp = 'I am passing by '

                if len(left_objects) != 0:
                    if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left.'
                    elif len(left_objects) == 2:
                        self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
                    elif len(left_objects) == 1:
                        self.text_exp += left_objects[0] + ' to my left.'

                if len(right_objects) != 0:
                    if len(right_objects) > 2:
                        for i in range(0, len(right_objects) - 2):    
                            self.text_exp += right_objects[i] + ', '
                        i += 1
                        self.text_exp += right_objects[i] + ' and '
                        i += 1
                        self.text_exp += right_objects[i] + ' to my right.'
                    elif len(right_objects) == 2:
                        self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                    elif len(right_objects) == 1:
                        self.text_exp += right_objects[0] + ' to my right.'
        
        elif len(neighborhood_objects_names) == 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])
            
            if len(left_objects) != 0 and len(right_objects) != 0:
                self.text_exp = 'I am passing by '
                self.text_exp += left_objects[0] + ' to my left and ' + right_objects[1] + ' to my right.'
            elif len(left_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
            elif len(right_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
        
        elif len(neighborhood_objects_names) == 1:
            self.text_exp = 'I am passing by '    
            self.text_exp += neighborhood_objects_names[0] + ' to my ' + neighborhood_objects_spatials[0] + '.'
        #print(self.text_exp)
            
    def publish_textual_icsr(self):
        self.pub_text_exp.publish(self.text_exp)

    def publish_textual_empty(self):
        text_exp = ''
        self.pub_text_exp.publish(text_exp)

def main():
    # ----------main-----------
    rospy.init_node('hixron', anonymous=False)

    # define hixron object
    hixron_obj = hixron()
    
    # call explanation once to establish static map
    hixron_obj.first_call = True
    hixron_obj.test_explain_icsr()
    hixron_obj.first_call = False
    #hixron_obj.test_explain_icsr()
    #hixron_obj.test_explain_icsr()
    #hixron_obj.test_explain_icsr()
    #hixron_obj.test_explain_icsr()
    #hixron_obj.test_explain_icsr()
    #hixron_obj.test_explain_icsr()
    #print('BEFORE SLEEP')
    
    # sleep for 10s until Amar starts the video
    d = rospy.Duration(1, 0)
    rospy.sleep(d)
    #print('AFTER SLEEP')
    
    # send the goal pose to start navigation
    hixron_obj.send_goal_pose()
    
    #rate = rospy.Rate(0.15)
    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        #print('spinning')
        #rate.sleep()
        hixron_obj.test_explain_icsr()
        #rospy.spin()
        
main()