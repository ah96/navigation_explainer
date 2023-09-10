#!/usr/bin/env python3

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
import rospy
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
import copy
import os
from gazebo_msgs.msg import ModelStates, ModelState
import math
from skimage.measure import regionprops
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import cv2
from sensor_msgs import point_cloud2
import struct
import math
import PIL
from visualization_msgs.msg import Marker, MarkerArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from std_msgs.msg import String

PI = math.pi

# convert euler angles to orientation quaternion
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


# hixron_subscriber class
class hixron(object):
    # constructor
    def __init__(self):
        self.explanation_window = 2.0

        self.human_blinking = True

        self.humans_nearby = False

        self.N_humans = 6

        self.moved_object_distance_threshold = 1.5

        self.failure_started = False
        self.failure_ended = False
        
        # visual explanation vars
        self.moved_object_countdown_textual_only = -1
        self.moved_object_value_textual_only = -1 
        self.last_object_moved_ID = -1
        self.last_object_moved_ID_for_publishing = -1
        self.old_plan = Path()
        self.old_plan_bool = False
        self.moved_object_countdown = -1
        self.moved_object_value = -1
        self.humans = []
        self.human_blinking = False
        self.object_arrow_blinking = False
        self.visual_explanation = []
        self.visual_explanation_resolution = 0.05


        # textual explanation vars
        self.pub_text_exp = rospy.Publisher('/textual_explanation', String, queue_size=10)
        self.text_exp = ''

        # point cloud vars
        self.goal_marker_array = MarkerArray()
        self.semantic_labels_marker_array = MarkerArray()
        self.current_path_marker_array = MarkerArray()
        self.old_path_marker_array = MarkerArray()
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]
        self.header = Header()

        # inflation
        self.inflation_radius = 0.275

        # directory
        self.dirCurr = os.getcwd()

        # robot vars
        self.robot_pose_map = Pose()
        self.robot_offset_x = 9.0
        self.robot_offset_y = 9.0
        
        # plans' variables
        self.global_plan_current = Path()
        self.global_plan_previous = Path()
        self.global_plan_current_goal = Pose()
        self.global_plan_previous_goal = Pose() 
        #self.global_plan_history = []
        #self.globalPlan_goalPose_indices_history = []
        self.deviation_threshold = 1.0
        self.global_plan_ctr = 0
        self.deviation = False
        self.same_goal_pose = True

        # goal pose
        self.init_goal()

        # ontology part
        self.scenario_name = 'icra_melodic'
        # load ontology
        self.ontology = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'ontology.csv')
        self.ontology = np.array(self.ontology)
        self.ont_len = self.ontology.shape[0]
        # apply map-world offset
        for i in range(self.ontology.shape[0]):
            self.ontology[i, 3] += self.robot_offset_x
            self.ontology[i, 4] -= self.robot_offset_y

            self.ontology[i, 12] += self.robot_offset_x
            self.ontology[i, 13] -= self.robot_offset_y

        # gazebo vars
        self.gazebo_names = []
        self.gazebo_poses = []
        self.gazebo_labels = []
        # load gazebo tags
        self.gazebo_labels = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'gazebo_tags.csv')) 

        # load global semantic map info
        self.global_semantic_map_info = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'map_info.csv')) 
        # global semantic map vars
        self.global_semantic_map_origin_x = float(self.global_semantic_map_info[0,4])
        self.global_semantic_map_origin_x += self.robot_offset_x 
        self.global_semantic_map_origin_y = float(self.global_semantic_map_info[0,5])
        self.global_semantic_map_origin_y -= self.robot_offset_y 
        self.global_semantic_map_resolution = float(self.global_semantic_map_info[0,1])
        self.global_semantic_map_size = [int(self.global_semantic_map_info[0,3]), int(self.global_semantic_map_info[0,2])]
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #print(self.global_semantic_map_origin_x, self.global_semantic_map_origin_y, self.global_semantic_map_resolution, self.global_semantic_map_size)
        self.global_semantic_map_complete = []

        # initialize objects which will be moved
        self.init_objects_to_move()

        # initialize an array of semantic labels (static map)
        self.init_semantic_labels()

        # subscribers 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)
        #self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)
        #self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_callback)

        # publishers
        self.pub_goal_pose = rospy.Publisher('/goal_pose', MarkerArray, queue_size=1)
        self.pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=1)
        self.pub_current_path = rospy.Publisher('/path_markers', MarkerArray, queue_size=1)
        self.pub_old_path = rospy.Publisher('/old_path_markers', MarkerArray, queue_size=1)
        self.pub_explanation_layer = rospy.Publisher("/explanation_layer", PointCloud2, queue_size=1)
        self.pub_move = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.pub_semantic_map = rospy.Publisher("/semantic_map", PointCloud2, queue_size=1)

    # intialize navigational goal and stop pose
    def init_goal(self):
        # Creates a goal to send to the action server.
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.seq = 0
        self.goal.target_pose.header.stamp.secs = 0
        self.goal.target_pose.header.stamp.nsecs = 0
        self.goal.target_pose.header.frame_id = "map"

        self.goal.target_pose.pose.position.x = 0.603
        self.goal.target_pose.pose.position.y = -6.632
        self.goal.target_pose.pose.position.z = 0.0

        self.goal.target_pose.pose.orientation.x = 0.0
        self.goal.target_pose.pose.orientation.y = 0.0
        self.goal.target_pose.pose.orientation.z = -0.71
        self.goal.target_pose.pose.orientation.w = 0.70

        # Creates a stop goal to send to the action server.
        self.stop_goal = MoveBaseGoal()
        self.stop_goal.target_pose.header.seq = 0
        self.stop_goal.target_pose.header.stamp.secs = 0
        self.stop_goal.target_pose.header.stamp.nsecs = 0
        self.stop_goal.target_pose.header.frame_id = "map"

        self.stop_goal.target_pose.pose.position.x = 0.603
        self.stop_goal.target_pose.pose.position.y = -6.632
        self.stop_goal.target_pose.pose.position.z = 0.0

        self.stop_goal.target_pose.pose.orientation.x = 0.0
        self.stop_goal.target_pose.pose.orientation.y = 0.0
        self.stop_goal.target_pose.pose.orientation.z = -0.71
        self.stop_goal.target_pose.pose.orientation.w = 0.70
   
    # intialize objects which will be moved
    def init_objects_to_move(self):
        self.chair_2_moved_moved = False

        self.chair_2_moved_state = ModelState()
        self.chair_2_moved_state.model_name = 'chair_2'
        self.chair_2_moved_state.reference_frame = 'world'
        # pose
        self.chair_2_moved_state.pose.position.x = -8.5
        self.chair_2_moved_state.pose.position.y = 7.2
        self.chair_2_moved_state.pose.position.z = 0
        quaternion = euler_to_quaternion(-1.559, 0, 0)
        self.chair_2_moved_state.pose.orientation.x = quaternion[0]
        self.chair_2_moved_state.pose.orientation.y = quaternion[1]
        self.chair_2_moved_state.pose.orientation.z = quaternion[2]
        self.chair_2_moved_state.pose.orientation.w = quaternion[3]
        # twist
        self.chair_2_moved_state.twist.linear.x = 0
        self.chair_2_moved_state.twist.linear.y = 0
        self.chair_2_moved_state.twist.linear.z = 0
        self.chair_2_moved_state.twist.angular.x = 0
        self.chair_2_moved_state.twist.angular.y = 0
        self.chair_2_moved_state.twist.angular.z = 0


        self.human_4_moved_moved = False

        self.human_4_moved_state = ModelState()
        self.human_4_moved_state.model_name = 'human_4'
        self.human_4_moved_state.reference_frame = 'world'
        # pose
        self.human_4_moved_state.pose.position.x = -6.22
        self.human_4_moved_state.pose.position.y = 4.16
        self.human_4_moved_state.pose.position.z = 0
        quaternion = euler_to_quaternion(-2.4, 0, 0)
        self.human_4_moved_state.pose.orientation.x = quaternion[0]
        self.human_4_moved_state.pose.orientation.y = quaternion[1]
        self.human_4_moved_state.pose.orientation.z = quaternion[2]
        self.human_4_moved_state.pose.orientation.w = quaternion[3]
        # twist
        self.human_4_moved_state.twist.linear.x = 0
        self.human_4_moved_state.twist.linear.y = 0
        self.human_4_moved_state.twist.linear.z = 0
        self.human_4_moved_state.twist.angular.x = 0
        self.human_4_moved_state.twist.angular.y = 0
        self.human_4_moved_state.twist.angular.z = 0


        self.human_4_original_moved = False

        self.human_4_original_state = ModelState()
        self.human_4_original_state.model_name = 'human_4'
        self.human_4_original_state.reference_frame = 'world'
        # pose
        self.human_4_original_state.pose.position.x = -5.7
        self.human_4_original_state.pose.position.y = 3.53
        self.human_4_original_state.pose.position.z = 0
        quaternion = euler_to_quaternion(-2.4, 0, 0)
        self.human_4_original_state.pose.orientation.x = quaternion[0]
        self.human_4_original_state.pose.orientation.y = quaternion[1]
        self.human_4_original_state.pose.orientation.z = quaternion[2]
        self.human_4_original_state.pose.orientation.w = quaternion[3]
        # twist
        self.human_4_original_state.twist.linear.x = 0
        self.human_4_original_state.twist.linear.y = 0
        self.human_4_original_state.twist.linear.z = 0
        self.human_4_original_state.twist.angular.x = 0
        self.human_4_original_state.twist.angular.y = 0
        self.human_4_original_state.twist.angular.z = 0
    
    # initialize an array of semantic labels (static map)
    def init_semantic_labels(self):
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
            marker.text = self.ontology[i][2]
            marker.ns = "my_namespace"
            self.semantic_labels_marker_array.markers.append(marker)

    # visualize goal pose
    def visualize_goal_pose(self):
        self.goal_marker_array.markers = []

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.goal.target_pose.pose.position.x
        marker.pose.position.y = self.goal.target_pose.pose.position.y
        marker.pose.position.z = 0.2 #self.goal.target_pose.pose.position.z
        marker.pose.orientation.x = self.goal.target_pose.pose.orientation.x
        marker.pose.orientation.y = self.goal.target_pose.pose.orientation.y
        marker.pose.orientation.z = self.goal.target_pose.pose.orientation.z
        marker.pose.orientation.w = self.goal.target_pose.pose.orientation.w
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.75
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        #marker.frame_locked = False
        marker.ns = "my_namespace"
        self.goal_marker_array.markers.append(marker)

        self.pub_goal_pose.publish(self.goal_marker_array)

    # send goal pose
    def send_goal_pose(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
  
        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()
   
        # Sends the goal to the action server.
        client.send_goal(self.goal)

    # send stop pose
    def send_stop_pose(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
  
        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()
   
        # Creates a goal to send to the action server.

        self.stop_goal.target_pose.pose.position.x = self.robot_pose_map.position.x
        self.stop_goal.target_pose.pose.position.y = self.robot_pose_map.position.y
        self.stop_goal.target_pose.pose.position.z = 0.0

        # Sends the goal to the action server.
        client.send_goal(self.stop_goal)

    # send goal pose
    def cancel_goal_pose(self):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
  
        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()
   
        # Cancel the goal.
        client.cancel_goal()

    # amcl (global) pose callback
    def amcl_callback(self, msg):
        #print('\namcl_callback')

        self.robot_pose_map = msg.pose.pose

    # move objects
    def move_objects(self):
        if self.chair_2_moved_moved == False:
            x = self.chair_2_moved_state.pose.position.x + self.robot_offset_x
            y = self.chair_2_moved_state.pose.position.y - self.robot_offset_y

            dx = self.robot_pose_map.position.x - x
            dy = self.robot_pose_map.position.y - y

            dist = math.sqrt(dx**2 + dy**2)

            if dist < self.moved_object_distance_threshold:
                self.pub_move.publish(self.chair_2_moved_state)
                self.pub_move.publish(self.chair_2_moved_state)
                self.pub_move.publish(self.chair_2_moved_state)
                self.pub_move.publish(self.chair_2_moved_state)
                self.pub_move.publish(self.chair_2_moved_state)

                self.chair_2_moved_moved = True

                self.update_ontology_quick(2)
                
                self.create_global_semantic_map()

                self.prepare_global_semantic_map_for_publishing()
                self.publish_global_semantic_map()
                
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()

        if self.human_4_moved_moved == False:
            x = self.human_4_moved_state.pose.position.x + self.robot_offset_x
            y = self.human_4_moved_state.pose.position.y - self.robot_offset_y

            dx = self.robot_pose_map.position.x - x
            dy = self.robot_pose_map.position.y - y

            dist = math.sqrt(dx**2 + dy**2)

            if dist < self.moved_object_distance_threshold:
                self.pub_move.publish(self.human_4_moved_state)
                self.pub_move.publish(self.human_4_moved_state)
                self.pub_move.publish(self.human_4_moved_state)
                self.pub_move.publish(self.human_4_moved_state)
                self.pub_move.publish(self.human_4_moved_state)

                self.human_4_moved_moved = True

                self.update_ontology_quick(35)
                
                self.create_global_semantic_map()

                self.prepare_global_semantic_map_for_publishing()
                self.publish_global_semantic_map()
                
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()

                #self.cancel_goal_pose()
                self.send_stop_pose()
                self.send_stop_pose()
                self.send_stop_pose()
                self.send_stop_pose()
                self.send_stop_pose()
                self.failure_start = time.time()
                self.failure_started = True

        if self.human_4_moved_moved == True and self.failure_ended == False:
            self.failure_end = time.time()
            self.failure_time = self.failure_end - self.failure_start
            #print(self.failure_time)

            if self.failure_time > 8.0 and self.human_4_original_moved == False:
                self.pub_move.publish(self.human_4_original_state)
                self.pub_move.publish(self.human_4_original_state)
                self.pub_move.publish(self.human_4_original_state)
                self.pub_move.publish(self.human_4_original_state)
                self.pub_move.publish(self.human_4_original_state)
                
                self.human_4_original_moved = True

                self.update_ontology_quick(35)
                
                self.create_global_semantic_map()

                self.prepare_global_semantic_map_for_publishing()
                self.publish_global_semantic_map()
                
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()
                #self.publish_semantic_labels_all()

                self.send_goal_pose()
                self.failure_started = False
                self.failure_ended = True

            #if self.failure_time > 9.0 and self.failure_ended == False:
            #    self.send_goal_pose()
            #    self.failure_started = False
            #    self.failure_ended = True

    # update ontology quick approach
    def update_ontology_quick(self, moved_ID):
        # heuristics - we know which objects could change their positions
        i = moved_ID - 1

        obj_x_current = copy.deepcopy(self.ontology[i][3])
        obj_y_current = copy.deepcopy(self.ontology[i][4])

        obj_x_new = 0
        obj_y_new = 0
        if i == 1:
            obj_x_new = self.chair_2_moved_state.pose.position.x + self.robot_offset_x
            obj_y_new = self.chair_2_moved_state.pose.position.y - self.robot_offset_y

            self.ontology[i][3] = obj_x_new
            self.ontology[i][4] = obj_y_new
            self.last_object_moved_ID = self.ontology[i][0]
            self.last_object_moved_ID_for_publishing = self.ontology[i][0]
            self.ontology[i][12] += obj_x_new - obj_x_current
            self.ontology[i][13] += obj_y_new - obj_y_current
            if self.ontology[i][11] == 'y':
                self.ontology[i][11] = 'n'

        elif i == 34:
            if self.human_4_original_moved == False:
                obj_x_new = self.human_4_moved_state.pose.position.x + self.robot_offset_x
                obj_y_new = self.human_4_moved_state.pose.position.y - self.robot_offset_y

                self.ontology[i][3] = obj_x_new
                self.ontology[i][4] = obj_y_new
                self.last_object_moved_ID = self.ontology[i][0]
                self.last_object_moved_ID_for_publishing = self.ontology[i][0]
                self.ontology[i][12] += obj_x_new - obj_x_current - 0.3
                self.ontology[i][13] += obj_y_new - obj_y_current

            elif self.human_4_original_moved == True:
                obj_x_new = self.human_4_original_state.pose.position.x + self.robot_offset_x
                obj_y_new = self.human_4_original_state.pose.position.y - self.robot_offset_y

                self.ontology[i][3] = obj_x_new
                self.ontology[i][4] = obj_y_new
                self.last_object_moved_ID = self.ontology[i][0]
                self.last_object_moved_ID_for_publishing = self.ontology[i][0]
                self.ontology[i][12] += obj_x_new - obj_x_current + 0.3
                self.ontology[i][13] += obj_y_new - obj_y_current
                #print('DOING!!!!!')

        # global plan callback
    
    # global plan callback
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')

        try:
            self.global_plan_ctr += 1
            
            # save global plan to class vars
            self.global_plan_current = copy.deepcopy(msg)
            #self.global_plan_history.append(self.global_plan_current)
            #self.globalPlan_goalPose_indices_history.append([len(self.global_plan_history), len(self.goal_pose_history)])

            if self.global_plan_ctr > 1: 
                # calculate deivation
                ind_current = int(0.5 * (len(self.global_plan_current.poses) - 1))
                ind_previous = int(0.5 * (len(self.global_plan_previous.poses) - 1))
                    
                dev = 0
                dev_x = self.global_plan_current.poses[ind_current].pose.position.x - self.global_plan_previous.poses[ind_previous].pose.position.x
                dev_y = self.global_plan_current.poses[ind_current].pose.position.y - self.global_plan_previous.poses[ind_previous].pose.position.y
                dev = dev_x**2 + dev_y**2
                dev = math.sqrt(dev)
                
                if dev > self.deviation_threshold:
                    #print('DEVIATION HAPPENED!!!')
                    self.old_plan = copy.deepcopy(self.global_plan_previous)
                    self.deviation = True

            # save global plan to class vars
            self.global_plan_previous = copy.deepcopy(msg)
        except:
            pass

    # create global semantic map
    def create_global_semantic_map(self):
        #print('\ncreate_global_semantic_map')
        #start = time.time()

        # do not update the map, if none object is moved 
        #if self.last_object_moved_ID == -1 and len(np.unique(self.global_semantic_map)) != 1:
        #    return -1
        
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #self.global_semantic_map_inflated = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
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

        return 0

    # prepare global semantic map for publishing
    def prepare_global_semantic_map_for_publishing(self):
        #print('prepare_global_semantic_map_for_publishing')
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

        #from PIL import Image
        #im = Image.fromarray(output)
        #im.save("semantic_map.png")

        z = 0.0
        a = 255                    
        self.points_semantic_map = []

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
                self.points_semantic_map.append(pt)
    
    # publish global semantic map
    def publish_global_semantic_map(self):
        #print('publish_global_semantic_map')
        # publish
        self.header.frame_id = 'map'
        pc2 = point_cloud2.create_cloud(self.header, self.fields, self.points_semantic_map)
        pc2.header.stamp = rospy.Time.now()
        self.pub_semantic_map.publish(pc2)

    # publish all semantic labels
    def publish_semantic_labels_all(self):
        # update poses of possible objects
        self.semantic_labels_marker_array.markers[1].pose.position.x = self.ontology[1][12]
        self.semantic_labels_marker_array.markers[1].pose.position.y = self.ontology[1][13]

        self.semantic_labels_marker_array.markers[34].pose.position.x = self.ontology[34][12]
        self.semantic_labels_marker_array.markers[34].pose.position.y = self.ontology[34][13]
        
        self.pub_semantic_labels.publish(self.semantic_labels_marker_array)

    # setup func
    def setup(self):
        self.create_global_semantic_map()
        
        self.prepare_global_semantic_map_for_publishing()
        
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        self.publish_global_semantic_map()
        
        #self.publish_semantic_labels_all()

        return

    # test whether explanation is needed
    def test_explain(self):
        self.create_visual_explanation()
        self.publish_visual_explanation()

    # create visual explanation
    def create_visual_explanation(self):
        # STATIC PART        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)

        # define local explanation window around robot
        around_robot_size_x = self.explanation_window
        around_robot_size_y = self.explanation_window

        # get the semantic map size
        semantic_map_size_y = self.global_semantic_map_size[0]
        semantic_map_size_x = self.global_semantic_map_size[1]

        # find the objects/obstacles in the robot's local neighbourhood
        robot_pose = copy.deepcopy(self.robot_pose_map)
        x_min = robot_pose.position.x - around_robot_size_x
        x_max = robot_pose.position.x + around_robot_size_x
        y_min = robot_pose.position.y - around_robot_size_y
        y_max = robot_pose.position.y + around_robot_size_y
        #print('(x_min,x_max,y_min,y_max) = ', (x_min,x_max,y_min,y_max))

        x_min_pixel = int((x_min - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)        
        x_max_pixel = int((x_max - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
        y_min_pixel = int((y_min - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        y_max_pixel = int((y_max - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        x_min_pixel = max(0, x_min_pixel)
        x_max_pixel = min(semantic_map_size_x - 1, x_max_pixel)
        y_min_pixel = max(0, y_min_pixel)
        y_max_pixel = min(semantic_map_size_y - 1, y_max_pixel)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))

        self.neighborhood_objects_IDs = np.unique(global_semantic_map_complete_copy[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel])
        if 0 in  self.neighborhood_objects_IDs:
             self.neighborhood_objects_IDs =  self.neighborhood_objects_IDs[1:]
        self.neighborhood_objects_IDs = [int(item) for item in  self.neighborhood_objects_IDs]
        #print('self.neighborhood_objects_IDs =', self.neighborhood_objects_IDs)

        # create the RGB explanation matrix of the same size as semantic map
        #print('(semantic_map_size_x,semantic_map_size_y)',(semantic_map_size_y,semantic_map_size_x))
        explanation_R = np.zeros((semantic_map_size_y, semantic_map_size_x))
        explanation_R[:,:] = 120 # free space
        explanation_R[global_semantic_map_complete_copy > 0] = 180.0 # obstacle
        explanation_G = copy.deepcopy(explanation_R)
        explanation_B = copy.deepcopy(explanation_R)
        self.vis_exp_coords = (y_min_pixel, y_max_pixel, x_min_pixel, x_max_pixel)

        # OBSTACLE COLORING using a yellow-green-red scheme
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

        # DEVIATION PART
        if self.deviation and self.last_object_moved_ID == 2:
            # define the moved object
            self.moved_object_value = copy.deepcopy(self.last_object_moved_ID)
            self.moved_object_countdown = 40
            
            # reset this variable
            self.last_object_moved_ID = -1
            self.deviation = False

            self.visualize_old_plan()

        # COLOR OBJECT THAT CAUSED DEVIATION
        if self.moved_object_countdown > 0:
            #print('self.moved_object_countdown = ', self.moved_object_countdown)
            #RGB_val = [201,9,9]
            RGB_val = [255,0,0]
            #start = time.time()
            explanation_R[global_semantic_map_complete_copy == self.moved_object_value] = RGB_val[0]
            explanation_G[global_semantic_map_complete_copy == self.moved_object_value] = RGB_val[1]
            explanation_B[global_semantic_map_complete_copy == self.moved_object_value] = RGB_val[2]
            #end = time.time()
            #print('DURATION = ', end-start)


        # FORM THE EXPLANATION IMAGE                  
        explanation = (np.dstack((explanation_R,explanation_G,explanation_B))).astype(np.uint8)

        fig = plt.figure(frameon=True)
        w = 0.01 * semantic_map_size_x
        h = 0.01 * semantic_map_size_y
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(np.fliplr(explanation)) #np.flip(explanation))#.astype(np.uint8))

        # VISUALIZE ARROWS AROUND MOVABLE OBJECTS USING MATPLOTLIB
        for i in range(0, 9):
            # if it is a movable object and in the robot's neighborhood
            if self.ontology[i][0] in self.neighborhood_objects_IDs or self.ontology[i][0] == self.moved_object_value:
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
                        xs_plot.append(semantic_map_size_x - x_pixel + dx - 1)
                        ys_plot.append(y_pixel - 1)
                        arrows.append('>')
                    elif self.ontology[i][10] == 'l':
                        xs_plot.append(semantic_map_size_x - x_pixel - dx - 1)
                        ys_plot.append(y_pixel - 1)
                        arrows.append('<')
                    elif self.ontology[i][10] == 't':
                        xs_plot.append(semantic_map_size_x - x_pixel - 2)
                        ys_plot.append(y_pixel - dy - 2)
                        arrows.append('^')
                    elif self.ontology[i][10] == 'b':
                        xs_plot.append(semantic_map_size_x - x_pixel - 2)
                        ys_plot.append(y_pixel + dy - 1)
                        arrows.append('v')
                # if object is not under the table
                elif self.ontology[i][11] == 'n' or self.ontology[i][11] == 'na':
                    xs_plot.append(semantic_map_size_x - x_pixel + dx - 1)
                    ys_plot.append(y_pixel - 1)
                    arrows.append('>')
                    xs_plot.append(semantic_map_size_x - x_pixel - dx - 1)
                    ys_plot.append(y_pixel - 1)
                    arrows.append('<')

                    xs_plot.append(semantic_map_size_x - x_pixel - 2)
                    ys_plot.append(y_pixel - dy - 2)
                    arrows.append('^')
                    xs_plot.append(semantic_map_size_x - x_pixel - 2)
                    ys_plot.append(y_pixel + dy - 1)
                    arrows.append('v')

                if self.ontology[i][0] == self.moved_object_value and self.moved_object_countdown > 0:
                    for j in range(0, len(arrows)):
                        C = np.array([255, 0, 0])
                        if 'chair' in self.ontology[i][2]:
                            plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)
                        else:
                            plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=2, alpha=0.4)
                else:
                    for j in range(0, len(arrows)):
                        R = explanation[y_pixel][x_pixel][0]
                        G = explanation[y_pixel][x_pixel][1]
                        B = explanation[y_pixel][x_pixel][2]
                        C = np.array([R, G, B])
                        if 'chair' in self.ontology[i][2]:
                            plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=3, alpha=0.4)
                        else:
                            plt.plot(xs_plot[j], ys_plot[j], marker=arrows[j], c=C/255.0, markersize=1, alpha=0.6)

        # PLOT HUMANS AS BLINKING EXCLAMATION MARKS        
        if self.human_blinking == True:
            for i in range(self.ont_len - 6, self.ont_len):
                x_map = copy.deepcopy(self.ontology[i][3])
                y_map = copy.deepcopy(self.ontology[i][4])

                #distance_human_robot = math.sqrt((x_map - self.robot_pose_map.position.x)**2 + (y_map - self.robot_pose_map.position.y)**2)
            
                #if distance_human_robot > self.explanation_window:
                #    continue   

                # for nicer plotting
                x_map += 0.2
                y_map += 0.2

                x_pixel = int((x_map + - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
                y_pixel = int((y_map - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)

                if self.failure_started == True and i == 34:
                    #C = np.array([255,102,102])
                    C = np.array([255,0,0])
                    ax.text(semantic_map_size_x - x_pixel, y_pixel, 'i', c=C/255.0)
                else:
                    ax.text(semantic_map_size_x - x_pixel, y_pixel, 'i', c='yellow')

            self.human_blinking = False
        else:
            self.human_blinking = True

        # CONVERT IMAGE TO NUMPY ARRAY 
        fig.savefig('explanation' + '.png', transparent=False)
        plt.close()
        output = PIL.Image.open(os.getcwd() + '/explanation.png').convert('RGB')        
        output = np.array(output)[:,:,:3].astype(np.uint8)
        self.visual_explanation = output #[self.vis_exp_coords[0]:self.vis_exp_coords[1], (semantic_map_size_x-self.vis_exp_coords[3]):(semantic_map_size_x-self.vis_exp_coords[2]), :]
        self.visual_explanation_origin_x = self.global_semantic_map_origin_x # + (self.vis_exp_coords[2]) * self.visual_explanation_resolution #robot_pose.position.x - self.explanation_representation_threshold #0.5 * self.visual_explanation_resolution * (vis_exp_coords[3] - vis_exp_coords[2]) 
        self.visual_explanation_origin_y = self.global_semantic_map_origin_y # + self.vis_exp_coords[0] * self.visual_explanation_resolution #robot_pose.position.y - self.explanation_representation_threshold #0.5 * self.visual_explanation_resolution * (vis_exp_coords[1] - vis_exp_coords[0]) 
        
        #from PIL import Image
        #im = Image.fromarray(self.visual_explanation)
        #im.save("visual_explanation.png")

        # VISUALIZE CURRENT PATH
        if self.failure_started == False:
            self.visualize_current_plan()
        else:
            self.visualize_empty_current_plan()
            #self.current_path_marker_array.markers = []

    # publish visual explanation
    def publish_visual_explanation(self):
        #print('publishing visual')
        #points_start = time.time()

        #'''
        if self.moved_object_countdown > 0:
            #print('self.moved_object_countdown = ', self.moved_object_countdown)
            self.moved_object_countdown -= 1
        elif self.moved_object_countdown == 0:
            #print('self.moved_object_countdown = ', self.moved_object_countdown)
            self.moved_object_countdown = -1
            self.moved_object_value = -1
            #self.prepare_global_semantic_map_for_publishing()
            #self.publish_global_semantic_map()
        #'''   
    
        z = 0.0
        a = 255                    
        points = []

        # draw layer
        size_1 = int(self.visual_explanation.shape[1])
        size_0 = int(self.visual_explanation.shape[0])
        for i in range(0, size_1):
            for j in range(0, size_0):
                #if (self.visual_explanation[j, i, 0] == 120 and self.visual_explanation[j, i, 1] == 120 and self.visual_explanation[j, i, 2] == 120) or (self.visual_explanation[j, i, 0] == 180 and self.visual_explanation[j, i, 1] == 180 and self.visual_explanation[j, i, 2] == 180):
                #if self.visual_explanation[j, i, 0] == self.visual_explanation[j, i, 1] and self.visual_explanation[j, i, 1] == self.visual_explanation[j, i, 2] and self.visual_explanation[j, i, 2] == self.visual_explanation[j, i, 0]:
                if self.visual_explanation[j, i, 0] == self.visual_explanation[j, i, 1] == self.visual_explanation[j, i, 2] == 120 or self.visual_explanation[j, i, 0] == self.visual_explanation[j, i, 1] == self.visual_explanation[j, i, 2] == 180:
                    continue
                x = self.visual_explanation_origin_x + (size_1-i) * self.visual_explanation_resolution
                y = self.visual_explanation_origin_y + j * self.visual_explanation_resolution
                z = 0.01
                r = int(self.visual_explanation[j, i, 0])
                g = int(self.visual_explanation[j, i, 1])
                b = int(self.visual_explanation[j, i, 2])
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                pt = [x, y, z, rgb]
                points.append(pt)

        #points_end = time.time()
        #print('explanation layer runtime = ', round(points_end - points_start,3))
        
        # publish
        self.header.frame_id = 'map'
        pc2 = point_cloud2.create_cloud(self.header, self.fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.pub_explanation_layer.publish(pc2)
        
        self.publish_semantic_labels_local()

        #print(len(self.current_path_marker_array.markers))
        self.pub_current_path.publish(self.current_path_marker_array)
        
        if self.moved_object_countdown == 39:
            self.pub_old_path.publish(self.old_path_marker_array)
            
    # publish local semantic labels
    def publish_semantic_labels_local(self):
        # check objects
        for i in range(0, len(self.ontology) - self.N_humans):
            if self.ontology[i][0] in self.neighborhood_objects_IDs:
                self.semantic_labels_marker_array.markers[i].action = self.semantic_labels_marker_array.markers[i].ADD
                x_map = self.ontology[i][12]
                y_map = self.ontology[i][13]            
                self.semantic_labels_marker_array.markers[i].pose.position.x = x_map
                self.semantic_labels_marker_array.markers[i].pose.position.y = y_map
            else:
                self.semantic_labels_marker_array.markers[i].action = self.semantic_labels_marker_array.markers[i].DELETE

        # check humans
        for i in range(len(self.ontology) - self.N_humans, len(self.ontology)):
            x_map = self.ontology[i][3]
            y_map = self.ontology[i][4]

            distance_human_robot = math.sqrt((x_map - self.robot_pose_map.position.x)**2 + (y_map - self.robot_pose_map.position.y)**2)
            
            if distance_human_robot < self.explanation_window:
                self.semantic_labels_marker_array.markers[i].action = self.semantic_labels_marker_array.markers[i].ADD
                x_map = self.ontology[i][12]
                y_map = self.ontology[i][13]            
                self.semantic_labels_marker_array.markers[i].pose.position.x = x_map
                self.semantic_labels_marker_array.markers[i].pose.position.y = y_map

                if self.failure_started:
                    self.semantic_labels_marker_array.markers[i].color.r = 1.0
                    self.semantic_labels_marker_array.markers[i].color.g = 0.0
                    self.semantic_labels_marker_array.markers[i].color.b = 0.0
                else:
                    self.semantic_labels_marker_array.markers[i].color.r = 1.0
                    self.semantic_labels_marker_array.markers[i].color.g = 1.0
                    self.semantic_labels_marker_array.markers[i].color.b = 1.0
            else:
                self.semantic_labels_marker_array.markers[i].action = self.semantic_labels_marker_array.markers[i].DELETE
     
        self.pub_semantic_labels.publish(self.semantic_labels_marker_array)

    # visualize current path
    def visualize_current_plan(self):
        self.visualize_empty_current_plan()

        self.global_plan_current_hold = copy.deepcopy(self.global_plan_current)
     
        current_path_length = len(self.global_plan_current_hold.poses)
        current_marker_array_length = len(self.current_path_marker_array.markers)
        
        #self.current_path_marker_array.markers = []        
        
        if current_path_length > 15:
            for i in range(15, current_path_length-8):
                # visualize path
                marker = Marker()
                marker.header.frame_id = 'map'
                k = i - 15
                marker.id = k
                marker.type = marker.SPHERE
                marker.action = marker.ADD #DELETEALL #ADD
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
                if k >= current_marker_array_length:
                    self.current_path_marker_array.markers.append(marker)
                else:
                    self.current_path_marker_array.markers[k] = marker

        else:
            for i in range(0, current_path_length-8):
                # visualize path
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = i
                marker.type = marker.SPHERE
                marker.action = marker.ADD #DELETEALL #ADD
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
                if i >= current_marker_array_length:
                    self.current_path_marker_array.markers.append(marker)
                else:
                    self.current_path_marker_array.markers[i] = marker

    # visualize empty current path
    def visualize_empty_current_plan(self):
        for i in range(0, len(self.current_path_marker_array.markers)):
            #self.current_path_marker_array.markers[i].pose.position.x = 2.3
            #self.current_path_marker_array.markers[i].pose.position.x = 2.3
            #self.current_path_marker_array.markers[i].pose.position.x = 2.3
            self.current_path_marker_array.markers[i].action = self.current_path_marker_array.markers[i].DELETE

    # visualize old path 
    def visualize_old_plan(self):
        #print('\nvisualize_old_plan')
        self.old_path_marker_array.markers = []

        #print('len(self.old_plan.poses) = ', len(self.old_plan.poses))
        for i in range(25, len(self.old_plan.poses), 1):
            #x_map = self.old_plan.poses[i].pose.position.x
            #y_map = self.old_plan.poses[i].pose.position.y

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

def main():
    # ----------main-----------
    rospy.init_node('hixron', anonymous=False)

    # define hixron object
    hixron_obj = hixron()
    
    # call explanation once to establish static map
    hixron_obj.setup()
    
    # sleep for x sec until Amar starts the video
    d = rospy.Duration(5, 0)
    rospy.sleep(d)
    
    # send the goal pose to start navigation
    hixron_obj.send_goal_pose()
    hixron_obj.visualize_goal_pose()

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    #rate = rospy.Rate(0.15)
    while not rospy.is_shutdown():
        #print('spinning')
        #rate.sleep()
        hixron_obj.move_objects()
        hixron_obj.test_explain()
        #rospy.spin()
        
main()