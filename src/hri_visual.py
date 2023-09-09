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
        # extroversion vars
        self.extroversion_prob = 0.0        
        self.explanation_window = 3.6 - 2.4 * self.extroversion_prob

        self.humans_nearby = False

        self.moved_object_distance_threshold = 1.5
        
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

        color_shape_path_combination = [2,1,0]
        
        self.color_schemes = ['only_red', 'red_nuanced', 'green_yellow_red']
        self.color_scheme = self.color_schemes[color_shape_path_combination[0]]
        #color_whole_objects = False

        #shape_schemes = ['wo_text', 'with_text']
        #shape_scheme = shape_schemes[color_shape_path_combination[1]]

        self.path_schemes = ['full_line', 'arrows']
        self.path_scheme = self.path_schemes[color_shape_path_combination[2]]


        # textual explanation vars
        self.pub_text_exp = rospy.Publisher('/textual_explanation', String, queue_size=10)
        self.text_exp = ''

        # point cloud vars
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
        self.goal_pose_current = Pose()
        #self.goal_pose_history = []

        # ontology part
        self.scenario_name = 'icra_melodic'
        # load ontology
        self.ontology = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'ontology.csv')
        self.ontology = np.array(self.ontology)
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
        #self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)
        #self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)
        #self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_callback)

        # publishers
        self.pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=1)
        self.pub_current_path = rospy.Publisher('/path_markers', MarkerArray, queue_size=1)
        self.pub_old_path = rospy.Publisher('/old_path_markers', MarkerArray, queue_size=1)
        self.pub_explanation_layer = rospy.Publisher("/explanation_layer", PointCloud2, queue_size=1)
        self.pub_move = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.pub_semantic_map = rospy.Publisher("/semantic_map", PointCloud2, queue_size=1)

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
            marker.text = self.ontology[i][1]
            marker.ns = "my_namespace"
            self.semantic_labels_marker_array.markers.append(marker)

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

        goal.target_pose.pose.position.x = 0.603
        goal.target_pose.pose.position.y = -6.632
        goal.target_pose.pose.position.z = 0.0

        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = -0.71
        goal.target_pose.pose.orientation.w = 0.70
   
        # Sends the goal to the action server.
        client.send_goal(goal)

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
        #self.move_objects()

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
                
                self.publish_semantic_labels()
                self.publish_semantic_labels()
                self.publish_semantic_labels()

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
                
                self.publish_semantic_labels()
                self.publish_semantic_labels()
                self.publish_semantic_labels()

                self.cancel_goal_pose()
                self.failure_start = time.time()

        if self.human_4_original_moved == False and self.human_4_moved_moved == True:
            self.failure_end = time.time()
            self.failure_time = self.failure_end - self.failure_start
            print(self.failure_time)

            if self.failure_time > 2.5:
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
                
                self.publish_semantic_labels()
                self.publish_semantic_labels()
                self.publish_semantic_labels()

            if self.failure_time > 2.99:
                self.send_goal_pose()

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
                print('DOING!!!!!')

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
        print('prepare_global_semantic_map_for_publishing')
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

    # publish semantic labels
    def publish_semantic_labels(self):
        # update poses of possible objects
        self.semantic_labels_marker_array.markers[1].pose.position.x = self.ontology[1][12]
        self.semantic_labels_marker_array.markers[1].pose.position.y = self.ontology[1][13]

        self.semantic_labels_marker_array.markers[34].pose.position.x = self.ontology[34][12]
        self.semantic_labels_marker_array.markers[34].pose.position.y = self.ontology[34][13]
        
        self.pub_semantic_labels.publish(self.semantic_labels_marker_array)

    # test whether explanation is needed
    def test_explain(self):
        #print('test_explain!')

        if self.first_call:
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
            
            self.publish_semantic_labels()

            return
            
        #self.publish_semantic_labels()

        #self.explain_visual()
        

def main():
    # ----------main-----------
    rospy.init_node('hixron', anonymous=False)

    # define hixron object
    hixron_obj = hixron()
    
    # call explanation once to establish static map
    hixron_obj.first_call = True
    hixron_obj.test_explain()
    hixron_obj.first_call = False
    
    # sleep for x sec until Amar starts the video
    d = rospy.Duration(1, 0)
    rospy.sleep(d)
    
    # send the goal pose to start navigation
    hixron_obj.send_goal_pose()

    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    #rate = rospy.Rate(0.15)
    while not rospy.is_shutdown():
        #print('spinning')
        #rate.sleep()
        #hixron_obj.test_explain()
        hixron_obj.move_objects()
        #rospy.spin()
        
main()