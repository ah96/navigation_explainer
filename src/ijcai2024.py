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

# get QSR value
def getIntrinsicQsrValue(angle):
    value = ''
    qsr_choice = 3    

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

    elif qsr_choice == 2:
        if -PI/4 <= angle < PI/4:
            value += 'right'
        elif 3*PI/4 <= angle < PI or -PI <= angle < -3*PI/4:
            value += 'left'

    elif qsr_choice == 3:
        if -PI/4 <= angle < PI/4:
            value += 'right'
        if PI/4 <= angle < 3*PI/4:
            value += 'front'
        elif 3*PI/4 <= angle < PI or -PI <= angle < -3*PI/4:
            value += 'left'
        elif -3*PI/4 <= angle < -PI/4:
            value += 'back'

    return value

# get spatial zone
def getSpatialZone(x, y):
    robot_offset = 9.0
    quiet_zone_coords = [-10+robot_offset, -7.3+robot_offset, -3.78-robot_offset, 10.82-robot_offset]
    no_hold_zone_coords = [-7.29+robot_offset, -4.01+robot_offset, -3.78-robot_offset, 10.82-robot_offset]
    social_zone_coords = [-4.0+robot_offset, -0.5+robot_offset, -3.78-robot_offset, 10.82-robot_offset]

    value = ''
    qsr_choice = 3    

    if social_zone_coords[0] <= x <= social_zone_coords[1] and social_zone_coords[2] <= y <= social_zone_coords[3]:
        value = 'social_zone'
    elif no_hold_zone_coords[0] <= x <= no_hold_zone_coords[1] and no_hold_zone_coords[2] <= y <= no_hold_zone_coords[3]:
        value = 'no_hold_zone'
    elif quiet_zone_coords[0] <= x <= quiet_zone_coords[1] and quiet_zone_coords[2] <= y <= quiet_zone_coords[3]:
        value = 'quiet_zone'    
    
    return value

# hixron class
class hixron(object):
    # constructor
    def __init__(self):
        # extroversion vars
        self.extroversion_prob = 1.0
        self.fully_extrovert = False
        if self.extroversion_prob == 1.0:
            self.fully_extrovert = True
        
        self.explanation_representation_threshold = 3.6 - 2.4 * self.extroversion_prob
        self.explanation_window = 3.6 - 2.4 * self.extroversion_prob

        self.timing = int(20 * (1 - self.extroversion_prob))
        self.duration = int(20 * (1 - self.extroversion_prob))
        self.explanation_cycle_len = self.timing + self.duration
        self.introvert_publish_ctr = copy.deepcopy(self.explanation_cycle_len)

        self.humans_nearby = False
        
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

        # statistics
        self.N_deviations_explained = 0
        self.N_words = 0
        self.N_objects = 0
        self.visual_N = 0
        self.textual_N = 0

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
        self.robot_offset = 9.0
        
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
        self.scenario_name = 'ijcai2024'
        # load ontology
        self.ontology = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/scenarios/' + self.scenario_name + '/' + 'ontology.csv')
        self.ontology = np.array(self.ontology)
        # apply map-world offset
        for i in range(self.ontology.shape[0]):
            self.ontology[i, 3] += self.robot_offset
            self.ontology[i, 4] -= self.robot_offset

            self.ontology[i, 12] += self.robot_offset
            self.ontology[i, 13] -= self.robot_offset

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
        self.global_semantic_map_origin_x += self.robot_offset  
        self.global_semantic_map_origin_y = float(self.global_semantic_map_info[0,5])
        self.global_semantic_map_origin_y -= self.robot_offset 
        self.global_semantic_map_resolution = float(self.global_semantic_map_info[0,1])
        self.global_semantic_map_size = [int(self.global_semantic_map_info[0,3]), int(self.global_semantic_map_info[0,2])]
        self.global_semantic_map = np.zeros((self.global_semantic_map_size[0],self.global_semantic_map_size[1]))
        #print(self.global_semantic_map_origin_x, self.global_semantic_map_origin_y, self.global_semantic_map_resolution, self.global_semantic_map_size)
        self.global_semantic_map_complete = []

        # initialize an array of semantic labels (static map)
        self.init_semantic_labels()

        # subscribers 
        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)
        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)
        self.sub_goal_pose = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback)
        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_callback)

        # publishers
        self.pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=1)
        self.pub_current_path = rospy.Publisher('/path_markers', MarkerArray, queue_size=1)
        self.pub_old_path = rospy.Publisher('/old_path_markers', MarkerArray, queue_size=1)
        self.pub_explanation_layer = rospy.Publisher("/explanation_layer", PointCloud2, queue_size=1)
        self.pub_move = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.pub_semantic_map = rospy.Publisher("/semantic_map", PointCloud2, queue_size=1)

        self.pub_zone = rospy.Publisher("/zone", String, queue_size=1)

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
        #print('\namcl_callback')

        self.robot_pose_map = msg.pose.pose
        
    # goal pose callback
    def goal_pose_callback(self, msg):
        print('goal_pose_callback')

        self.goal_pose_current = msg.pose
        self.goal_pose_current.position.x -= self.robot_offset
        self.goal_pose_current.position.y += self.robot_offset
        #self.goal_pose_history.append(msg.pose)

    # Gazebo callback
    def gazebo_callback(self, states_msg):
        #print('gazebo_callback')  
        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

        # position update of dynamic objects
        for i in range(0, len(self.gazebo_names)):
            if 'citizen' in self.gazebo_names[i] or 'human' in self.gazebo_names[i]:
                self.gazebo_poses[i].position.x += self.robot_offset
                self.gazebo_poses[i].position.y -= self.robot_offset 
                self.humans.append(self.gazebo_poses[i])

    # global plan callback
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')

        self.global_plan_ctr += 1
        
        # save global plan to class vars
        self.global_plan_current = copy.deepcopy(msg)
        self.global_plan_current_goal = copy.deepcopy(self.goal_pose_current) 
        #self.global_plan_history.append(self.global_plan_current)
        #self.globalPlan_goalPose_indices_history.append([len(self.global_plan_history), len(self.goal_pose_history)])

        # save global plan to class vars
        #self.global_plan_previous = copy.deepcopy(msg)
        #self.global_plan_previous_goal = copy.deepcopy(self.goal_pose_current) 
         
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

    # publish empty current plan
    def publish_empty_current_plan(self):
        for i in range(0, len(self.current_path_marker_array.markers)):
            # visualize path
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.SPHERE
            marker.action = marker.DELETE #DELETEALL #ADD
            #marker.lifetime = 1.0
            marker.pose = Pose()
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0.95
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 0
            marker.color.r = 0.043
            marker.color.g = 0.941
            marker.color.b = 1.0
            marker.color.a = 0.5        
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            #marker.frame_locked = False
            marker.ns = "my_namespace"
            self.current_path_marker_array.markers[i] = marker

            self.pub_current_path.publish(self.current_path_marker_array)
    
    # publish semantic labels
    def publish_semantic_labels(self):
        ID = self.ontology.shape[0]
        for human_pose in self.humans:
            x_map = human_pose.position.x
            y_map = human_pose.position.y

            human_nearby = False
            distance_human_robot = math.sqrt((x_map - self.robot_pose_map.position.x)**2 + (y_map - self.robot_pose_map.position.y)**2)
            if distance_human_robot < self.explanation_representation_threshold:
                human_nearby = True
            
            # visualize orientations and semantic labels of humans
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = ID
            ID += 1
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = x_map# + 0.25
            marker.pose.position.y = y_map# - 0.6
            marker.pose.position.z = 0.5
            marker.pose.orientation.x = 0.0#qx
            marker.pose.orientation.y = 0.0#qy
            marker.pose.orientation.z = 0.0#qz
            marker.pose.orientation.w = 0.0#qw
            if human_nearby:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.scale.x = 0.35
                marker.scale.y = 0.35
                marker.scale.z = 0.35
            else:
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

    # visualize current path
    def visualize_current_plan(self):
        self.global_plan_current_hold = copy.deepcopy(self.global_plan_current)
        if self.path_scheme == self.path_schemes[0]:         
            previous_marker_array_length = len(self.current_path_marker_array.markers)
            current_marker_array_length = len(self.global_plan_current_hold.poses)-25
            delete_needed = False
            if current_marker_array_length < previous_marker_array_length:
                delete_needed = True
            len_max = max(previous_marker_array_length, current_marker_array_length)
            #print('len_max = ', len_max)
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
        
        # VISUALIZE CURRENT PATH
        self.visualize_current_plan()

        # determine zone
        zone = getSpatialZone(self.robot_pose_map.position.x, self.robot_pose_map.position.y)
        print('\nzone: ', zone)
        self.pub_zone.publish(zone)


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
    d = rospy.Duration(2, 0)
    rospy.sleep(d)
    
    # send the goal pose to start navigation
    hixron_obj.send_goal_pose()

    with open('eval.csv', "a") as myfile:
        myfile.write('extroversion,visual_time,visual_publish_time,visual_N,textual_time,textual_publish_time,textual_N,N_objects,N_words,N_deviations' + '\n')
    myfile.close()
    
    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    #rate = rospy.Rate(0.15)
    while not rospy.is_shutdown():
        #print('spinning')
        #rate.sleep()
        hixron_obj.test_explain()
        #rospy.spin()
        
main()