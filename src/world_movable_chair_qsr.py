#!/usr/bin/env python3

from gazebo_msgs.msg import ModelStates, ModelState
import rospy
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import copy
from std_msgs.msg import Float32MultiArray
import pandas as pd
#import tf2_ros
import os
from scipy.spatial.transform import Rotation as R

PI = math.pi

# Initialize the ROS Node named 'qsr', allow multiple nodes to be run with this name
rospy.init_node('qsr_live_2', anonymous=True)

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

# objects class
class Object():
    # constructor
    def __init__(self,ID,label,position_map,position_odom,centroid_map,centroid_odom,distance,known):
        self.ID = ID
        self.label = label
        self.position_map = Point(position_map[0],position_map[1],0.0)
        self.position_odom = Point(position_odom[0],position_odom[1],0.0)
        self.centroid_map = Point(centroid_map[0],centroid_map[1],0.0)
        self.centroid_odom = Point(centroid_odom[0],centroid_odom[1],0.0)
        self.distance = Point(distance[0],distance[1],0.0)
        self.known = known
        
        
        if 'cabinet' in self.name:
            self.orientation_map = euler_to_quaternion(0.0,0.0,PI/2)
        else:
            self.orientation_map = Quaternion(0.0,0.0,0.0,1.0)
        
        [self.yaw_map,self.pitch_map,self.roll_map] = quaternion_to_euler(self.orientation_map.x,self.orientation_map.y,self.orientation_map.z,self.orientation_map.w)
        
        self.position_obj = Point(0.0,0.0,0.0)
        self.orientation_obj = Quaternion(0.0,0.0,0.0,1.0)

        self.orientation_odom = Quaternion(0.0,0.0,0.0,1.0)
        
        self.transform_object_to_map = [self.position_map.x, self.position_map.y, self.position_map.z, 0.0, 0.0, 0.0, 1.0]
        self.transform_map_to_object = [-self.position_map.x, -self.position_map.y, -self.position_map.z, 0.0, 0.0, 0.0, 1.0]

        self.lime_coefficients = []

        self.PI = PI
        
        # define intrinsic_qsr for cabinets and bookshelfs
        self.intrinsic_qsr = False
        if 'cabinet' in self.name or 'bookshelf' in self.name:
            self.intrinsic_qsr = True
            self.qsr_choice = 1
            self.defineIntrinsicQsrCalculus()

    # define intrinsic QSR calculus
    def defineIntrinsicQsrCalculus(self):
        if self.qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        # used for deriving NLP annotations
        self.qsr_dict_inv = {v: k for k, v in self.qsr_dict.items()}

    # get QSR value
    def getIntrinsicQsrValue(self, angle):
        value = ''    

        if self.qsr_choice == 0:
            if -self.PI/2 <= angle < self.PI/2:
                value += 'right'
            elif self.PI/2 <= angle < self.PI or -self.PI <= angle < -self.PI/2:
                value += 'left'

        elif self.qsr_choice == 1:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        return value


# known objects
# [ID, label, x_odom, y_odom, x_map, y_map, cx_odom, cy_odom, cx_map, cy_map, dx, dy]
known_objects = []
known_objects_names =  []
known_objects_positions =  []
N_known_objects = len(known_objects_names)

# unknown objects
# [ID, label, x_odom, y_odom, x_map, y_map, cx_odom, cy_odom]
unknown_objects = []
unknown_objects_names =  []
unknown_objects_positions =  []
N_unknown_objects = len(unknown_objects_names)

# objects in lc
lc_objects = []
lc_objects_names =  []
lc_objects_positions =  []
N_lc_objects = len(lc_objects_names)



















# intialize static objects
static_objects = []
marker_array_semantic_labels = MarkerArray()
marker_array_orientations = MarkerArray()
marker_array_robot_human_orientations = MarkerArray()
marker_array_robot_human_semantic_labels = MarkerArray()

for i in range(0, len(static_objects_names)):
    static_objects.append(Object(static_objects_names[i], static_objects_positions[i]))

    # visualize oriented static objects
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = i
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = static_objects[i].position_map.x
    marker.pose.position.y = static_objects[i].position_map.y
    marker.pose.position.z = 0.5
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    #marker.frame_locked = False
    marker.text = static_objects[i].name
    marker.ns = "my_namespace"
    marker_array_semantic_labels.markers.append(marker) 

    if 'cabinet' in static_objects[i].name or 'bookshelf' in static_objects[i].name:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = i
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = static_objects[i].position_map.x
        marker.pose.position.y = static_objects[i].position_map.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = static_objects[i].orientation_map.x
        marker.pose.orientation.y = static_objects[i].orientation_map.y
        marker.pose.orientation.z = static_objects[i].orientation_map.z
        marker.pose.orientation.w = static_objects[i].orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        marker_array_orientations.markers.append(marker)
#print('\nlen(static_objects) = ', len(static_objects))


pub_markers_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
pub_markers_orientations = rospy.Publisher('/orientations', MarkerArray, queue_size=10)

# Robot class
class Robot():
    # constructor
    def __init__(self,name):
        self.name = name
        
        self.position_map = Point(0.0,0.0,0.0)        
        self.orientation_map = Quaternion(0.0,0.0,0.0,1.0)

        self.position_odom = Point(0.0,0.0,0.0)        
        self.orientation_odom = Quaternion(0.0,0.0,0.0,1.0)

        self.velocity_vector = [0.0,0.0]

        self.intrinsic_qsr_choice = 1
        self.relative_qsr_choice = 2

        self.PI = PI
        
    # print attributes    
    def printAttributes(self):
        print('\nname = ', self.name)
        print('position_map = ', self.position_map)
        print('orientation_map = ', self.orientation_map)
        print('position_odom = ', self.position_map)
        print('orientation_odom = ', self.orientation_map)
        print('velocity_vector = ', self.velocity_vector)

    # define intrinsic QSR calculus
    def defineIntrinsicQsrCalculus(self):
        if self.intrinsic_qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.intrinsic_qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.intrinsic_qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.intrinsic_qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        # used for deriving NLP annotations
        self.qsr_dict_inv = {v: k for k, v in self.qsr_dict.items()}

    def getIntrinsicQsrValue(self, angle):
        value = ''    

        if self.intrinsic_qsr_choice == 0:
            if -self.PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif self.intrinsic_qsr_choice == 1:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        return value

        # define QSR calculus
    def defineRelativeQsrCalculus(self, qsr_choice):
        if self.relative_qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.relative_qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.relative_qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.relative_qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        elif self.relative_qsr_choice == 2:
            # TPCC reference system
            # my modified version from 'moratz2008qualitative'
            # used for getting semantic costmap
            if self.R == 0:
                self.relative_qsr_dict = {
                    'sb': 0,
                    'lb': 1,
                    'bl': 2,
                    'sl': 3,
                    'fl': 4,
                    'lf': 5,
                    'sf': 6,
                    'rf': 7,
                    'fr': 8,
                    'sr': 9,
                    'br': 10,
                    'rb': 11
                }
            else:
                self.relative_qsr_dict = {
                    'csb': 0,
                    'dsb': 1,
                    'clb': 2,
                    'dlb': 3,
                    'cbl': 4,
                    'dbl': 5,
                    'csl': 6,
                    'dsl': 7,
                    'cfl': 8,
                    'dfl': 9,
                    'clf': 10,
                    'dlf': 11,
                    'csf': 12,
                    'dsf': 13,
                    'crf': 14,
                    'drf': 15,
                    'cfr': 16,
                    'dfr': 17,
                    'csr': 18,
                    'dsr': 19,
                    'cbr': 20,
                    'dbr': 21,
                    'crb': 22,                    
                    'drb': 23
                }

                self.comb_table = [
                    [[0,1],[0,1],[2,3],[2,3],[2,3],[2,3,4,5],[2,3],[2,3,4,5],[2,3,4,5],[4,5,6,7,8,9],[2,4],[6,8,9,10,11],[0],[12,13]],
                    [[1],[1],[3],[3],[3],[3,5],[3],[3,5],[2,3,4,5],[4,5,7,9],[2,3,4,5],[4,5,6,7,8,9,10,11],[0,1],[12,13]],
                    [[2,3],[2,3],[2,3,4,5],[2,3,4,5],[2,3,4,5],[2,3,4,5,6,7,8,9],[2,3,4,5],[4,5,6,7,8,9],[2,3,4,5,6,7,8,9],[4,5,6,7,8,9,10,11],[2,4,6,8],[4,6,8,9,10,11,12,13,14,15],[2],[14,15]],
                    [[3],[3],[3,5],[3,5],[3,5],[3,5,7,9],[3,5],[5,7,9],[2,3,4,5,6,7,8,9],[4,5,6,7,8,9,11],[2,3,4,5,6,7,8,9],[4,5,6,7,8,9,10,11,12,13,14,15],[2,3],[14,15]],
                    [[4,5],[4,5],[4,5,6,7,8,9],[4,5,6,7,8,9],[4,5,6,7,8,9],[4,5,6,7,8,9,10,11],[4,5,6,7,8,9],[8,9,10,11],[4,5,6,7,8,9,10,11],[8,9,10,11,12,13,14,15],[4,6,8,10],[8,10,11,12,13,14,15,17],[4],[16,17]],
                    [[5],[5],[5,7,9],[5,7,9],[5,7,9],[5,7,9,11],[5,7,9],[9,11],[4,5,6,7,8,9,10,11],[8,9,10,11,13,15],[4,5,6,7,8,9,10,11],[9,10,11,12,13,14,15,16,17],[4,5],[16,17]],
                    [[6,7],[6,7],[8,9],[8,9],[8,9],[8,9,10,11],[8,9],[10,11],[8,9,10,11],[10,11,12,13,14,15],[8,10],[10,12,14,15,16,17],[6],[18,19]],
                    [[7],[7],[9],[9],[9],[9,11],[9],[11],[8,9,10,11],[10,11,13,15],[8,9,10,11],[10,11,12,13,14,15,18],[6,7],[18,19]],
                    [[8,9],[8,9],[8,9,10,11],[8,9,10,11],[8,9,10,11],[8,9,10,11,12,13,14,15],[8,9,10,11],[10,11,12,13,14,15],[8,9,10,11,12,13,14,15],[10,11,12,13,14,15,16,17],[8,10,12,14],[10,12,14,15,16,17,18,19,20,21],[8],[20,21]],
                    [[9],[9],[9,11],[9,11],[9,11],[9,11,13,15],[9,11],[11,13,15],[8,9,10,11,12,13,14,15],[10,11,12,13,14,15,17],[8,9,10,11,12,13,14,15],[10,11,12,13,14,15,16,17,18,19,20,21],[8,9],[20,21]],
                    [[10,11],[10,11],[10,11,12,13,14,15],[10,11,12,13,14,15],[10,11,12,13,14,15],[10,11,12,13,14,15,16,17],[10,11,12,13,14,15],[12,13,14,15,16,17],[10,11,12,13,14,15,16,17],[15,16,17,18,19,20,21],[10,12,14,16],[14,16,17,18,19,20,21,22,23],[10],[22,23]],
                    [[11],[11],[11,13,15],[11,13,15],[11,13,15],[11,13,15,17],[11,13,15],[13,15,17],[10,11,12,13,14,15,16,17],[14,15,16,17,19,21],[10,11,12,13,14,15,16,17],[14,15,16,17,18,19,20,21,22,23],[10,11],[22,23]],
                    [[12,13],[12,13],[14,15],[14,15],[14,15],[14,15,16,17],[14,15],[14,15,16,17],[14,15,16,17],[16,17,18,19,20,21],[14,16],[16,18,20,21,22,23],[12],[0,1]],
                    [[13],[13],[15],[15],[15],[15,17],[15],[15,17],[14,15,16,17],[16,17,19,21],[14,15,16,17],[16,17,18,19,20,21,22,23],[12,13],[0,1]],
                    [[14,15],[14,15],[14,15,16,17],[14,15,16,17],[14,15,16,17],[15,16,17,18,19,20,21],[14,15,16,17],[16,17,18,19,20,21],[14,15,16,17,18,19,20,21],[16,17,18,19,20,21,22,23],[14,16,18,20],[0,1,2,3,16,18,20,21,22,23],[14],[2,3]],
                    [[15],[15],[15,17],[15,17],[15,17],[15,17,19,21],[15,17],[17,19,21],[14,15,16,17,18,19,20,21],[16,17,18,19,20,21,23],[14,15,16,17,18,19,20,21],[0,1,2,3,16,17,18,19,20,21,22,23],[14,15],[2,3]],
                    [[16,17],[16,17],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21],[16,17,18,19,20,21,22,23],[16,17,18,19,20,21],[20,21,22,23],[16,17,18,19,20,21,22,23],[0,1,2,3,20,21,22,23],[16,18,20,22],[0,1,2,3,4,5,20,22,23],[16],[4,5]],
                    [[17],[17],[17,19,21],[17,19,21],[17,19,21],[17,19,21,23],[17,19,21],[21,23],[16,17,18,19,20,21,22,23],[1,3,20,21,22,23],[16,17,18,19,20,21,22,23],[0,1,2,3,4,5,20,21,22,23],[16,17],[4,5]],
                    [[18,19],[18,19],[20,21],[20,21],[20,21],[20,21,22,23],[20,21],[22,23],[20,21,22,23],[0,1,2,3,22,23],[20,22],[0,2,3,4,5,22],[18],[6,7]],
                    [[19],[19],[21],[21],[21],[21,23],[21],[23],[20,21,22,23],[1,3,22,23],[20,21,22,23],[0,1,2,3,4,5,22,23],[18,19],[6,7]],
                    [[20,21],[20,21],[20,21,22,23],[20,21,22,23],[20,21,22,23],[0,1,2,3,20,21,22,23],[20,21,22,23],[0,1,2,3,22,23],[0,1,2,3,20,21,22,23],[0,1,2,3,4,5,22,23],[0,2,20,22],[0,2,3,4,5,6,7,8,9,22],[20],[8,9]],
                    [[21],[21],[21,23],[21,23],[21,23],[1,3,21,23],[21,23],[1,3,23],[0,1,2,3,20,21,22,23],[0,1,2,3,5,22],[0,1,2,3,20,21,22,23],[0,1,2,3,4,5,6,7,8,9,23],[20,21],[8,9]],
                    [[22,23],[22,23],[0,1,2,3,22,23],[0,1,2,3,22,23],[0,1,2,3,22,23],[0,1,2,3,4,5,22,23],[0,1,2,3,22,23],[0,1,2,3,4,5],[0,1,2,3,4,5,22,23],[2,3,4,5,6,7,8,9],[0,2,4,22],[2,4,5,6,7,8,9,10,11],[22],[10,11]],
                    [[23],[23],[1,3,23],[1,3,23],[1,3,23],[1,3,5,23],[1,3,23],[1,3,5],[0,1,2,3,4,5,22,23],[2,3,4,5,7,9],[0,1,2,3,4,5,22,23],[3,4,5,6,7,8,9,10,11],[22,23],[10,11]]
                    ]

                self.perm_dict = {
                    'ID':0,
                    'INV':1,
                    'SC':2,
                    'SCI':3,
                    'HM':4,
                    'HMI':5
                }

                self.per_table = [
                    [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]],
                    [[13],[13],[15],[15],[15],[15,17],[15],[15,17],[14,15,16,17],[16,17,19,21],[14,16],[16,18,20,21,22,23],[12],[0,1]],
                    [[12],[12],[14,16],[14,16],[14,16,18],[14],[16,18],[14],[16,17,18,20],[14,15,16,17],[17,19,20,21,22,23],[15,17],[0,1],[15]],
                    [[12],[12],[10],[10],[10,11],[8,9,10],[10,11],[8,9,10],[8,9,10,11],[4,6,8,9],[9,11],[2,3,4,5,7,9],[13],[2,3]],
                    [[13],[13],[11],[11],[9,11],[11],[9],[11],[5,7,8,9],[8,9,10,11],[2,3,4,5,6,8],[8,10],[2,3],[0,1]],
                    [[1],[0,1],[22,23],[22,23],[20,21],[20,21,22],[21],[20,21],[16,17,19,21],[16,17,18,20],[14,15,17],[14,15,17],[15],[14,15]]        
                ]

        elif self.relative_qsr_choice == 3:
            # A model [(Herrmann, 1990),(Hernandez, 1994)] from 'moratz2002spatial'
            # used for getting semantic costmap
            self.relative_qsr_dict = {
                'front': 0,
                'left': 1,
                'back': 2,
                'right': 3
            }

        elif self.relative_qsr_choice == 4:
            # Model for combined expressions from 'moratz2002spatial'
            # used for getting semantic costmap
            self.relative_qsr_dict = {
                'left-front': 0,
                'left-back': 1,
                'right-front': 2,
                'right-back': 3,
                'straight-front': 4,
                'exactly-left': 5,
                'straight-back': 6,
                'exactly-right': 7
            }    

        # used for deriving NLP annotations
        self.relative_qsr_inv = {v: k for k, v in self.relative_qsr_dict.items()}

    def getRelativeQsrValue(self, r, angle, R):
        value = ''    

        if self.relative_qsr_choice == 0:
            if -self.PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif self.relative_qsr_choice == 1:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        elif self.relative_qsr_choice == 2:
            if r <= R:
                value += 'c'
            else: 
                value += 'd'    

            if angle == 0:
                value += 'sb'
            elif 0 < angle <= self.PI/4:
                value += 'lb'
            elif self.PI/4 < angle < self.PI/2:
                value += 'bl'
            elif angle == self.PI/2:
                value += 'sl'
            elif self.PI/2 < angle < 3*self.PI/4:
                value += 'fl'
            elif 3*self.PI/4 <= angle < self.PI:
                value += 'lf'
            elif angle == self.PI or angle == -self.PI:
                value += 'sf'
            elif -self.PI < angle <= -3*self.PI/4:
                value += 'rf'
            elif -3*self.PI/4 < angle < -self.PI/2:
                value += 'fr'
            elif angle == -self.PI/2:
                value += 'sr'
            elif -self.PI/2 < angle < -self.PI/4:
                value += 'br'        
            elif -self.PI/4 <= angle < 0:
                value += 'rb'

        elif self.relative_qsr_choice == 3:
            if -self.PI/4 <= angle <= self.PI/4:
                value += 'front'
            elif self.PI/4 < angle < 3*self.PI/4:
                value += 'left'
            elif 3*self.PI/4 <= angle or angle <= -3*self.PI/4:
                value += 'back'
            elif -3*self.PI/4 < angle < -self.PI/4:
                value += 'right'  

        elif self.relative_qsr_choice == 4:
            if angle == 0:
                value += 'straight-front'
            elif 0 < angle < self.PI/2:
                value += 'left-front'
            elif angle == self.PI/2:
                value += 'exactly-left'
            elif self.PI/2 < angle < self.PI:
                value += 'left-back'
            elif angle == self.PI or angle == -self.PI:
                value += 'straight-back'
            elif -self.PI < angle < -self.PI/2:
                value += 'right-back'
            elif angle == -self.PI/2:
                value += 'exactly-right'
            elif -self.PI/2 < angle < 0:
                value += 'right-front'

        return value

# Human class
class Human():
    # constructor
    def __init__(self,name):
        self.name = name
        
        self.position_map = Point(0.0,0.0,0.0)        
        self.orientation_map = Quaternion(0.0,0.0,0.0,1.0)

        self.intrinsic_qsr_choice = 1

        self.PI = PI

    # print attributes    
    def printAttributes(self):
        print('\nname = ', self.name)
        print('position_map = ', self.position_map)
        print('orientation_map = ', self.orientation_map)

        # define intrinsic QSR calculus
    def defineIntrinsicQsrCalculus(self):
        if self.intrinsic_qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.intrinsic_qsr_dict = {
                'left': 0,
                'right': 1
            }

        elif self.intrinsic_qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.intrinsic_qsr_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        # used for deriving NLP annotations
        self.qsr_dict_inv = {v: k for k, v in self.qsr_dict.items()}

    def getIntrinsicQsrValue(self, angle):
        value = ''    

        if self.intrinsic_qsr_choice == 0:
            if -self.PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif self.intrinsic_qsr_choice == 1:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        return value

# human and robot variables
robot = Robot('tiago')
human = Human('human')


# visualize robot
marker = Marker()
marker.header.frame_id = 'map'
marker.id = N_static_objects
marker.type = marker.TEXT_VIEW_FACING
marker.action = marker.ADD
marker.pose = Pose()
marker.pose.position.x = robot.position_map.x
marker.pose.position.y = robot.position_map.y
marker.pose.position.z = 2.0
marker.color.r = 1.0
marker.color.g = 0.0
marker.color.b = 0.0
marker.color.a = 1.0
marker.scale.x = 0.5
marker.scale.y = 0.5
marker.scale.z = 0.5
#marker.frame_locked = False
marker.text = robot.name
marker.ns = "my_namespace"
marker_array_semantic_labels.markers.append(marker) 

marker = Marker()
marker.header.frame_id = 'map'
marker.id = N_static_objects
marker.type = marker.ARROW
marker.action = marker.ADD
marker.pose = Pose()
marker.pose.position.x = robot.position_map.x
marker.pose.position.y = robot.position_map.y
marker.pose.position.z = -1.0
marker.pose.orientation.x = robot.orientation_map.x
marker.pose.orientation.y = robot.orientation_map.y
marker.pose.orientation.z = robot.orientation_map.z
marker.pose.orientation.w = robot.orientation_map.w
marker.color.r = 0.0
marker.color.g = 1.0
marker.color.b = 0.0
marker.color.a = 1.0
marker.scale.x = 0.8
marker.scale.y = 0.3
marker.scale.z = 0.1
marker.ns = "my_namespace"
marker_array_orientations.markers.append(marker)

    # visualize robot
marker = Marker()
marker.header.frame_id = 'map'
marker.id = N_static_objects + 1
marker.type = marker.TEXT_VIEW_FACING
marker.action = marker.ADD
marker.pose = Pose()
marker.pose.position.x = human.position_map.x
marker.pose.position.y = human.position_map.y
marker.pose.position.z = 2.0
marker.color.r = 1.0
marker.color.g = 0.0
marker.color.b = 0.0
marker.color.a = 1.0
marker.scale.x = 0.5
marker.scale.y = 0.5
marker.scale.z = 0.5
#marker.frame_locked = False
marker.text = human.name
marker.ns = "my_namespace"
marker_array_semantic_labels.markers.append(marker) 

marker = Marker()
marker.header.frame_id = 'map'
marker.id = N_static_objects + 1
marker.type = marker.ARROW
marker.action = marker.ADD
marker.pose = Pose()
marker.pose.position.x = human.position_map.x
marker.pose.position.y = human.position_map.y
marker.pose.position.z = -1.0
marker.pose.orientation.x = human.orientation_map.x
marker.pose.orientation.y = human.orientation_map.y
marker.pose.orientation.z = human.orientation_map.z
marker.pose.orientation.w = human.orientation_map.w
marker.color.r = 0.0
marker.color.g = 1.0
marker.color.b = 0.0
marker.color.a = 1.0
marker.scale.x = 0.8
marker.scale.y = 0.3
marker.scale.z = 0.1
marker.ns = "my_namespace"
marker_array_orientations.markers.append(marker)


dirCurr = os.getcwd()
dirName = 'lime_rt_data'
file_path_odom = dirName + '/odom_tmp.csv'
file_path_amcl = dirName + '/amcl_pose_tmp.csv'

class LocalCostmap():
    def __init__(self):
        self.resolution = 0.025
        self.width = 160
        self.height = 160
        self.origin_x_odom = 0.0
        self.origin_y_odom = 0.0

    def printAttributes(self):
        print('\nlocal_costmap')
        print('resolution = ', self.resolution)
        print('width = ', self.width)
        print('height = ', self.height)
        print('origin_x_odom = ', self.origin_x_odom)
        print('origin_y_odom = ', self.origin_y_odom)

local_costmap = LocalCostmap()
file_path_lc = dirName + '/costmap_info_tmp.csv'

file_path_tf_map_odom = dirName + '/tf_map_odom_tmp.csv'

class TF():
    def __init__(self):
        self.translation = Point(0.0,0.0,0.0)
        self.rotation = Quaternion(0.0,0.0,0.0,1.0)
    def printAttributes(self):
        print("\ntranslation = ", self.translation)
        print('rotation = ', self.rotation)

tf_map_odom = TF() 


class LimeExplanation():
    def __init__(self):
        self.exp = []
        self.original_deviation = 0.0

    def printAttributes(self):
        print("\nLIME explanation = ", self.exp)
        print('LIME original deviation = ', self.original_deviation)
     
lime_explanation = LimeExplanation()

# lime explanation callback
def lime_callback(msg):
    #print('\n\n\n\n\nlime_callback\n')
    print('\n\n\n\n\n')
    # [x, y, exp]*N_FEATURES + ORIGINAL_DEVIATION
    lime_explanation.exp = []
    for i in range(1, int((len(msg.data)-1)/3)): # not taking free-space weight, which is always 0
        lime_explanation.exp.append([msg.data[3*i],msg.data[3*i+1],msg.data[3*i+2]])
    lime_explanation.original_deviation = msg.data[-1]
    #lime_explanation.printAttributes()


    # update local costmap
    if os.path.getsize(file_path_lc) == 0 or os.path.exists(file_path_lc) == False:
        return
    lc_tmp = pd.read_csv(dirCurr + '/' + file_path_lc)
    local_costmap.resolution = lc_tmp.iloc[0][0]
    local_costmap.height = lc_tmp.iloc[1][0]
    local_costmap.width = lc_tmp.iloc[2][0]
    local_costmap.origin_x_odom = lc_tmp.iloc[3][0]
    local_costmap.origin_y_odom = lc_tmp.iloc[4][0] 
    #local_costmap.printAttributes()


    # update tf_map_odom
    if os.path.getsize(file_path_tf_map_odom) == 0 or os.path.exists(file_path_tf_map_odom) == False:
        return
    tf_msg = pd.read_csv(dirCurr + '/' + file_path_tf_map_odom)
    tf_map_odom.translation = Point(tf_msg.iloc[0][0],tf_msg.iloc[1][0],tf_msg.iloc[2][0])
    tf_map_odom.rotation = Quaternion(tf_msg.iloc[3][0],tf_msg.iloc[4][0],tf_msg.iloc[5][0],tf_msg.iloc[6][0])
    #tf_map_odom.printAttributes()
    t = np.asarray([tf_map_odom.translation.x,tf_map_odom.translation.y,tf_map_odom.translation.z])
    r = R.from_quat([tf_map_odom.rotation.x,tf_map_odom.rotation.y,tf_map_odom.rotation.z,tf_map_odom.rotation.w])
    r_ = np.asarray(r.as_matrix())

    # find objects in the current local costmap
    #print('\nObjects in the current local costmap:')
    objects_in_lc = []
    for i in range(0, len(static_objects_names)):
        p = np.array([static_objects_positions[i][0], static_objects_positions[i][1], 0.0])
        pnew = p.dot(r_) + t
        x_temp = int(pnew[0] - local_costmap.origin_x_odom) / local_costmap.resolution
        y_temp = int(pnew[1] - local_costmap.origin_y_odom) / local_costmap.resolution
        if 0 <= x_temp <= local_costmap.width and 0 <= y_temp <= local_costmap.height:
            objects_in_lc.append(static_objects[i])
            objects_in_lc[-1].lime_coefficients = []
            objects_in_lc[-1].lime_coefficients_distances = []
            # printing object that is in the current local costmap
            #print(static_objects[i].name)
            # update odom position of the object
            static_objects[i].position_odom = Point(pnew[0],pnew[1],pnew[2])

    N_objects_in_lc = len(objects_in_lc)        

    # append LIME coefficients to the objects in the current local costmap
    N_coefficients = len(lime_explanation.exp)
    #print('\nN of LIME coeficients = ', N_coefficients)
    lime_coeffs_string = ''
    indices_of_closest_objects_from_objects_in_lc = [0] * N_coefficients
    for i in range(0, N_coefficients):
        #[x,y,coeff]
        local_distances_of_segment_centroid_from_objects = []
        for j in range(0, N_objects_in_lc):
            dist = math.sqrt( (lime_explanation.exp[i][0] - objects_in_lc[j].position_odom.x)**2 + (lime_explanation.exp[i][1] - objects_in_lc[j].position_odom.y)**2)
            local_distances_of_segment_centroid_from_objects.append(dist)
        dist_min = min(local_distances_of_segment_centroid_from_objects)
        index_min = local_distances_of_segment_centroid_from_objects.index(dist_min)
        indices_of_closest_objects_from_objects_in_lc[i] = index_min
        objects_in_lc[index_min].lime_coefficients.append(lime_explanation.exp[i][2])
        objects_in_lc[index_min].lime_coefficients_distances.append(dist_min)
    #print('\nindices_of_closest_objects_from_objects_in_lc = ', indices_of_closest_objects_from_objects_in_lc)
    # print objects from the local costmap with their lime coefficients
    #print('\n')
    used_coefficients = []
    used_objects = []
    for i in range(0, N_objects_in_lc):
        lime_coeff_str = ''
        if len(objects_in_lc[i].lime_coefficients) > 1:
            dist_min = min(objects_in_lc[i].lime_coefficients_distances)
            index_min = objects_in_lc[i].lime_coefficients_distances.index(dist_min)
            lime_coeff_str = str(objects_in_lc[i].lime_coefficients[index_min])
            #print(objects_in_lc[i].name + ' has a LIME coefficient ' + lime_coeff_str)
            lime_coeffs_string += objects_in_lc[i].name + ' has a LIME coefficient ' + lime_coeff_str + '\n'
            used_coefficients.append(objects_in_lc[i].lime_coefficients[index_min])
            used_objects.append(i)
        elif len(objects_in_lc[i].lime_coefficients) == 1:
            lime_coeff_str = str(objects_in_lc[i].lime_coefficients[0])
            #print(objects_in_lc[i].name + ' has a LIME coefficient ' + lime_coeff_str)
            lime_coeffs_string += objects_in_lc[i].name + ' has a LIME coefficient ' + lime_coeff_str + '\n'
            used_coefficients.append(objects_in_lc[i].lime_coefficients[0])
            used_objects.append(i)
        else:
            pass

    if len(used_objects) < N_objects_in_lc:
        for i in range(0, N_objects_in_lc):
            if i in used_objects:
                continue
            local_distances = []
            local_distances_indices = []
            for j in range(0, N_coefficients):
                if lime_explanation.exp[j][2] in used_coefficients:
                    continue
                dist = math.sqrt( (lime_explanation.exp[j][0] - objects_in_lc[i].position_odom.x)**2 + (lime_explanation.exp[j][1] - objects_in_lc[i].position_odom.y)**2)
                local_distances.append(dist)
                local_distances_indices.append(j)
            if len(local_distances) == 0:
                continue    
            dist_min = min(local_distances)
            index_min = local_distances.index(dist_min)
            coeff = lime_explanation.exp[local_distances_indices[index_min]][2]
            #print(objects_in_lc[i].name + ' has a LIME coefficient ' + str(coeff))
            lime_coeffs_string += objects_in_lc[i].name + ' has a LIME coefficient ' + str(coeff) + '\n'
            used_coefficients.append(coeff)
            used_objects.append(i)

    print('\nLIME coefficients:')
    print(lime_coeffs_string)

    # how a robot passes relative to the objects in the local costmap
    #print('')
    objects_intrinsic_string = ''
    for i in range(0, N_objects_in_lc):
        if objects_in_lc[i].intrinsic_qsr == False or i not in used_objects:
            continue
        d_x = robot.position_map.x - objects_in_lc[i].position_map.x
        d_y = robot.position_map.y - objects_in_lc[i].position_map.y
        #r = math.sqrt(d_x**2+d_y**2)
        angle = np.arctan2(d_y, d_x)
        angle_ref = objects_in_lc[i].yaw_map
        angle = angle - angle_ref
        if angle >= PI:
            angle -= 2*PI
        elif angle < -PI:
            angle += 2*PI
        qsr_value = objects_in_lc[i].getIntrinsicQsrValue(angle)
        #print(robot.name + ' passes ' + qsr_value + ' of the ' + objects_in_lc[i].name)
        objects_intrinsic_string += robot.name + ' is to the ' + qsr_value + ' of the ' + objects_in_lc[i].name + '\n'

    print('\nRobot relative to the immediate objects:')
    print(objects_intrinsic_string)

    # where are objects in the local costmap relative to the robot
    #print('')
    robot_string = ''
    for i in range(0, N_objects_in_lc):
        if i not in used_objects:
            continue
        d_x = objects_in_lc[i].position_map.x - robot.position_map.x 
        d_y = objects_in_lc[i].position_map.y - robot.position_map.y 
        #r = math.sqrt(d_x**2+d_y**2)
        angle = np.arctan2(d_y, d_x)
        [angle_ref,pitch,roll] = quaternion_to_euler(robot.orientation_map.x,robot.orientation_map.y,robot.orientation_map.z,robot.orientation_map.w)
        angle = angle - angle_ref
        if angle >= PI:
            angle -= 2*PI
        elif angle < -PI:
            angle += 2*PI
        qsr_value = robot.getIntrinsicQsrValue(angle)
        #print(objects_in_lc[i].name + ' is ' + qsr_value + ' of the ' + robot.name)
        robot_string += objects_in_lc[i].name + ' is to my ' + qsr_value + '\n'
    print('\n' + robot.name + ':')
    print(robot_string)

    # visualize robot
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = N_static_objects
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = robot.position_map.x
    marker.pose.position.y = robot.position_map.y
    marker.pose.position.z = 2.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    #marker.frame_locked = False
    marker.text = robot.name
    marker.ns = "my_namespace"
    marker_array_semantic_labels.markers[-2] = copy.deepcopy(marker) 

    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = N_static_objects
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = robot.position_map.x
    marker.pose.position.y = robot.position_map.y
    marker.pose.position.z = 3.0
    marker.pose.orientation.x = robot.orientation_map.x
    marker.pose.orientation.y = robot.orientation_map.y
    marker.pose.orientation.z = robot.orientation_map.z
    marker.pose.orientation.w = robot.orientation_map.w
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.8
    marker.scale.y = 0.3
    marker.scale.z = 0.1
    marker.ns = "my_namespace"
    marker_array_orientations.markers[-2] = copy.deepcopy(marker)

    # visualize robot
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = N_static_objects + 1
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = human.position_map.x
    marker.pose.position.y = human.position_map.y
    marker.pose.position.z = 2.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    #marker.frame_locked = False
    marker.text = human.name
    marker.ns = "my_namespace"
    marker_array_semantic_labels.markers[-1] = copy.deepcopy(marker) 

    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = N_static_objects + 1
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = human.position_map.x
    marker.pose.position.y = human.position_map.y
    marker.pose.position.z = -1.0
    marker.pose.orientation.x = human.orientation_map.x
    marker.pose.orientation.y = human.orientation_map.y
    marker.pose.orientation.z = human.orientation_map.z
    marker.pose.orientation.w = human.orientation_map.w
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.8
    marker.scale.y = 0.3
    marker.scale.z = 0.1
    marker.ns = "my_namespace"
    marker_array_orientations.markers[-1] = copy.deepcopy(marker)

    # publish orientations
    pub_markers_orientations.publish(marker_array_orientations)
    # publish semantic labels
    pub_markers_semantic_labels.publish(marker_array_semantic_labels)

    '''
    nula = euler_to_quaternion(0.0,0.0,0.0)
    print('nula = ', nula)
    pi_2 = euler_to_quaternion(0.0,0.0,PI/2)
    print('pi_2 = ', pi_2)
    minus_pi_2 = euler_to_quaternion(0.0,0.0,-PI/2)
    print('minus_pi_2 = ', minus_pi_2)
    pi_ = euler_to_quaternion(0.0,0.0,PI)
    print('pi_ = ', pi_)
    minus_pi_ = euler_to_quaternion(0.0,0.0,-PI)
    print('minus_pi_ = ', minus_pi_)
    '''

    d_x = human.position_map.x - robot.position_map.x 
    d_y = human.position_map.x - robot.position_map.y 
    R_ = math.sqrt(d_x**2+d_y**2)
    
    # find objects in human POV
    #print('\nObjects in human POV:')
    objects_in_human_POV = []
    objects_in_human_POV_distances = []
    wall_blocking = False
    wall_blocking_name = ''
    for i in range(0, N_static_objects):
        d_x = static_objects[i].position_map.x - human.position_map.x 
        d_y = static_objects[i].position_map.y - human.position_map.y
        angle = np.arctan2(d_y, d_x)
        [angle_ref,pitch,roll] = quaternion_to_euler(human.orientation_map.x,human.orientation_map.y,human.orientation_map.z,human.orientation_map.w)
        angle = angle - angle_ref
        if abs(angle) <= PI/3:
            objects_in_human_POV.append(static_objects[i])
            #print(static_objects[i].name)
            r = math.sqrt(d_x**2+d_y**2) 
            objects_in_human_POV_distances.append(r)
            if 'wall' in static_objects[i].name and r < R_ and abs(angle) <= PI/5:
                wall_blocking = True
                wall_blocking_name = static_objects[i].name
                break  
    if wall_blocking:
        print('\n' + human.name + ' cannot see ' + robot.name + ' because of ' + wall_blocking_name)
    else:
        # where are objects in the local costmap relative to the human
        tpcc_string = ''
        #human_string = ''
        for i in range(0, N_objects_in_lc):
            if i not in used_objects:
                continue
            
            # TPCC part
            d_x = objects_in_lc[i].position_map.x - robot.position_map.x 
            d_y = objects_in_lc[i].position_map.y - robot.position_map.y 
            r = math.sqrt(d_x**2+d_y**2) 
            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(human.orientation_map.x,human.orientation_map.y,human.orientation_map.z,human.orientation_map.w)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = robot.getRelativeQsrValue(r,angle,R_)
            #print(objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + robot.name + ' seen by ' + human.name)
            #tpcc_string += objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + robot.name + ' seen by ' + human.name + '\n'
            tpcc_string += human.name + ',' + robot.name + ' ' + qsr_value + ' ' + objects_in_lc[i].name + '\n'

            '''
            # Human intrinsic part
            d_x = objects_in_lc[i].position_map.x - human.position_map.x 
            d_y = objects_in_lc[i].position_map.y - human.position_map.y 
            angle = np.arctan2(d_y, d_x)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = human.getIntrinsicQsrValue(angle)
            #print(objects_in_lc[i].name + ' is to my ' + qsr_value)
            human_string += objects_in_lc[i].name + ' is to my ' + qsr_value + '\n'
            '''
        
        print('\nTPCC:')
        print(tpcc_string)    

        #print('\n' + human.name + ':')
        #print(human_string)



             

update_robot_from_gazebo = False
# update human and robot variables
def model_state_callback(states_msg):
    # update robot map pose from amcl_pose
    if update_robot_from_gazebo == True:
        robot_idx = states_msg.name.index('tiago')
        robot.position_map = states_msg.pose[robot_idx].position
        robot.orientation_map = states_msg.pose[robot_idx].orientation
    # update robot map pose from amcl_pose
    else:
        try:
            if os.path.getsize(file_path_amcl) == 0 or os.path.exists(file_path_amcl) == False:
                pass
            else:
                amcl_tmp = pd.read_csv(dirCurr + '/' + file_path_amcl)
                robot.position_map = Point(amcl_tmp.iloc[0][0],amcl_tmp.iloc[1][0],0.0)
                robot.orientation_map = Quaternion(0.0,0.0,amcl_tmp.iloc[2][0],amcl_tmp.iloc[3][0])
        except:
            pass
        
    # update robot odom pose from odom
    try:
        if os.path.getsize(file_path_odom) == 0 or os.path.exists(file_path_odom) == False:
            pass
        else:
            odom_tmp = pd.read_csv(dirCurr + '/' + file_path_odom)
            robot.position_odom = Point(odom_tmp.iloc[0][0],odom_tmp.iloc[1][0],0.0)
            robot.orientation_odom = Quaternion(0.0,0.0,odom_tmp.iloc[2][0],odom_tmp.iloc[3][0])
            robot.velocities = [odom_tmp.iloc[4][0],odom_tmp.iloc[5][0]]
            #robot.printAttributes()
    except:
        pass

    # update human
    human_idx = -1
    if 'citizen_extras_female_02' in states_msg.name:
        human_idx = states_msg.name.index('citizen_extras_female_02')
    elif 'citizen_extras_female_03' in states_msg.name:
        human_idx = states_msg.name.index('citizen_extras_female_03')
    elif 'citizen_extras_male_03' in states_msg.name:
        human_idx = states_msg.name.index('citizen_extras_male_03')    
    if human_idx != -1:
        human.position_map = states_msg.pose[human_idx].position

        d_x = robot.position_map.x - human.position_map.x 
        d_y = robot.position_map.y - human.position_map.y 
        angle_yaw = np.arctan2(d_y, d_x)
        human.orientation_map = euler_to_quaternion(0.0,0.0,angle_yaw)
        #human.orientation_map = states_msg.pose[human_idx].orientation
        #human.printAttributes()


# ----------main-----------

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
#sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, model_state_callback)

#[x,y,coeff]
#sub_lime = rospy.Subscriber("/lime_exp", Float32MultiArray, lime_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()
    