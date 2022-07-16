#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import copy
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R

# global variables
PI = math.PI

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

# path variables
dirCurr = os.getcwd()
dirName = 'lime_rt_data'

# Local costmap class
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

# TF class
class TF():
    def __init__(self):
        self.translation = Point(0.0,0.0,0.0)
        self.rotation = Quaternion(0.0,0.0,0.0,1.0)
    def printAttributes(self):
        print("\ntranslation = ", self.translation)
        print('rotation = ', self.rotation)
tf_map_odom = TF() 

# orientations and semantic labels
pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
semantic_labels = MarkerArray()

pub_semantic_labels_unknown = rospy.Publisher('/semantic_labels_unknown', MarkerArray, queue_size=10)
semantic_labels_unknown = MarkerArray()

pub_orientations = rospy.Publisher('/orientations', MarkerArray, queue_size=10)
orientations = MarkerArray()

# object class
class Object():
    # constructor
    def __init__(self,ID,label,position_map,centroid_map,distance,known):
        self.ID = ID
        self.name = label
        
        self.position_map = position_map
        self.centroid_map = centroid_map
        
        self.distance = distance
        self.known = known
        
        self.orientation_map = Quaternion(0.0,0.0,0.0,1.0)
        [self.yaw_map,self.pitch_map,self.roll_map] = quaternion_to_euler(self.orientation_map.x,self.orientation_map.y,self.orientation_map.z,self.orientation_map.w)
        
        self.position_obj = Point(0.0,0.0,0.0)
        self.orientation_obj = Quaternion(0.0,0.0,0.0,1.0)

        self.orientation_odom = Quaternion(0.0,0.0,0.0,1.0)
        
        self.transform_object_to_map = [self.position_map.x, self.position_map.y, self.position_map.z, 0.0, 0.0, 0.0, 1.0]
        self.transform_map_to_object = [-self.position_map.x, -self.position_map.y, -self.position_map.z, 0.0, 0.0, 0.0, 1.0]

        self.lime_coefficients = []

        self.orientiable = False
        if 'cabinet' in self.name or 'bookshelf' in self.name or 'chair' in self.name:
            self.orientable = True
        
        # define intrinsic_qsr for cabinets and bookshelfs
        self.intrinsic_qsr = False
        if 'cabinet' in self.name or 'bookshelf' in self.name:
            self.intrinsic_qsr = True
            self.qsr_choice = 1
            self.defineIntrinsicQsrCalculus()

        self.PI = PI

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

# KNOWN OBJECTS
# [ID, label, cx_map, cy_map, dx, dy, x_map, y_map]
known_objects = []
known_objects_names =  []
known_objects_positions =  []
known_objects_ids =  []
N_known_objects = len(known_objects_names)
# semantic part
semantic_worlds_names = ['world_movable_chair', 'world_movable_chair_2', 'world_movable_chair_3', 'world_no_openable_door', 'world_openable_door']
idx = 3
known_objects_pd = pd.read_csv(dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '_tags.csv')
# fill in known objects
for i in range(0, known_objects_pd.shape[0]):
    ID = known_objects_pd.iloc[i,0]
    label = known_objects_pd.iloc[i,1]
    centroid_map = Point(known_objects_pd.iloc[i,2],known_objects_pd.iloc[i,3],0.0)
    distance = Point(known_objects_pd.iloc[i,4],known_objects_pd.iloc[i,5],0.0)
    known = True
    position_map = Point(known_objects_pd.iloc[i,6],known_objects_pd.iloc[i,7],0.0)
    known_objects.append(Object(ID,label,position_map,centroid_map,distance,known))
    known_objects_names.append(label)
    known_objects_positions.append(position_map)
    known_objects_ids.append(ID)
    N_known_objects = len(known_objects_names)

    # visualize orientations and semantic labels of known objects
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.id = i
    marker.type = marker.TEXT_VIEW_FACING
    marker.action = marker.ADD
    marker.pose = Pose()
    marker.pose.position.x = known_objects[i].position_map.x
    marker.pose.position.y = known_objects[i].position_map.y
    marker.pose.position.z = 0.5
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    #marker.frame_locked = False
    marker.text = known_objects[i].name
    marker.ns = "my_namespace"
    semantic_labels.markers.append(marker)

    if known_objects[i].orientable == True:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = i
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = known_objects[i].position_map.x
        marker.pose.position.y = known_objects[i].position_map.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = known_objects[i].orientation_map.x
        marker.pose.orientation.y = known_objects[i].orientation_map.y
        marker.pose.orientation.z = known_objects[i].orientation_map.z
        marker.pose.orientation.w = known_objects[i].orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        orientations.markers.append(marker)

# UNKNOWN OBJECTS
# [ID, label, x_odom, y_odom, x_map, y_map, cx_odom, cy_odom]
unknown_objects = []
unknown_objects_names =  []
unknown_objects_positions =  []
N_unknown_objects = len(unknown_objects_names)

# objects in LC (Local Costmap)
lc_objects = []
lc_objects_names =  []
lc_objects_positions =  []
N_lc_objects = len(lc_objects_names)


# lime explanation class
class LimeExplanation():
    def __init__(self):
        self.exp = []
        self.original_deviation = 0.0

    def printAttributes(self):
        print("\nLIME explanation = ", self.exp)
        print('LIME original deviation = ', self.original_deviation)
lime_explanation = LimeExplanation()


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


# QSR class
class qsr_rt():
    # initialization
    def __init__(self):
        # Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

        # human and robot variables
        self.human = Human('human')
        self.robot = Robot('tiago')

        # visualize robot label
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.robot.position_map.x
        marker.pose.position.y = self.robot.position_map.y
        marker.pose.position.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        #marker.frame_locked = False
        marker.text = self.robot.name
        marker.ns = "my_namespace"
        semantic_labels.markers.append(marker) 

        # visualize robot orientation
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.robot.position_map.x
        marker.pose.position.y = self.robot.position_map.y
        marker.pose.position.z = -1.0
        marker.pose.orientation.x = self.robot.orientation_map.x
        marker.pose.orientation.y = self.robot.orientation_map.y
        marker.pose.orientation.z = self.robot.orientation_map.z
        marker.pose.orientation.w = self.robot.orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        orientations.markers.append(marker)

        # visualize human label
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects + 1
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.human.position_map.x
        marker.pose.position.y = self.human.position_map.y
        marker.pose.position.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        #marker.frame_locked = False
        marker.text = self.human.name
        marker.ns = "my_namespace"
        semantic_labels.markers.append(marker) 

        # visualize human orientation
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects + 1
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.human.position_map.x
        marker.pose.position.y = self.human.position_map.y
        marker.pose.position.z = -1.0
        marker.pose.orientation.x = self.human.orientation_map.x
        marker.pose.orientation.y = self.human.orientation_map.y
        marker.pose.orientation.z = self.human.orientation_map.z
        marker.pose.orientation.w = self.human.orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        orientations.markers.append(marker)

        # update robot data from Gazebo?
        self.update_robot_from_gazebo = True

        # path variables
        self.file_path_odom = dirName + '/odom_tmp.csv'
        self.file_path_amcl = dirName + '/amcl_pose_tmp.csv'

        # lime explanation subscribers
        # N_segments * (label, coefficient) + (original_deviation)
        self.sub_lime = rospy.Subscriber("/lime_exp", Float32MultiArray, self.lime_callback)

    # update human and robot variables
    def model_state_callback(self, states_msg):
        # publish semantic labels of the unknown objects
        pub_semantic_labels_unknown.publish(semantic_labels_unknown)

        # update robot map pose from amcl_pose
        if self.update_robot_from_gazebo == True:
            robot_idx = states_msg.name.index('tiago')
            self.robot.position_map = states_msg.pose[robot_idx].position
            self.robot.orientation_map = states_msg.pose[robot_idx].orientation
        # update robot map pose from amcl_pose
        else:
            try:
                if os.path.getsize(self.file_path_amcl) == 0 or os.path.exists(self.file_path_amcl) == False:
                    pass
                else:
                    amcl_tmp = pd.read_csv(dirCurr + '/' + dirName + '/' + self.file_path_amcl)
                    self.robot.position_map = Point(amcl_tmp.iloc[0][0],amcl_tmp.iloc[1][0],0.0)
                    self.robot.orientation_map = Quaternion(0.0,0.0,amcl_tmp.iloc[2][0],amcl_tmp.iloc[3][0])
            except:
                pass
            
        # update robot odom pose from odom
        try:
            if os.path.getsize(self.file_path_odom) == 0 or os.path.exists(self.file_path_odom) == False:
                pass
            else:
                odom_tmp = pd.read_csv(dirCurr + '/' + dirName + '/' + self.file_path_odom)
                self.robot.position_odom = Point(odom_tmp.iloc[0][0],odom_tmp.iloc[1][0],0.0)
                self.robot.orientation_odom = Quaternion(0.0,0.0,odom_tmp.iloc[2][0],odom_tmp.iloc[3][0])
                self.robot.velocity_vector = [odom_tmp.iloc[4][0],odom_tmp.iloc[5][0]]
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
            
            self.human.position_map = states_msg.pose[human_idx].position
            d_x = self.robot.position_map.x - self.human.position_map.x 
            d_y = self.robot.position_map.y - self.human.position_map.y 
            angle_yaw = np.arctan2(d_y, d_x)
            self.human.orientation_map = euler_to_quaternion(0.0,0.0,angle_yaw)
            #human.orientation_map = states_msg.pose[human_idx].orientation
            #human.printAttributes()

        # visualize robot label
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.robot.position_map.x
        marker.pose.position.y = self.robot.position_map.y
        marker.pose.position.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        #marker.frame_locked = False
        marker.text = self.robot.name
        marker.ns = "my_namespace"
        semantic_labels.markers[N_known_objects] = copy.deepcopy(marker) 

        # visualize robot orientation
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.robot.position_map.x
        marker.pose.position.y = self.robot.position_map.y
        marker.pose.position.z = 3.0
        marker.pose.orientation.x = self.robot.orientation_map.x
        marker.pose.orientation.y = self.robot.orientation_map.y
        marker.pose.orientation.z = self.robot.orientation_map.z
        marker.pose.orientation.w = self.robot.orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        orientations.markers[-2] = copy.deepcopy(marker)

        # visualize human label
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects + 1
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.human.position_map.x
        marker.pose.position.y = self.human.position_map.y
        marker.pose.position.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        #marker.frame_locked = False
        marker.text = self.human.name
        marker.ns = "my_namespace"
        semantic_labels.markers[N_known_objects + 1] = copy.deepcopy(marker) 

        # visualize human orientation
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = N_known_objects + 1
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = self.human.position_map.x
        marker.pose.position.y = self.human.position_map.y
        marker.pose.position.z = -1.0
        marker.pose.orientation.x = self.human.orientation_map.x
        marker.pose.orientation.y = self.human.orientation_map.y
        marker.pose.orientation.z = self.human.orientation_map.z
        marker.pose.orientation.w = self.human.orientation_map.w
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.ns = "my_namespace"
        orientations.markers[-1] = copy.deepcopy(marker)

        # publish orientations
        #print('len(marker_array_orientations) = ', len(marker_array_orientations.markers))
        orientations.publish(orientations)
        # publish semantic labels
        #print('len(pub_markers_semantic_labels) = ', len(marker_array_semantic_labels.markers))
        semantic_labels.publish(semantic_labels)

    # lime explanation callback
    def lime_callback(self, msg):
        #print('\n\n\n\n\nlime_callback\n')
        print('\n\n\n\n\n')
        # [v-ID, coeff]*N_FEATURES + ORIGINAL_DEVIATION
        #print('msg = ', msg)
        lime_explanation.exp = []
        for i in range(0, int((len(msg.data)-1)/2)): # not taking free-space weight, which is always 0
            lime_explanation.exp.append([msg.data[2*i],msg.data[2*i+1]])
        lime_explanation.original_deviation = msg.data[-1]
        lime_explanation.printAttributes()

        # update local costmap
        if os.path.getsize(self.file_path_lc) == 0 or os.path.exists(self.file_path_lc) == False:
            return
        lc_tmp = pd.read_csv(dirCurr + '/' + self.file_path_lc)
        local_costmap.resolution = lc_tmp.iloc[0][0]
        local_costmap.height = lc_tmp.iloc[1][0]
        local_costmap.width = lc_tmp.iloc[2][0]
        local_costmap.origin_x_odom = lc_tmp.iloc[3][0]
        local_costmap.origin_y_odom = lc_tmp.iloc[4][0] 
        #local_costmap.printAttributes()

        # update tf_map_odom
        '''
        if os.path.getsize(file_path_tf_map_odom) == 0 or os.path.exists(file_path_tf_map_odom) == False:
            return
        tf_msg = pd.read_csv(dirCurr + '/' + file_path_tf_map_odom)
        tf_map_odom.translation = Point(tf_msg.iloc[0][0],tf_msg.iloc[1][0],tf_msg.iloc[2][0])
        tf_map_odom.rotation = Quaternion(tf_msg.iloc[3][0],tf_msg.iloc[4][0],tf_msg.iloc[5][0],tf_msg.iloc[6][0])
        #tf_map_odom.printAttributes()
        t = np.asarray([tf_map_odom.translation.x,tf_map_odom.translation.y,tf_map_odom.translation.z])
        r = R.from_quat([tf_map_odom.rotation.x,tf_map_odom.rotation.y,tf_map_odom.rotation.z,tf_map_odom.rotation.w])
        r_ = np.asarray(r.as_matrix())
        '''

        # find objects in the current local costmap
        # [ID, label, x_odom, y_odom, x_map, y_map, cx_odom, cy_odom]
        unknown_objects_pd = pd.read_csv(dirCurr + '/' + dirName + '/unknown_objects.csv')
        #print('\nunknown_objects_pd = ', unknown_objects_pd)
        N_unknown_objects = unknown_objects_pd.shape[0]
        lc_labels_pd = np.array(pd.read_csv(dirCurr + '/' + dirName + '/lc_labels.csv'))
        #print('lc_labels_pd = ', lc_labels_pd)
        #print('lc_labels_pd.shape = ', lc_labels_pd.shape)
        N_objects_in_lc = lc_labels_pd.shape[0]
        objects_in_lc = []
        for i in range(0, N_objects_in_lc):
            if lc_labels_pd[i, 0] in known_objects_names:
                idx = known_objects_names.index(lc_labels_pd[i, 0])
                objects_in_lc.append(known_objects[idx])

        # append LIME coefficients to the objects in the current local costmap
        N_coefficients = len(lime_explanation.exp)
        #print('\nN of LIME coeficients = ', N_coefficients)
        #print('\nlime_explanation.exp = ', lime_explanation.exp)
        lime_coeffs_string = ''
        for i in range(0, N_coefficients):
            #[v,coeff]*N_coefficients + original_deviation
            coeff = lime_explanation.exp[i][1]
            v = lime_explanation.exp[i][0]
            #print('(v, coeff) = ', (v, coeff))
            found_unknown = False
            # first check unknown objects
            for j in range(0, N_unknown_objects):
                #print('unknown_objects_pd.iloc[j, 0] = ', unknown_objects_pd.iloc[j, 0])
                if v == unknown_objects_pd.iloc[j, 0]:
                    lime_coeffs_string += unknown_objects_pd.iloc[j, 1] + ' has a LIME coefficient ' + str(coeff) + '\n'
                    found_unknown = True
            if found_unknown == False:
                # then check unknown objects
                for j in range(0, N_known_objects):
                    if v == known_objects_ids[j]:
                        lime_coeffs_string += known_objects_names[j] + ' has a LIME coefficient ' + str(coeff) + '\n'
        print('\nLIME coefficients:')
        print(lime_coeffs_string)


        # how a robot passes relative to the objects in the local costmap
        #print('')
        objects_intrinsic_string = ''
        for i in range(0, len(objects_in_lc)):
            if objects_in_lc[i].intrinsic_qsr == False:
                continue
            d_x = self.robot.position_map.x - objects_in_lc[i].position_map.x
            d_y = self.robot.position_map.y - objects_in_lc[i].position_map.y
            #r = math.sqrt(d_x**2+d_y**2)
            angle = np.arctan2(d_y, d_x)
            angle_ref = objects_in_lc[i].yaw_map
            angle = angle - angle_ref
            if angle >= self.PI:
                angle -= 2*self.PI
            elif angle < -self.PI:
                angle += 2*self.PI
            qsr_value = objects_in_lc[i].getIntrinsicQsrValue(angle)
            #print(robot.name + ' passes ' + qsr_value + ' of the ' + objects_in_lc[i].name)
            objects_intrinsic_string += self.robot.name + ' is to the ' + qsr_value + ' of the ' + objects_in_lc[i].name + '\n'

        print('\nRobot relative to the immediate objects:')
        print(objects_intrinsic_string)

        # where are objects in the local costmap relative to the robot
        #print('')
        robot_string = ''
        for i in range(0, len(objects_in_lc)):
            d_x = objects_in_lc[i].position_map.x - self.robot.position_map.x 
            d_y = objects_in_lc[i].position_map.y - self.robot.position_map.y 
            #r = math.sqrt(d_x**2+d_y**2)
            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(self.robot.orientation_map.x,self.robot.orientation_map.y,self.robot.orientation_map.z,self.robot.orientation_map.w)
            angle = angle - angle_ref
            if angle >= self.PI:
                angle -= 2*self.PI
            elif angle < -self.PI:
                angle += 2*self.PI
            qsr_value = self.robot.getIntrinsicQsrValue(angle)
            #print(objects_in_lc[i].name + ' is ' + qsr_value + ' of the ' + robot.name)
            robot_string += objects_in_lc[i].name + ' is to my ' + qsr_value + '\n'
        for j in range(0, N_unknown_objects):
            if unknown_objects_pd.iloc[j, 1] in lc_labels_pd:
                d_x = unknown_objects_pd.iloc[j, 2] - self.robot.position_map.x 
                d_y = unknown_objects_pd.iloc[j, 3] - self.robot.position_map.y 
                angle = np.arctan2(d_y, d_x)
                [angle_ref,pitch,roll] = quaternion_to_euler(self.robot.orientation_map.x,self.robot.orientation_map.y,self.robot.orientation_map.z,self.robot.orientation_map.w)
                angle = angle - angle_ref
                if angle >= self.PI:
                    angle -= 2*self.PI
                elif angle < -self.PI:
                    angle += 2*self.PI
                qsr_value = self.robot.getIntrinsicQsrValue(angle)
                robot_string += unknown_objects_pd.iloc[j, 1] + ' is to my ' + qsr_value + '\n'
        print('\n' + self.robot.name + ':')
        print(robot_string)


        d_x = self.human.position_map.x - self.robot.position_map.x 
        d_y = self.human.position_map.y - self.robot.position_map.y 
        R_ = math.sqrt(d_x**2+d_y**2)
        
        # find objects in human POV - currently only working with known objects
        #print('\nObjects in human POV:')
        objects_in_human_POV = []
        objects_in_human_POV_distances = []
        wall_blocking = False
        wall_blocking_name = ''
        for i in range(0, N_known_objects):
            d_x = known_objects[i].position_map.x - self.human.position_map.x 
            d_y = known_objects[i].position_map.y - self.human.position_map.y
            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(self.human.orientation_map.x,self.human.orientation_map.y,self.human.orientation_map.z,self.human.orientation_map.w)
            angle = angle - angle_ref
            if abs(angle) <= self.PI/3:
                objects_in_human_POV.append(known_objects[i])
                #print(static_objects[i].name)
                r = math.sqrt(d_x**2+d_y**2) 
                objects_in_human_POV_distances.append(r)
                #print('known_objects[i].name,r,R = ', known_objects[i].name,r,R_)
                if 'wall' in known_objects[i].name and r < R_ and abs(angle) <= self.PI/3:
                    wall_blocking = True
                    wall_blocking_name = known_objects[i].name
                    break  
        if wall_blocking:
            print('\n' + self.human.name + ' cannot see ' + self.robot.name + ' because of ' + wall_blocking_name)
        else:
            # where are objects in the local costmap relative to the human
            tpcc_string = ''
            #human_string = ''
            for i in range(0, len(objects_in_lc)):
                # TPCC part
                d_x = objects_in_lc[i].position_map.x - self.robot.position_map.x 
                d_y = objects_in_lc[i].position_map.y - self.robot.position_map.y 
                r = math.sqrt(d_x**2+d_y**2) 
                angle = np.arctan2(d_y, d_x)
                [angle_ref,pitch,roll] = quaternion_to_euler(self.human.orientation_map.x,self.human.orientation_map.y,self.human.orientation_map.z,self.human.orientation_map.w)
                angle = angle - angle_ref
                if angle >= self.PI:
                    angle -= 2*self.PI
                elif angle < -self.PI:
                    angle += 2*self.PI
                qsr_value = self.robot.getRelativeQsrValue(r,angle,R_)
                #print(objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + robot.name + ' seen by ' + human.name)
                #tpcc_string += objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + robot.name + ' seen by ' + human.name + '\n'
                tpcc_string += self.human.name + ',' + self.robot.name + ' ' + qsr_value + ' ' + objects_in_lc[i].name + '\n'

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

        #marker_array_semantic_labels_unknown = MarkerArray()

        #lc_labels_pd = [l for lc in lc_labels_pd for l in lc]
        #print(lc_labels_pd)
        # action and openable unknown objects
        for i in range(0, N_unknown_objects):
            if unknown_objects_pd.iloc[i, 1] in lc_labels_pd:
                #print('USAO')
                if 'door' in unknown_objects_pd.iloc[j, 1]:
                    print('\nPlease open the ' + str(unknown_objects_pd.iloc[i, 1]) + ' so I can proceed!')
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.TEXT_VIEW_FACING
                    marker.action = marker.ADD
                    marker.pose = Pose()
                    marker.pose.position.x = unknown_objects_pd.iloc[i, 2]
                    marker.pose.position.y = unknown_objects_pd.iloc[i, 3]
                    marker.pose.position.z = 2.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    #marker.frame_locked = False
                    marker.text = unknown_objects_pd.iloc[i, 1]
                    marker.ns = "my_namespace"
                    self.semantic_labels_unknown.markers.append(marker)
                    self.pub_semantic_labels_unknown.publish(self.semantic_labels_unknown)
                elif 'chair' in unknown_objects_pd.iloc[i, 1]: 
                    #print(unknown_objects_pd.iloc[i, 1])   
                    for j in range(0, len(objects_in_lc)):
                        if objects_in_lc[j].intrinsic_qsr == True:
                            d_x = unknown_objects_pd.iloc[i, 2] - objects_in_lc[j].position_map.x 
                            d_y = unknown_objects_pd.iloc[i, 3] - objects_in_lc[j].position_map.y 
                            r = math.sqrt(d_x**2+d_y**2) 
                            angle = np.arctan2(d_y, d_x)
                            [angle_ref,pitch,roll] = quaternion_to_euler(objects_in_lc[j].orientation_map.x,objects_in_lc[j].orientation_map.y,objects_in_lc[j].orientation_map.z,objects_in_lc[j].orientation_map.w)
                            angle = angle - angle_ref
                            if angle >= self.PI:
                                angle -= 2*self.PI
                            elif angle < -self.PI:
                                angle += 2*self.PI
                            qsr_value = objects_in_lc[j].getIntrinsicQsrValue_(angle)
                            r_unknown_object = copy.deepcopy(r)
                            angle_unknown_object = copy.deepcopy(angle)
                            qsr_unknown_object = copy.deepcopy(qsr_value)

                            d_x = self.robot.position_map.x - objects_in_lc[j].position_map.x 
                            d_y = self.robot.position_map.y - objects_in_lc[j].position_map.y 
                            r = math.sqrt(d_x**2+d_y**2) 
                            angle = np.arctan2(d_y, d_x)
                            [angle_ref,pitch,roll] = quaternion_to_euler(objects_in_lc[j].orientation_map.x,objects_in_lc[j].orientation_map.y,objects_in_lc[j].orientation_map.z,objects_in_lc[j].orientation_map.w)
                            angle = angle - angle_ref
                            if angle >= self.PI:
                                angle -= 2*self.PI
                            elif angle < -self.PI:
                                angle += 2*self.PI
                            qsr_value = objects_in_lc[j].getIntrinsicQsrValue(angle)
                            if 'left' in qsr_value:
                                robot_dir = 'left'                                        
                            else:
                                robot_dir = 'right'
                            qsr_vals = []    
                            for k in range(0, len(objects_in_lc)):
                                if k != j:
                                    d_x = objects_in_lc[k].position_map.x - objects_in_lc[j].position_map.x 
                                    d_y = objects_in_lc[k].position_map.y - objects_in_lc[j].position_map.y 
                                    r = math.sqrt(d_x**2+d_y**2) 
                                    angle = np.arctan2(d_y, d_x)
                                    [angle_ref,pitch,roll] = quaternion_to_euler(objects_in_lc[j].orientation_map.x,objects_in_lc[j].orientation_map.y,objects_in_lc[j].orientation_map.z,objects_in_lc[j].orientation_map.w)
                                    angle = angle - angle_ref
                                    if angle >= self.PI:
                                        angle -= 2*self.PI
                                    elif angle < -self.PI:
                                        angle += 2*self.PI
                                    qsr_value = objects_in_lc[j].getIntrinsicQsrValue_(angle)            
                                    qsr_vals.append(qsr_value)
                            if robot_dir == 'left':
                                if 'right' not in qsr_vals:
                                    print('\nPlease move the ' + unknown_objects_pd.iloc[i, 1] + ' to the ' + 'right' + ' of the ' + objects_in_lc[j].name + ' so I can proceed!')
                                    marker = Marker()
                                    marker.header.frame_id = 'map'
                                    marker.id = i
                                    marker.type = marker.TEXT_VIEW_FACING
                                    marker.action = marker.ADD
                                    marker.pose = Pose()
                                    marker.pose.position.x = unknown_objects_pd.iloc[i, 2]
                                    marker.pose.position.y = unknown_objects_pd.iloc[i, 3]
                                    marker.pose.position.z = 2.0
                                    marker.color.r = 1.0
                                    marker.color.g = 0.0
                                    marker.color.b = 0.0
                                    marker.color.a = 1.0
                                    marker.scale.x = 0.5
                                    marker.scale.y = 0.5
                                    marker.scale.z = 0.5
                                    #marker.frame_locked = False
                                    marker.text = unknown_objects_pd.iloc[i, 1]
                                    marker.ns = "my_namespace"
                                    self.semantic_labels_unknown.markers.append(marker)
                                    self.pub_semantic_labels_unknown.publish(self.semantic_labels_unknown)
                                    break
                            else:
                                if 'left' not in qsr_vals:
                                    marker = Marker()
                                    marker.header.frame_id = 'map'
                                    marker.id = i
                                    marker.type = marker.TEXT_VIEW_FACING
                                    marker.action = marker.ADD
                                    marker.pose = Pose()
                                    marker.pose.position.x = unknown_objects_pd.iloc[i, 2]
                                    marker.pose.position.y = unknown_objects_pd.iloc[i, 3]
                                    marker.pose.position.z = 2.0
                                    marker.color.r = 1.0
                                    marker.color.g = 0.0
                                    marker.color.b = 0.0
                                    marker.color.a = 1.0
                                    marker.scale.x = 0.5
                                    marker.scale.y = 0.5
                                    marker.scale.z = 0.5
                                    #marker.frame_locked = False
                                    marker.text = unknown_objects_pd.iloc[i, 1]
                                    marker.ns = "my_namespace"
                                    self.semantic_labels_unknown.markers.append(marker)
                                    self.pub_semantic_labels_unknown.publish(self.semantic_labels_unknown)
                                    print('\nPlease move the ' + unknown_objects_pd.iloc[i, 1] + ' to the ' + 'left' + ' of the ' + objects_in_lc[j].name + ' so I can proceed!')
                                    break



########--------------- MAIN -------------#############
qsr_rt_obj = qsr_rt()

# Initialize the ROS Node named 'qsr_rt', allow multiple nodes to be run with this name
rospy.init_node('qsr_rt', anonymous=True)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()