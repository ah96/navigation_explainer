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
import time

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

# semantic labels and orientations
pub_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
semantic_labels = MarkerArray()

pub_orientations = rospy.Publisher('/orientations', MarkerArray, queue_size=10)
orientations = MarkerArray()

pub_semantic_labels_unknown = rospy.Publisher('/semantic_labels_unknown', MarkerArray, queue_size=10)
semantic_labels_unknown = MarkerArray()

pub_orientations_unknown = rospy.Publisher('/orientations_unknown', MarkerArray, queue_size=10)
orientations_unknown = MarkerArray()

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

        self.orientable = False
        if 'cabinet' in self.name or 'bookshelf' in self.name or 'chair' in self.name or 'citizen' in self.name:
            self.orientable = True

        self.openable = False
        if 'door' in self.name:
            self.openable = True

        self.moveable = False
        if 'chair' in self.name or 'table' in self.name:
            self.moveable = True

        self.is_human = False
        if 'citizen' in self.name:
            self.is_human = True    

        # define intrinsic_qsr for cabinets and bookshelfs
        self.intrinsic_qsr = False
        if 'cabinet' in self.name or 'bookshelf' in self.name or 'chair' in self.name or 'citizen' in self.name:
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
# semantic part
semantic_worlds_names = ['world_movable_chair', 'world_no_openable_door', 'world_openable_door']
idx = 0
known_objects_pd = pd.read_csv(dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '_tags.csv')
# fill in known objects
known_objects = []
known_objects_names =  []
known_objects_positions =  []
known_objects_ids =  []
N_known_objects = len(known_objects_names)
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

# OBJECTS IN LC (Local Costmap)
lc_objects = []
lc_objects_names =  []
lc_objects_positions =  []

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
        self.humans = []
        self.robot = Robot('tiago')

        self.PI = PI

        self.pub_semantic_labels_rh = rospy.Publisher('/semantic_labels_rh', MarkerArray, queue_size=10)
        self.semantic_labels_rh = MarkerArray()

        self.pub_orientations_rh = rospy.Publisher('/orientations_rh', MarkerArray, queue_size=10)
        self.orientations_rh = MarkerArray()

        # update robot data from Gazebo or not
        self.update_robot_from_gazebo = True

        # path variables
        self.file_path_odom = dirName + '/odom_tmp.csv'
        self.file_path_amcl = dirName + '/amcl_pose_tmp.csv'
        self.file_path_lc = dirName + '/costmap_info_tmp.csv'

        # lime explanation subscribers
        # N_segments * (label, coefficient) + (original_deviation)
        self.sub_lime = rospy.Subscriber("/lime_rt_exp", Float32MultiArray, self.lime_callback)

    # update human and robot variables
    def model_state_callback(self, states_msg):
        # ROBOT
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

        self.semantic_labels_rh.markers = []
        self.orientations_rh.markers = []

        # visualize robot label
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = 0
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
        self.semantic_labels_rh.markers.append(marker) 

        # visualize robot orientation
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = 0
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
        self.orientations_rh.markers.append(marker)

        # HUMANS
        # update humans
        self.humans = []
        for i in range(0, len(states_msg.name)):
            if 'citizen' in states_msg.name[i]:
                human = Human(states_msg.name[i])
                human.position_map = states_msg.pose[i].position
                #d_x = self.robot.position_map.x - human.position_map.x 
                #d_y = self.robot.position_map.y - human.position_map.y 
                #angle_yaw = np.arctan2(d_y, d_x)
                #human.orientation_map = euler_to_quaternion(0.0, 0.0, angle_yaw)
                human.orientation_map = states_msg.pose[i].orientation
                #human.printAttributes()
                self.humans.append(human)

                # visualize human label
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = len(self.semantic_labels_rh.markers)
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
                self.semantic_labels_rh.markers.append(marker) 

                # visualize human orientation
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = len(self.orientations_rh.markers)
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
                self.orientations_rh.markers.append(marker)

        # publish orientations of robots and humans
        self.pub_orientations_rh.publish(self.orientations_rh)
        # publish semantic labels of robots and humans
        self.pub_semantic_labels_rh.publish(self.semantic_labels_rh)

        # publish orientations of known objects
        pub_orientations.publish(orientations)
        # publish semantic labels of known objects
        pub_semantic_labels.publish(semantic_labels)

        # publish orientations of known objects
        pub_orientations_unknown.publish(orientations_unknown)
        # publish semantic labels of unknown objects
        pub_semantic_labels_unknown.publish(semantic_labels_unknown)

    # lime explanation callback
    def lime_callback(self, msg):
        start = time.time()

        try:
            #print('\n\n\n\nlime_callback\n')

            # update tf_map_odom - currently is not needed
            '''
            if os.path.getsize(file_path_tf_map_odom) == 0 or os.path.exists(file_path_tf_map_odom) == False:
                return
            tf_msg = pd.read_csv(dirCurr + '/' + dirName + '/' + file_path_tf_map_odom)
            tf_map_odom.translation = Point(tf_msg.iloc[0][0],tf_msg.iloc[1][0],tf_msg.iloc[2][0])
            tf_map_odom.rotation = Quaternion(tf_msg.iloc[3][0],tf_msg.iloc[4][0],tf_msg.iloc[5][0],tf_msg.iloc[6][0])
            #tf_map_odom.printAttributes()
            t = np.asarray([tf_map_odom.translation.x,tf_map_odom.translation.y,tf_map_odom.translation.z])
            r = R.from_quat([tf_map_odom.rotation.x,tf_map_odom.rotation.y,tf_map_odom.rotation.z,tf_map_odom.rotation.w])
            r_ = np.asarray(r.as_matrix())
            '''

            # get the LIME explanation
            # N_segments * (label, coefficient) + (original_deviation)
            lime_explanation.exp = []
            for i in range(0, int((len(msg.data)-1)/2)):
                lime_explanation.exp.append([msg.data[2*i],msg.data[2*i+1]])
            lime_explanation.original_deviation = msg.data[-1]
            print('\nnumber of lime weights = ', len(lime_explanation.exp))
            lime_explanation.printAttributes()

            # update local costmap variables
            if os.path.getsize(self.file_path_lc) == 0 or os.path.exists(self.file_path_lc) == False:
                return
            lc_tmp = pd.read_csv(dirCurr + '/' + self.file_path_lc)
            local_costmap.resolution = lc_tmp.iloc[0][0]
            local_costmap.height = lc_tmp.iloc[1][0]
            local_costmap.width = lc_tmp.iloc[2][0]
            local_costmap.origin_x_odom = lc_tmp.iloc[3][0]
            local_costmap.origin_y_odom = lc_tmp.iloc[4][0] 
            #local_costmap.printAttributes()

            # load unknown_objects and lc_labels
            # [ID, label, x_map, y_map, x_pixel, y_pixel]
            N_unknown_objects = 0
            N_objects_in_lc = 0
            try:
                #unknown_objects_pd = pd.read_csv(dirCurr + '/' + dirName + '/unknown_objects.csv')
                #N_unknown_objects = unknown_objects_pd.shape[0]
                #print('N_unknown_objects = ', N_unknown_objects)

                lc_objects_pd = np.array(pd.read_csv(dirCurr + '/' + dirName + '/lc_objects.csv'))
                N_objects_in_lc = lc_objects_pd.shape[0]
                print('N_objects_in_lc = ', N_objects_in_lc)
            except:
                pass

            # append LIME coefficients to the objects in the current local costmap
            N_coefficients = len(lime_explanation.exp)
            lime_coeffs_string = ''
            for i in range(0, N_coefficients):
                #[v,coeff]*N_coefficients + original_deviation
                coeff = lime_explanation.exp[i][1]
                v = lime_explanation.exp[i][0]
                # first check unknown objects
                for j in range(0, N_objects_in_lc):
                    if v == lc_objects_pd[j, 0]:
                        lime_coeffs_string += lc_objects_pd[j, 1] + ' has a LIME coefficient ' + str(coeff) + '\n'
                        break
            print('\nLIME coefficients:')
            print(lime_coeffs_string)

                
            # populate objects_in_lc with objects from LC
            objects_in_lc = []
            # populate also a list of uknown objects
            unknown_objects = []
            semantic_labels_unknown.markers = []
            orientations_unknown.markers = []
            for i in range(0, N_objects_in_lc):
                # if an object is a known object, just find it in the global list of known objects
                # so far adding real centroids even of the objects whose centroids are not in the LC
                # KNOWN OBJECT
                if lc_objects_pd[i, 1] in known_objects_names:
                    idx = known_objects_names.index(lc_objects_pd[i, 1])
                    objects_in_lc.append(known_objects[idx])
                # if an object could not be found in a global list of known objects, then it is an unknown one, and we create a new Object class instance
                # orientation of the unknown objects from Gazebo is still not included
                # UNKNOWN OBJECT
                else:
                    pos_map = Point(lc_objects_pd[i, 2],lc_objects_pd[i, 3],0.0)
                    unknown_objects.append(Object(ID=lc_objects_pd[i, 0],label=lc_objects_pd[i, 1],position_map=pos_map,centroid_map=pos_map,distance=0,known=False))
                    #print('unknown object - ' + lc_objects_pd[i, 1])
                    objects_in_lc.append(unknown_objects[-1])

                    # visualize semantic label of unknown object
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.TEXT_VIEW_FACING
                    marker.action = marker.ADD
                    marker.pose = Pose()
                    marker.pose.position.x = pos_map.x
                    marker.pose.position.y = pos_map.y
                    marker.pose.position.z = 2.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    #marker.frame_locked = False
                    marker.text = lc_objects_pd[i, 1]
                    marker.ns = "my_namespace"
                    semantic_labels_unknown.markers.append(marker)
                    pub_semantic_labels_unknown.publish(semantic_labels_unknown)

                    # visualize orientation of unknown object
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.id = i
                    marker.type = marker.ARROW
                    marker.action = marker.ADD
                    marker.pose = Pose()
                    marker.pose.position.x = pos_map.x
                    marker.pose.position.y = pos_map.y
                    marker.pose.position.z = 3.0
                    marker.pose.orientation.x = objects_in_lc[i].orientation_map.x
                    marker.pose.orientation.y = objects_in_lc[i].orientation_map.y
                    marker.pose.orientation.z = objects_in_lc[i].orientation_map.z
                    marker.pose.orientation.w = objects_in_lc[i].orientation_map.w
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.scale.x = 0.8
                    marker.scale.y = 0.3
                    marker.scale.z = 0.1
                    marker.ns = "my_namespace"
                    orientations_unknown.markers.append(marker)

            N_unknown_objects = len(unknown_objects)
            #print('N_unknown_objects = ', N_unknown_objects)

            # how a robot passes relative to the objects in the local costmap
            objects_intrinsic_string = ''
            for i in range(0, N_objects_in_lc):
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
            robot_string = ''
            for i in range(0, N_objects_in_lc):
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
                robot_string += objects_in_lc[i].name + ' is to my ' + qsr_value + '\n'
            print('\n' + self.robot.name + ':')
            print(robot_string)


            # find objects in humans' POV, as well as QSR between humans, robot and LC objects
            for i in range(0, len(self.humans)):
                human = self.humans[i]
                d_x = human.position_map.x - self.robot.position_map.x 
                d_y = human.position_map.y - self.robot.position_map.y 
                R_ = math.sqrt(d_x**2+d_y**2)
                objects_in_human_POV = []
                objects_in_human_POV_distances = []
                objects_in_human_POV_names = ''
                for j in range(0, N_known_objects):
                    d_x = known_objects[j].position_map.x - human.position_map.x 
                    d_y = known_objects[j].position_map.y - human.position_map.y
                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(human.orientation_map.x,human.orientation_map.y,human.orientation_map.z,human.orientation_map.w)
                    angle = angle - angle_ref
                    if abs(angle) <= self.PI/3:
                        objects_in_human_POV.append(known_objects[j])
                        r = math.sqrt(d_x**2+d_y**2) 
                        objects_in_human_POV_distances.append(r)
                        objects_in_human_POV_names += objects_in_human_POV[-1].name + '\n'
                for j in range(0, N_unknown_objects):
                    d_x = unknown_objects[j].position_map.x - human.position_map.x 
                    d_y = unknown_objects[j].position_map.y - human.position_map.y
                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(human.orientation_map.x,human.orientation_map.y,human.orientation_map.z,human.orientation_map.w)
                    angle = angle - angle_ref
                    if abs(angle) <= self.PI/3:
                        objects_in_human_POV.append(unknown_objects[j])
                        r = math.sqrt(d_x**2+d_y**2) 
                        objects_in_human_POV_distances.append(r)
                        objects_in_human_POV_names += objects_in_human_POV[-1].name + '\n'
                #print('\nObjects in ' + human.name + ' POV:\n' + objects_in_human_POV_names)

                # where are objects in the local costmap relative to the robot as seen from humans
                tpcc_human_string = ''
                tpcc_robot_string = ''
                human_string = ''
                for j in range(0, N_objects_in_lc):
                    if objects_in_lc[j].name == human.name:
                        continue
                    # TPCC human part
                    d_x = objects_in_lc[j].position_map.x - self.robot.position_map.x 
                    d_y = objects_in_lc[j].position_map.y - self.robot.position_map.y 
                    r = math.sqrt(d_x**2+d_y**2) 
                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(human.orientation_map.x,human.orientation_map.y,human.orientation_map.z,human.orientation_map.w)
                    angle = angle - angle_ref
                    if angle >= self.PI:
                        angle -= 2*self.PI
                    elif angle < -self.PI:
                        angle += 2*self.PI
                    qsr_value = self.robot.getRelativeQsrValue(r,angle,R_)
                    #tpcc_human_string += objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + self.robot.name + ' seen by ' + human.name + '\n'
                    tpcc_human_string += human.name + ',' + self.robot.name + ' ' + qsr_value + ' ' + objects_in_lc[j].name + '\n'

                    #'''
                    # Human intrinsic part
                    d_x = objects_in_lc[j].position_map.x - human.position_map.x 
                    d_y = objects_in_lc[j].position_map.y - human.position_map.y 
                    angle = np.arctan2(d_y, d_x)
                    angle = angle - angle_ref
                    if angle >= PI:
                        angle -= 2*PI
                    elif angle < -PI:
                        angle += 2*PI
                    qsr_value = human.getIntrinsicQsrValue(angle)
                    #print(objects_in_lc[j].name + ' is to my ' + qsr_value)
                    human_string += objects_in_lc[j].name + ' is to my ' + qsr_value + '\n'
                    #'''

                    # TPCC robot part
                    r = math.sqrt(d_x**2+d_y**2) 
                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(self.robot.orientation_map.x,self.robot.orientation_map.y,self.robot.orientation_map.z,self.robot.orientation_map.w)
                    angle = angle - angle_ref
                    if angle >= self.PI:
                        angle -= 2*self.PI
                    elif angle < -self.PI:
                        angle += 2*self.PI
                    qsr_value = self.robot.getRelativeQsrValue(r,angle,R_)
                    #tpcc_robot_string += objects_in_lc[i].name + ' is to the ' + qsr_value + ' of the ' + human.name + ' seen by ' + self.robot.name + '\n'
                    tpcc_robot_string += self.robot.name + ',' + human.name + ' ' + qsr_value + ' ' + objects_in_lc[j].name + '\n'
                '''
                print(human.name + ' TPCC:')
                print(tpcc_human_string)    

                print(self.robot.name + ' TPCC:')
                print(tpcc_robot_string)    

                print(human.name + ':')
                print(human_string)
                '''

            # ACTIONABLE PART
            transformed_plan = pd.read_csv(dirCurr + '/' + dirName + '/transformed_plan.csv')
            # fill the list of transformed plan coordinates
            transformed_plan_xs = []
            transformed_plan_ys = [] 
            for i in range(0, transformed_plan.shape[0]):
                if math.isnan(transformed_plan.iloc[i, 0]) == True or math.isnan(transformed_plan.iloc[i, 1]) == True:
                    continue
                x_temp = int((transformed_plan.iloc[i, 0] - local_costmap.origin_x_odom) / local_costmap.resolution)
                y_temp = int((transformed_plan.iloc[i, 1] - local_costmap.origin_y_odom) / local_costmap.resolution)

                if 0 <= x_temp < local_costmap.width and 0 <= y_temp < local_costmap.height:
                    transformed_plan_xs.append(x_temp)
                    transformed_plan_ys.append(y_temp)
            transformed_plan_len = len(transformed_plan_xs)
            #print('len(transformed_plan_xs) = ', transformed_plan_len)
            #print('transformed_plan_xs = ', transformed_plan_xs)

            local_plan_original = pd.read_csv(dirCurr + '/' + dirName + '/local_plan_original.csv')
            # fill the list of local plan coordinates
            local_plan_xs = []
            local_plan_ys = []
            for j in range(0, local_plan_original.shape[0]):
                x_temp = int((local_plan_original.iloc[j, 0] - local_costmap.origin_x_odom) / local_costmap.resolution)
                y_temp = int((local_plan_original.iloc[j, 1] - local_costmap.origin_y_odom) / local_costmap.resolution)
                if 0 <= x_temp < local_costmap.width and 0 <= y_temp < local_costmap.height:
                    local_plan_xs.append(x_temp)
                    local_plan_ys.append(y_temp)
            local_plan_len = len(local_plan_xs)
            #print('len(local_plan_xs) = ', local_plan_len)

            # load segments
            segments = np.array(pd.read_csv(dirCurr + '/' + dirName + '/segments.csv')) 

            # find unknown objects in the LC
            for i in range(0, N_objects_in_lc):
                for j in range(0, N_unknown_objects):
                    if objects_in_lc[i] == unknown_objects[j]:
                        print('\nfound unknown objects in LC - ' + objects_in_lc[i].name)

                        # find semantic value of this object
                        v = objects_in_lc[i].ID

                        # check whether global plan goes through the inflated area of the unknown obstacle
                        global_plan_goes_through = False
                        for k in range(0, transformed_plan_len):
                            if segments[transformed_plan_ys[k],transformed_plan_xs[k]] == v:
                                global_plan_goes_through = True
                                break

                        # check whether the local plan goes through any non-free space
                        local_plan_goes_through = False
                        if global_plan_goes_through == True:
                            print('Global plan goes through ' + objects_in_lc[i].name)
                            for k in range(0, local_plan_len):
                                if segments[local_plan_ys[k], local_plan_xs[k]] != 0:
                                    local_plan_goes_through = True
                                    break

                            if local_plan_goes_through == True:
                                if objects_in_lc[i].is_human == True:
                                    print(objects_in_lc[i].name + ', please move so I can proceed!')
                                elif objects_in_lc[i].moveable == True:
                                    print('Please move the ' + objects_in_lc[i].name + ' so I can proceed!')
                                elif objects_in_lc[i].openable == True:
                                    print('Please open the ' + objects_in_lc[i].name + ' so I can proceed!')
                                else:
                                    print('The ' + objects_in_lc[i].name + ' is blocking me. I cannot proceed!')

            end = time.time()
            with open('TIME_2.csv','a') as file:
                file.write(str(end-start))
                file.write('\n')

        except:
            end = time.time()
            with open('TIME_2.csv','a') as file:
                file.write(str(end-start))
                file.write('\n') 



                                            
########--------------- MAIN -------------#############
# Initialize the ROS Node named 'qsr_rt', allow multiple nodes to be run with this name
rospy.init_node('qsr_rt', anonymous=True)

qsr_rt_obj = qsr_rt()

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()