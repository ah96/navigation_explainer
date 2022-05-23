#!/usr/bin/env python3

from gazebo_msgs.msg import ModelStates, ModelState
import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import copy
from std_msgs.msg import Float32MultiArray
#import tf2_ros     


class qsr():
    # initialization
    def __init__(self):
        self.tpcc_dict = dict()
        self.tpcc_dict_inv = dict()
        self.angle_ref = dict()
        self.R = -1
        self.ORIGIN_pos = [0.0,0.0]
        self.ORIGIN_name = 'citizen_extras_female_02' #'citizen_extras_female_02','citizen_extras_female_03','citizen_extras_male_03'
        self.RELATUM_pos = [0.0,0.0]
        self.RELATUM_name = 'tiago'
        self.origin_pos = [0.0,0.0]
        self.origin_name = '' #'citizen_extras_female_02','citizen_extras_female_03','citizen_extras_male_03'
        self.relatum_pos = [0.0,0.0]
        self.relatum_name = ''
        self.referent_pos = [0.0,0.0]
        self.referent_name = ''
        self.referents_positions = []
        self.referents_names = []
        self.PI = math.pi
        self.pose = Pose()
        self.twist = Twist()
        self.models = []
        self.init = True
        self.triples = []
        self.marker_array_semantic_labels = MarkerArray()
        self.marker_array_orientations = MarkerArray()
        self.lime_exp = []
        self.lime_names = []
        self.lime_coeffs = []
        self.I = 0
        
        # reasoning variables
        self.original_deviation = -1
        self.comb_table = []
        self.perm_table = []
        
        # markers
        self.marker = Marker()
        self.marker_orientation = Marker()

        # objects in local costmap
        self.objects_in_lc_positions = []
        self.objects_in_lc_names = []     

    # define QSR calculus
    def defineQsrCalculus(self, qsr_choice):
        if qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.tpcc_dict = {
                'left': 0,
                'right': 1
            }

        elif qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.tpcc_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        elif qsr_choice == 2:
            # TPCC reference system
            # my modified version from 'moratz2008qualitative'
            # used for getting semantic costmap
            if self.R == 0:
                self.tpcc_dict = {
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
                self.tpcc_dict = {
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

        elif qsr_choice == 3:
            # A model [(Herrmann, 1990),(Hernandez, 1994)] from 'moratz2002spatial'
            # used for getting semantic costmap
            self.tpcc_dict = {
                'front': 0,
                'left': 1,
                'back': 2,
                'right': 3
            }

        elif qsr_choice == 4:
            # Model for combined expressions from 'moratz2002spatial'
            # used for getting semantic costmap
            self.tpcc_dict = {
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
        self.tpcc_dict_inv = {v: k for k, v in self.tpcc_dict.items()}

    # define QSR calculus
    def defineRobotQsrCalculus(self, qsr_choice):
        if qsr_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.tpcc_dict_robot = {
                'left': 0,
                'right': 1
            }

        elif qsr_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            # used for getting semantic costmap
            self.tpcc_dict_robot = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

        elif qsr_choice == 2:
            # TPCC reference system
            # my modified version from 'moratz2008qualitative'
            # used for getting semantic costmap
            if self.R == 0:
                self.tpcc_dict_robot = {
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
                self.tpcc_dict_robot = {
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

                self.comb_table_robot = [
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

                self.perm_dict_robot = {
                    'ID':0,
                    'INV':1,
                    'SC':2,
                    'SCI':3,
                    'HM':4,
                    'HMI':5
                }

                self.per_table_robot = [
                    [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]],
                    [[13],[13],[15],[15],[15],[15,17],[15],[15,17],[14,15,16,17],[16,17,19,21],[14,16],[16,18,20,21,22,23],[12],[0,1]],
                    [[12],[12],[14,16],[14,16],[14,16,18],[14],[16,18],[14],[16,17,18,20],[14,15,16,17],[17,19,20,21,22,23],[15,17],[0,1],[15]],
                    [[12],[12],[10],[10],[10,11],[8,9,10],[10,11],[8,9,10],[8,9,10,11],[4,6,8,9],[9,11],[2,3,4,5,7,9],[13],[2,3]],
                    [[13],[13],[11],[11],[9,11],[11],[9],[11],[5,7,8,9],[8,9,10,11],[2,3,4,5,6,8],[8,10],[2,3],[0,1]],
                    [[1],[0,1],[22,23],[22,23],[20,21],[20,21,22],[21],[20,21],[16,17,19,21],[16,17,18,20],[14,15,17],[14,15,17],[15],[14,15]]        
                ]

        elif qsr_choice == 3:
            # A model [(Herrmann, 1990),(Hernandez, 1994)] from 'moratz2002spatial'
            # used for getting semantic costmap
            self.tpcc_dict_robot = {
                'front': 0,
                'left': 1,
                'back': 2,
                'right': 3
            }

        elif qsr_choice == 4:
            # Model for combined expressions from 'moratz2002spatial'
            # used for getting semantic costmap
            self.tpcc_dict_robot = {
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
        self.tpcc_dict_inv_robot = {v: k for k, v in self.tpcc_dict.items()}
    
    def getRobotValue(self, angle):
        value = ''    

        if qsr_choice_ == 1:
            if 0 <= angle < self.PI/2:
                value += 'right/back'
            elif self.PI/2 <= angle <= self.PI:
                value += 'right/front'
            elif -self.PI/2 <= angle < 0:
                value += 'left/back'
            elif -self.PI <= angle < -self.PI/2:
                value += 'left/front'

        return value

    # convert orientation quaternion to euler angles
    def quaternion_to_euler(self, x, y, z, w):
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

    def updateRefSys(self):
        # reference direction == direction between RELATUM and ORIGIN
        d_x = self.RELATUM_pos[0] - self.ORIGIN_pos[0]
        d_y = self.RELATUM_pos[1] - self.ORIGIN_pos[1]
        
        self.R = math.sqrt((d_x)**2+(d_y)**2)
        self.angle_ref = np.arctan2(d_y, d_x)
        #print('angle_ref = ', angle_ref)
        #print('angle_ref (in deg) = ', angle_ref * 180 / PI)
    
    def reason(self, states_msg):
        # ABC and ABD -> BCD, we must choose C
        A_name = self.ORIGIN_name
        B_name = self.RELATUM_name
        #C_name = 
        D_name = self.referent_name
        
        # choose C first one that is different from A, B and D
        for i in range(0, len(states_msg.name)):
            if states_msg.name[i] != A_name and states_msg.name[i] != B_name and states_msg.name[i] != D_name:
                C_name = states_msg.name[i]
                break

        if C_name == '':
            return 0

        # do ABC

        # do BCD

    def printTriples(self, triples):
        #print('len(triples) = ', len(triples))
        for i in range(0, len(triples)):
            if triples[i] == None:
                continue
            print(str(i) + ' --- ' + triples[i])

    def model_state_callback(self, states_msg):
        self.referents_positions = []
        self.referents_names = []
        self.marker_array_semantic_labels.markers = []
        self.marker_array_orientations.markers = []
        self.objects_in_lc_names = []
        self.objects_in_lc_positions = []

        #'''
        # go through objects
        for i in range(0, len(states_msg.name)):
            
            # get orientation vectors of the objects/models
            '''
            [yaw, pitch, roll] = self.quaternion_to_euler(states_msg.pose[i].orientation.x, states_msg.pose[i].orientation.y, states_msg.pose[i].orientation.z, states_msg.pose[i].orientation.w)
            #print('\n[yaw, pitch, roll] = ', [yaw, pitch, roll])
            if yaw > self.PI:
                while yaw > self.PI:
                    yaw -= self.PI
            elif yaw <= -self.PI:
                while yaw <= -self.PI:
                    yaw += self.PI
            #print('[yaw, pitch, roll] = ', [yaw, pitch, roll])
            #print('YAW in DEG = ', yaw * 180.0 / self.PI)
            '''
            
            #'''
            # Visualize orientations of the objects as marker arrays
            self.marker_orientation = Marker()
            self.marker_orientation.header.frame_id = 'map'
            self.marker_orientation.id = i
            self.marker_orientation.type = self.marker_orientation.ARROW
            self.marker_orientation.action = self.marker_orientation.ADD
            self.marker_orientation.pose.position.x = states_msg.pose[i].position.x
            self.marker_orientation.pose.position.y = states_msg.pose[i].position.y
            self.marker_orientation.pose.position.z = -1.0
            self.marker_orientation.pose.orientation.x = states_msg.pose[i].orientation.x
            self.marker_orientation.pose.orientation.y = states_msg.pose[i].orientation.y
            self.marker_orientation.pose.orientation.z = states_msg.pose[i].orientation.z
            self.marker_orientation.pose.orientation.w = states_msg.pose[i].orientation.w
            self.marker_orientation.color.r = 0.0
            self.marker_orientation.color.g = 1.0
            self.marker_orientation.color.b = 0.0
            self.marker_orientation.color.a = 1.0
            self.marker_orientation.scale.x = 0.8
            self.marker_orientation.scale.y = 0.3
            self.marker_orientation.scale.z = 0.1
            #marker.frame_locked = False
            self.marker_orientation.ns = "my_namespace"
            if self.marker_orientation not in self.marker_array_orientations.markers:
                self.marker_array_orientations.markers.append(self.marker_orientation)
            #'''
            
            #'''
            # do not visualize ground_plane
            if states_msg.name[i] == 'ground_plane':
                continue
            # visualize origin and update ORIGIN_pos
            elif states_msg.name[i] == self.ORIGIN_name:
                self.ORIGIN_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                self.marker = Marker()
                self.marker.header.frame_id = 'map'
                self.marker.id = 0
                self.marker.type = self.marker.TEXT_VIEW_FACING
                self.marker.action = self.marker.ADD
                self.marker.pose = Pose()
                self.marker.pose.position.x = states_msg.pose[i].position.x
                self.marker.pose.position.y = states_msg.pose[i].position.y
                self.marker.pose.position.z = 2.0
                self.marker.color.r = 1.0
                self.marker.color.g = 0.0
                self.marker.color.b = 0.0
                self.marker.color.a = 1.0
                self.marker.scale.x = 0.5
                self.marker.scale.y = 0.5
                self.marker.scale.z = 0.5
                #marker.frame_locked = False
                self.marker.text = self.ORIGIN_name + ',ORIGIN'
                self.marker.ns = "my_namespace"
                if self.marker not in self.marker_array_semantic_labels.markers:
                    self.marker_array_semantic_labels.markers.append(self.marker)
            # visualize relatum and update RELATUM_pos
            elif states_msg.name[i] == self.RELATUM_name:
                self.RELATUM_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                self.marker = Marker()
                self.marker.header.frame_id = 'map'
                self.marker.id = 1
                self.marker.type = self.marker.TEXT_VIEW_FACING
                self.marker.action = self.marker.ADD
                self.marker.pose = Pose()
                self.marker.pose.position.x = states_msg.pose[i].position.x
                self.marker.pose.position.y = states_msg.pose[i].position.y
                self.marker.pose.position.z = 2.0
                self.marker.color.r = 1.0
                self.marker.color.g = 0.0
                self.marker.color.b = 0.0
                self.marker.color.a = 1.0
                self.marker.scale.x = 0.5
                self.marker.scale.y = 0.5
                self.marker.scale.z = 0.5
                #marker.frame_locked = False
                self.marker.text = self.RELATUM_name + ',RELATUM'
                self.marker.ns = "my_namespace"
                if self.marker not in self.marker_array_semantic_labels.markers:
                    self.marker_array_semantic_labels.markers.append(self.marker)
            else:
                '''        
                # update current origin_pos and add referent data to lists
                if states_msg.name[i] == self.origin_name:
                    self.origin_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                    self.referents_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                    self.referents_names.append(states_msg.name[i])
                # update current relatum_pos and add referent data to lists    
                elif states_msg.name[i] == self.relatum_name:
                    self.relatum_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                    self.referents_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                    self.referents_names.append(states_msg.name[i])
                # update current referent_pos and add referent data to lists    
                elif states_msg.name[i] == self.referent_name:
                    self.referent_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                    self.referents_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                    self.referents_names.append(states_msg.name[i])
                # add referent data to lists
                else:
                    self.referents_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                    self.referents_names.append(states_msg.name[i])
                '''
                
                # add referent data to lists
                self.referents_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                self.referents_names.append(states_msg.name[i])

                # visualize referents
                self.marker = Marker()
                self.marker.header.frame_id = 'map'
                self.marker.id = len(self.referents_names) + 1
                self.marker.type = self.marker.TEXT_VIEW_FACING
                self.marker.action = self.marker.ADD
                self.marker.pose = Pose()
                self.marker.pose.position.x = states_msg.pose[i].position.x
                self.marker.pose.position.y = states_msg.pose[i].position.y
                self.marker.pose.position.z = 2.0
                self.marker.color.r = 1.0
                self.marker.color.g = 0.0
                self.marker.color.b = 0.0
                self.marker.color.a = 1.0
                self.marker.scale.x = 0.5
                self.marker.scale.y = 0.5
                self.marker.scale.z = 0.5
                #marker.frame_locked = False
                self.marker.text = states_msg.name[i]
                self.marker.ns = "my_namespace"
                if self.marker not in self.marker_array_semantic_labels.markers:
                    self.marker_array_semantic_labels.markers.append(self.marker)

                # test if an object is in a local costmap
                d_x = states_msg.pose[i].position.x - self.RELATUM_pos[0] # should be robot, not relatum
                d_y = states_msg.pose[i].position.y - self.RELATUM_pos[1] # should be robot, not relatum
                r = math.sqrt((d_x)**2+(d_y)**2)
                if r <= 4:
                    self.objects_in_lc_names.append(states_msg.name[i])
                    self.objects_in_lc_positions.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                    
            #'''
        #'''    
        
        print('self.referents_names = ', self.referents_names)
        print('self.referents_positions = ', self.referents_positions)

        # publish orientations
        pub_markers_orientations.publish(self.marker_array_orientations)
        
        # publish semantic labels
        pub_markers_semantic_labels.publish(self.marker_array_semantic_labels)

        # update R and angle_ref
        self.updateRefSys()
        
        # print obstacles in local costmap
        #print('\n\nThere are ' + str(len(self.objects_in_lc_names)) + " obstacles in the local costmap")
        #print('These obstacles are: ')
        #self.printTriples(self.objects_in_lc_names)

        
        #'''
        # Do something every 100 iterations, so things are not printed too fast
        self.I += 1
        if self.I == 100:
            self.I = 0

            # make QSR triples without reasoning included, after we have collected the freshest positions of objects
            #self.marker_array_semantic_labels.markers = []
            triples = []
            for i in range(0, len(self.referents_names)):
                d_x = self.referents_positions[i][0] - self.RELATUM_pos[0]
                d_y = self.referents_positions[i][1] - self.RELATUM_pos[1]
                r = math.sqrt(d_x**2+d_y**2)
                angle = np.arctan2(d_y, d_x)
                angle -= self.angle_ref
                if angle > self.PI:
                    angle -= 2*self.PI
                elif angle <= -self.PI:
                    angle += 2*self.PI
                
                value = self.getValue(r, angle)
                if value != None:
                    triples.append(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + self.referents_names[i])

                '''
                # visualize referents with QSR without reasoning
                self.marker = Marker()
                self.marker.header.frame_id = 'map'
                self.marker.id = i + 2
                self.marker.type = self.marker.TEXT_VIEW_FACING
                self.marker.action = self.marker.ADD
                self.marker.pose = Pose()
                self.marker.pose.position.x = states_msg.pose[i].position.x
                self.marker.pose.position.y = states_msg.pose[i].position.y
                self.marker.pose.position.z = 2.0
                self.marker.color.r = 1.0
                self.marker.color.g = 0.0
                self.marker.color.b = 0.0
                self.marker.color.a = 1.0
                self.marker.scale.x = 0.5
                self.marker.scale.y = 0.5
                self.marker.scale.z = 0.5
                #marker.frame_locked = False
                self.marker.text = self.referents_names[i] + ', ' + value
                self.marker.ns = "my_namespace"
                if self.marker not in self.marker_array_semantic_labels.markers:
                    self.marker_array_semantic_labels.markers.append(self.marker)
                '''

            # publish semantic labels with QSR without reasoning
            # visualize referents with QSR without reasoning
            #pub_markers_semantic_labels.publish(self.marker_array_semantic_labels)
        
            # print QSR triples every I=100 iterations
            #print('\n\n')
            #print(self.printTriples(triples))

            # Textual explanations based on LIME
            N_segments = len(self.lime_exp)
            if N_segments > 0:
                N_objects_in_lc = len(self.objects_in_lc_names)
                print('\n\n')
                #print('self.lime_exp = ', self.lime_exp)
                #print('N_segments = ', N_segments)
                #print('N_objects_in_lc = ', N_objects_in_lc)
                #print('self.objects_in_lc_names = ', self.objects_in_lc_names)

                #distances = [[0.0]*N_objects_in_lc]*N_segments # N_segments x N_objects_in_lc
                distances = []
                objects_min_dist = []
                segments_min_dist = []

                for i in range(0, N_segments):
                    distances.append([])
                    for j in range(0, N_objects_in_lc):
                        d_x = self.objects_in_lc_positions[j][0] - self.lime_exp[i][0]
                        d_y = self.objects_in_lc_positions[j][1] - self.lime_exp[i][1]
                        dist = math.sqrt((d_x)**2+(d_y)**2)
                        distances[i].append(dist)
                    segments_min_dist.append(min(distances[i]))
                    #print('distances[' + str(i) + '] = ', distances[i])
                    #print('min(distances[' + str(i) + ']) = ', min(distances[i]))    
                    
                #objects_min_dist = [min(distances[:][j]) for j in range(0, N_objects_in_lc)]    

                #print('\ndistances = ', distances)
                #print('\nobjects_min_dist = ', objects_min_dist)
                #print('\nsegments_min_dist = ', segments_min_dist)

                #sorted_indices_of_obj_min_dist = sorted(range(N_objects_in_lc), key = lambda k: objects_min_dist[k])
                sorted_indices_of_seg_min_dist = sorted(range(N_segments), key = lambda k: segments_min_dist[k])
                #print('sorted_indices_of_seg_min_dist = ', sorted_indices_of_seg_min_dist)

                # match objects and lime coefficients
                objects_coeff_pairs = dict()
                used_names = [] 
                for i in range(0, N_segments):
                    idx = sorted_indices_of_seg_min_dist[i]
                    obj_name_ = self.objects_in_lc_names[distances[idx].index(segments_min_dist[idx])]
                    if obj_name_ in used_names:
                        continue
                    used_names.append(obj_name_)
                    lime_coeff_ = self.lime_exp[idx][2]
                    #print(obj_name_ + ' has weight ' + str(lime_coeff_))
                    objects_coeff_pairs[obj_name_] = lime_coeff_

                    # find triple
                    name_idx = self.referents_names.index(obj_name_)
                    d_x = self.referents_positions[name_idx][0] - self.RELATUM_pos[0]
                    d_y = self.referents_positions[name_idx][1] - self.RELATUM_pos[1]
                    r = math.sqrt(d_x**2+d_y**2)
                    angle_ = np.arctan2(d_y, d_x)
                    angle = angle_ - self.angle_ref
                    if angle > self.PI:
                        angle -= 2*self.PI
                    elif angle <= -self.PI:
                        angle += 2*self.PI
                    value = self.getValue(r, angle)    
                    print(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + obj_name_ + ' with lime coefficient of ' + str(lime_coeff_))

                    value = self.getRobotValue(angle)
                    print(obj_name_ + ' is ' + value + ' from the ' + self.RELATUM_name + ' with lime coefficient of ' + str(lime_coeff_))
        #'''

        # do QSR reasoning with 4 points
        #reason(states_msg)

    def getValue(self, r, angle):
        value = ''    

        if qsr_choice == 0:
            if -self.PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif qsr_choice == 1:
            if 0 <= angle < self.PI/2:
                value += 'left/front'
            elif self.PI/2 <= angle <= self.PI:
                value += 'left/back'
            elif -self.PI/2 <= angle < 0:
                value += 'right/front'
            elif -self.PI <= angle < -self.PI/2:
                value += 'right/back'

        elif qsr_choice == 2:
            if r <= self.R:
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

        elif qsr_choice == 3:
            if -self.PI/4 <= angle <= self.PI/4:
                value += 'front'
            elif self.PI/4 < angle < 3*self.PI/4:
                value += 'left'
            elif 3*self.PI/4 <= angle or angle <= -3*self.PI/4:
                value += 'back'
            elif -3*self.PI/4 < angle < -self.PI/4:
                value += 'right'  

        elif qsr_choice == 4:
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

    def ORIGIN_callback(self, msg):
        print('\nreceived ORIGIN string:' + msg.data)
        self.ORIGIN_name = copy.deepcopy(msg.data)
    
    def RELATUM_callback(self, msg):
        print('\nreceived RELATUM string: ' + msg.data)
        self.RELATUM_name = copy.deepcopy(msg.data)
    
    def triple_callback(self, msg):
        #print('\nreceived triple string: ' + msg.data)
        strings = msg.data.split(',', -1)
        self.origin_name = strings[0]
        #print('origin_name = ', self.origin_name)
        self.relatum_name = strings[1]
        #print('relatum_name = ', self.relatum_name)
        self.referent_name = strings[2]
        #print('referent_name = ', self.referent_name)

    def lime_callback(self, msg):
        # [x, y, exp]*N_FEATURES + ORIGINAL_DEVIATION
        self.lime_exp = []
        for i in range(1, int((len(msg.data)-1)/3)): # not taking free-space weight, which is always 0
            self.lime_exp.append([msg.data[3*i],msg.data[3*i+1],msg.data[3*i+2]])
        self.original_deviation = msg.data[-1]
        #print("\nLIME message received = ", msg.data)
        #print("\nLIME message received processed = ", self.lime_exp)
        #print('\nLIME original deviation = ', self.original_deviation)


# choose qsr calculus [0,4]
qsr_obj = qsr()

qsr_choice = 2 
qsr_obj.defineQsrCalculus(qsr_choice)

qsr_choice_ = 1 
qsr_obj.defineRobotQsrCalculus(qsr_choice_)

#tfBuffer = tf2_ros.Buffer()

# Initialize the ROS Node named 'qsr_live', allow multiple nodes to be run with this name
rospy.init_node('qsr_live', anonymous=True)

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, qsr_obj.model_state_callback)

pub_markers_semantic_labels = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
pub_markers_orientations = rospy.Publisher('/orientations', MarkerArray, queue_size=10)

sub_ORIGIN = rospy.Subscriber("/ORIGIN", String, qsr_obj.ORIGIN_callback)
#rostopic pub /ORIGIN std_msgs/String 'cabinet'

sub_RELATUM = rospy.Subscriber("/RELATUM", String, qsr_obj.RELATUM_callback)
#rostopic pub /RELATUM std_msgs/String 'tiago'

sub_triple = rospy.Subscriber("/triple", String, qsr_obj.triple_callback)
#rostopic pub /triple std_msgs/String 'cabinet,tiago,wall_1_model'

#[x,y,coeff]
sub_lime = rospy.Subscriber("/lime_exp", Float32MultiArray, qsr_obj.lime_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()
