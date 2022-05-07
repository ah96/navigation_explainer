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
        self.referents_poss = []
        self.referents_names = []
        self.PI = math.pi
        self.pose = Pose()
        self.twist = Twist()
        self.models = []
        self.init = True
        self.triples = []
        self.marker_array_msg = MarkerArray()
        self.marker_array_orients_msg = MarkerArray()
        self.lime_exp = []
        self.lime_names = []
        self.lime_coeffs = []
        self.I = 0
        self.original_deviation = -1
        self.comb_table = []
        self.perm_table = []

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

                self.comb_table = []
                self.perm_table = []

                self.comb_table = [
                    [[0,1],   [0,1],   [2,3], [2,3], [2,3], [2,3,4,5], [2,3], [2,3,4,5], [2,3,4,5], [4,5,6,7,8,9], [2,4], [6,8,9,10,11], [0], [12,13]],
                    [[1],     [1],     [], [], [], [], [], [], [], [], [], [], [], []],
                    [[2,3],   [2,3],   [], [], [], [], [], [], [], [], [], [], [], []],
                    [[3],     [3],     [], [], [], [], [], [], [], [], [], [], [], []],
                    [[4,5],   [4,5],   [], [], [], [], [], [], [], [], [], [], [], []],
                    [[5],     [5],     [], [], [], [], [], [], [], [], [], [], [], []],
                    [[6,7],   [6,7],   [], [], [], [], [], [], [], [], [], [], [], []],
                    [[7],     [7],     [], [], [], [], [], [], [], [], [], [], [], []],
                    [[8,9],   [8,9],   [], [], [], [], [], [], [], [], [], [], [], []],
                    [[9],     [9],     [], [], [], [], [], [], [], [], [], [], [], []],
                    [[10,11], [10,11], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[11],    [11],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[12,13], [12,13], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[13],    [13],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[14,15], [14,15], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[15],    [15],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[16,17], [16,17], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[17],    [17],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[18,19], [18,19], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[19],    [19],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[20,21], [20,21], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[21],    [21],    [], [], [], [], [], [], [], [], [], [], [], []],
                    [[22,23], [22,23], [], [], [], [], [], [], [], [], [], [], [], []],
                    [[23],    [23],    [], [], [], [], [], [], [], [], [], [], [], []]
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
        for t in triples:
            print(t)

    def model_state_callback(self, states_msg):
        self.referents_poss = []
        self.referents_names = []
        self.marker_array_msg.markers = []
        self.marker_array_orients_msg.markers = []
        
        for i in range(0, len(states_msg.name)):
            # get orientations of the objects/models
            #print('\nstates_msg.name[i] = ', states_msg.name[i])
            #print('states_msg.pose[i] = ', states_msg.pose[i])
            [yaw, pitch, roll] = self.quaternion_to_euler(states_msg.pose[i].orientation.x, states_msg.pose[i].orientation.y, states_msg.pose[i].orientation.z, states_msg.pose[i].orientation.w)
            #print('[yaw, pitch, roll] = ', [yaw, pitch, roll])
            if yaw > 2*self.PI:
                while yaw > 2*self.PI:
                    yaw -= 2*self.PI
            elif yaw < -2*self.PI:
                while yaw < -2*self.PI:
                    yaw += 2*self.PI
            #print('[yaw, pitch, roll] = ', [yaw, pitch, roll])
            #print('YAW in DEG = ', yaw * 180.0 / PI)
            # Orientation
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = states_msg.pose[i].position.x
            marker.pose.position.y = states_msg.pose[i].position.y
            marker.pose.position.z = 2.0
            marker.pose.orientation.x = states_msg.pose[i].orientation.x
            marker.pose.orientation.y = states_msg.pose[i].orientation.y
            marker.pose.orientation.z = states_msg.pose[i].orientation.z
            marker.pose.orientation.w = states_msg.pose[i].orientation.w
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.8
            marker.scale.y = 0.3
            marker.scale.z = 0.1
            #marker.frame_locked = False
            marker.ns = "my_namespace"
            if marker not in self.marker_array_orients_msg.markers:
                self.marker_array_orients_msg.markers.append(marker)

            if states_msg.name[i] == 'ground_plane':
                continue
            elif states_msg.name[i] == self.ORIGIN_name:
                self.ORIGIN_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = 0
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                marker.pose = Pose()
                marker.pose.position.x = states_msg.pose[i].position.x
                marker.pose.position.y = states_msg.pose[i].position.y
                marker.pose.position.z = 2.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                #marker.frame_locked = False
                marker.text = self.ORIGIN_name + ',ORIGIN'
                marker.ns = "my_namespace"
                if marker not in self.marker_array_msg.markers:
                    self.marker_array_msg.markers.append(marker)
            elif states_msg.name[i] == self.RELATUM_name:
                RELATUM_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = 1
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                marker.pose = Pose()
                marker.pose.position.x = states_msg.pose[i].position.x
                marker.pose.position.y = states_msg.pose[i].position.y
                marker.pose.position.z = 2.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                #marker.frame_locked = False
                marker.text = self.RELATUM_name + ',RELATUM'
                marker.ns = "my_namespace"
                if marker not in self.marker_array_msg.markers:
                    self.marker_array_msg.markers.append(marker)
            elif states_msg.name[i] == self.origin_name:
                self.origin_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                self.referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                self.referents_names.append(states_msg.name[i])
            elif states_msg.name[i] == self.relatum_name:
                self.relatum_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                self.referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                self.referents_names.append(states_msg.name[i])
            elif states_msg.name[i] == self.referent_name:
                self.referent_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
                self.referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                self.referents_names.append(states_msg.name[i])
            else:
                self.referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
                self.referents_names.append(states_msg.name[i])
    
        self.updateRefSys()

        # make triples
        triples = []
        for i in range(0, len(self.referents_names)):
            d_x = self.referents_poss[i][0] - RELATUM_pos[0]
            d_y = self.referents_poss[i][1] - RELATUM_pos[1]
            r = math.sqrt(d_x**2+d_y**2)
            angle = np.arctan2(d_y, d_x)
            angle -= self.angle_ref
            if angle > self.PI:
                angle -= 2*self.PI
            elif angle < -self.PI:
                angle += 2*self.PI
            
            value = self.getValue(r, angle)

            # code for publish
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = 2+i
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = self.referents_poss[i][0]
            marker.pose.position.y = self.referents_poss[i][1]
            marker.pose.position.z = 2.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            #marker.frame_locked = False
            marker.text = self.referents_names[i] + ',' + value
            marker.ns = "my_namespace"
            #if marker not in marker_array_msg.markers:
            self.marker_array_msg.markers.append(marker)        

            triples.append(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + self.referents_names[i])

        pub_markers.publish(self.marker_array_msg)
        pub_markers_orients.publish(self.marker_array_orients_msg)

        self.I += 1
        # print triples
        if self.I == 100:
            #print('\n\n')
            #printTriples(triples)
            self.I = 0

        #reason(states_msg)

        # append lime coefficients to the objects
        lime_names = []
        lime_coeffs = []
        for exp in self.lime_exp:
            mini = math.sqrt( (exp[1] - self.referents_poss[0][0])**2 + (exp[2] - self.referents_poss[0][1])**2 )
            id = 0
            #print('len(referents_names) = ', len(referents_names))
            #print('exp = ', exp)
            #print('len(referents_poss) = ', len(referents_poss))
            for j in range(1, len(self.referents_names)):
                dist = math.sqrt( (exp[1] - self.referents_poss[j][0])**2 + (exp[2] - self.referents_poss[j][1])**2 )
                if dist < mini:
                    mini = dist
                    id = j
            if self.referents_names[id] not in lime_names:
                lime_names.append(self.referents_names[id])
                lime_coeffs.append([exp[0]])
            else:
                index = lime_names.index(self.referents_names[id])
                lime_coeffs[index].append(exp[0])    
        

        '''
        # printing out contributing obstacles with all their  weights
        for j in range(0, len(lime_names)):
            if len(lime_coeffs[j]) == 1:
                print(lime_names[j] + ' has a weight ' + str(lime_coeffs[j][0]))
            elif len(lime_coeffs[j]) > 1:
                s = ''
                for c in lime_coeffs[j]:
                    s += str(c) + ', '
                print(lime_names[j] + ' has weights ' + s)    
        '''

        '''
        # printing out contributing obstacles with their triples and max absolute weights
        #print('len(lime_names) = ', len(lime_names))
        print('\n\n')
        for i in range(0, len(lime_names)):
            name = lime_names[i]
            idx = self.referents_names.index(name)
        
            if len(lime_coeffs[i]) > 1:
                coeff = max(lime_coeffs[i], key=abs)
            else:
                coeff = lime_coeffs[i][0]

            d_x = self.referents_poss[idx][0] - RELATUM_pos[0]
            d_y = self.referents_poss[idx][1] - RELATUM_pos[1]
            r = math.sqrt(d_x**2+d_y**2)
            angle = np.arctan2(d_y, d_x)
            angle -= self.angle_ref
            if angle > self.PI:
                angle -= 2*self.PI
            elif angle < -self.PI:
                angle += 2*self.PI
            value = self.getValue(r, angle)    
            print(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + name + ' with weight of ' + str(coeff))
        '''

        # printing with the sorted obstacles
        lime_coeffs_max_by_abs = []
        for i in range(0, len(lime_coeffs)):
            lime_coeffs_max_by_abs.append(max(lime_coeffs[i], key=abs))
        lime_coeffs_max_by_abs_sorted = sorted(lime_coeffs_max_by_abs, key=abs, reverse=True)
        for i in range(0, len(lime_coeffs_max_by_abs_sorted)):
            idx_ = lime_coeffs_max_by_abs.index(lime_coeffs_max_by_abs_sorted[i])
            name = lime_names[idx_]
            idx = self.referents_names.index(name)
            d_x = self.referents_poss[idx][0] - RELATUM_pos[0]
            d_y = self.referents_poss[idx][1] - RELATUM_pos[1]
            r = math.sqrt(d_x**2+d_y**2)
            angle = np.arctan2(d_y, d_x)
            angle -= self.angle_ref
            if angle > self.PI:
                angle -= 2*self.PI
            elif angle < -self.PI:
                angle += 2*self.PI
            value = self.getValue(r, angle)

            '''
            coeff = lime_coeffs_max_by_abs[idx_]
            if coeff > 0:
                print('\n')
                print(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + name + ' is the ' + str(i+1) + '. most important reason for the robot\'s deviation with the weight of ' + str(coeff))
                print('Robot must deviate from its initial path because of ' + name)
                if i == 0:
                    print(name + ' is the main reason for the robot’s deviation and it increases it')
                print('without that segment, the local plan would deviate less from the global plan')    
            else:
                print('\n')
                print(self.ORIGIN_name + ',' + self.RELATUM_name + ' ' + value + ' ' + name + ' with the weight of ' + str(coeff))
                print('If ' + name + ' was not there robot would deviate (more) from its initial path')
                print('without that segment, the local plan would deviate more from the global plan')
            '''

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

# Initialize the ROS Node named 'qsr', allow multiple nodes to be run with this name
rospy.init_node('qsr', anonymous=True)

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, qsr_obj.model_state_callback)

pub_markers = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
pub_markers_orients = rospy.Publisher('/orientations', MarkerArray, queue_size=10)

sub_ORIGIN = rospy.Subscriber("/ORIGIN", String, qsr_obj.ORIGIN_callback)
#rostopic pub /ORIGIN std_msgs/String 'cabinet'

sub_RELATUM = rospy.Subscriber("/RELATUM", String, qsr_obj.RELATUM_callback)
#rostopic pub /RELATUM std_msgs/String 'tiago'

sub_triple = rospy.Subscriber("/triple", String, qsr_obj.triple_callback)
#rostopic pub /triple std_msgs/String 'cabinet,tiago,wall_1_model'

sub_lime = rospy.Subscriber("/lime_exp", Float32MultiArray, qsr_obj.lime_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()