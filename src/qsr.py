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


I = 0

# global variables
global tpcc_dict, tpcc_dict_inv, angle_ref
R = -1
ORIGIN_pos = [0.0,0.0]
ORIGIN_name = 'citizen_extras_female_02' #'citizen_extras_female_02','citizen_extras_female_03','citizen_extras_male_03'
RELATUM_pos = [0.0,0.0]
RELATUM_name = 'tiago'
origin_pos = [0.0,0.0]
origin_name = '' #'citizen_extras_female_02','citizen_extras_female_03','citizen_extras_male_03'
relatum_pos = [0.0,0.0]
relatum_name = ''
referent_pos = [0.0,0.0]
referent_name = ''
referents_poss = []
referents_names = []
PI = math.pi
pose = Pose()
twist = Twist()
models = []
init = True
triples = []
marker_array_msg = MarkerArray()
marker_array_orients_msg = MarkerArray()
lime_exp = []

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

def updateRefSys():
    global ORIGIN_pos, ORIGIN_name, RELATUM_pos, RELATUM_name, R, angle_ref
    # reference direction == direction between RELATUM and ORIGIN
    d_x = RELATUM_pos[0] - ORIGIN_pos[0]
    d_y = RELATUM_pos[1] - ORIGIN_pos[1]
    
    R = math.sqrt((d_x)**2+(d_y)**2)
    angle_ref = np.arctan2(d_y, d_x)
    #print('angle_ref = ', angle_ref)
    #print('angle_ref (in deg) = ', angle_ref * 180 / PI)
    
def reason(states_msg):
    global ORIGIN_name, RELATUM_name, origin_name, relatum_name, referent_name
    # ABC and ABD -> BCD, we must choose C
    A_name = ORIGIN_name
    B_name = RELATUM_name
    #C_name = 
    D_name = referent_name
    
    # choose C first one that is different from A, B and D
    for i in range(0, len(states_msg.name)):
        if states_msg.name[i] != A_name and states_msg.name[i] != B_name and states_msg.name[i] != D_name:
            C_name = states_msg.name[i]
            break

    if C_name == '':
        return 0

    # do ABC

    # do BCD

def printTriples(triples):
    for t in triples:
        print(t)

def model_state_callback(states_msg):
    global triples, init, pose, twist, ORIGIN_pos, ORIGIN_name, RELATUM_pos, RELATUM_name, R, angle_ref, qsr_choice, marker_array_msg, marker_array_orients_msg, pub_markers_orients 
    global origin_name, origin_pos, relatum_name, relatum_pos, referent_name, referent_pos,referents_poss, referents_names, tpcc_dict, tpcc_dict_inv, PI, pub_markers, I

    referents_poss = []
    referents_names = []
    marker_array_msg.markers = []
    marker_array_orients_msg.markers = []
    

    for i in range(0, len(states_msg.name)):
        # get orientations of the objects/models
        #print('\nstates_msg.name[i] = ', states_msg.name[i])
        #print('states_msg.pose[i] = ', states_msg.pose[i])
        [yaw, pitch, roll] = quaternion_to_euler(states_msg.pose[i].orientation.x, states_msg.pose[i].orientation.y, states_msg.pose[i].orientation.z, states_msg.pose[i].orientation.w)
        #print('[yaw, pitch, roll] = ', [yaw, pitch, roll])
        if yaw > 2*PI:
            while yaw > 2*PI:
                yaw -= 2*PI
        elif yaw < -2*PI:
            while yaw < -2*PI:
                yaw += 2*PI
        #print('[yaw, pitch, roll] = ', [yaw, pitch, roll])
        #print('YAW in DEG = ', yaw * 180.0 / PI)
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
        if marker not in marker_array_orients_msg.markers:
            marker_array_orients_msg.markers.append(marker)

        if states_msg.name[i] == 'ground_plane':
            continue
        elif states_msg.name[i] == ORIGIN_name:
            ORIGIN_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
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
            marker.text = ORIGIN_name + ',ORIGIN'
            marker.ns = "my_namespace"
            if marker not in marker_array_msg.markers:
                marker_array_msg.markers.append(marker)
        elif states_msg.name[i] == RELATUM_name:
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
            marker.text = RELATUM_name + ',RELATUM'
            marker.ns = "my_namespace"
            if marker not in marker_array_msg.markers:
                marker_array_msg.markers.append(marker)
        elif states_msg.name[i] == origin_name:
            origin_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
            referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
            referents_names.append(states_msg.name[i])
        elif states_msg.name[i] == relatum_name:
            relatum_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
            referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
            referents_names.append(states_msg.name[i])
        elif states_msg.name[i] == referent_name:
            referent_pos = [states_msg.pose[i].position.x, states_msg.pose[i].position.y]
            referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
            referents_names.append(states_msg.name[i])
        else:
            referents_poss.append([states_msg.pose[i].position.x, states_msg.pose[i].position.y])
            referents_names.append(states_msg.name[i])
   
    updateRefSys()

    # make triples
    triples = []
    for i in range(0, len(referents_names)):
        d_x = referents_poss[i][0] - RELATUM_pos[0]
        d_y = referents_poss[i][1] - RELATUM_pos[1]
        r = math.sqrt(d_x**2+d_y**2)
        angle = np.arctan2(d_y, d_x)
        angle -= angle_ref
        if angle > PI:
            angle -= 2*PI
        elif angle < -PI:
            angle += 2*PI
        
        value = ''    

        if qsr_choice == 0:
            if -PI <= angle < 0:
                value += 'right'
            else:
                value += 'left'

        elif qsr_choice == 1:
            if 0 <= angle < PI/2:
                value += 'left/front'
            elif PI/2 <= angle <= PI:
                value += 'left/back'
            elif -PI/2 <= angle < 0:
                value += 'right/front'
            elif -PI <= angle < -PI/2:
                value += 'right/back'

        elif qsr_choice == 2:
            if r <= R:
                value += 'c'
            else: 
                value += 'd'    

            if angle == 0:
                value += 'sb'
            elif 0 < angle <= PI/4:
                value += 'lb'
            elif PI/4 < angle < PI/2:
                value += 'bl'
            elif angle == PI/2:
                value += 'sl'
            elif PI/2 < angle < 3*PI/4:
                value += 'fl'
            elif 3*PI/4 <= angle < PI:
                value += 'lf'
            elif angle == PI or angle == -PI:
                value += 'sf'
            elif -PI < angle <= -3*PI/4:
                value += 'rf'
            elif -3*PI/4 < angle < -PI/2:
                value += 'fr'
            elif angle == -PI/2:
                value += 'sr'
            elif -PI/2 < angle < -PI/4:
                value += 'br'        
            elif -PI/4 <= angle < 0:
                value += 'rb'

        elif qsr_choice == 3:
            if -PI/4 <= angle <= PI/4:
                value += 'front'
            elif PI/4 < angle < 3*PI/4:
                value += 'left'
            elif 3*PI/4 <= angle or angle <= -3*PI/4:
                value += 'back'
            elif -3*PI/4 < angle < -PI/4:
                value += 'right'  

        elif qsr_choice == 4:
            if angle == 0:
                value += 'straight-front'
            elif 0 < angle < PI/2:
                value += 'left-front'
            elif angle == PI/2:
                value += 'exactly-left'
            elif PI/2 < angle < PI:
                value += 'left-back'
            elif angle == PI or angle == -PI:
                value += 'straight-back'
            elif -PI < angle < -PI/2:
                value += 'right-back'
            elif angle == -PI/2:
                value += 'exactly-right'
            elif -PI/2 < angle < 0:
                value += 'right-front'

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.id = 2+i
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose = Pose()
        marker.pose.position.x = referents_poss[i][0]
        marker.pose.position.y = referents_poss[i][1]
        marker.pose.position.z = 2.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        #marker.frame_locked = False
        marker.text = referents_names[i] + ',' + value
        marker.ns = "my_namespace"
        #if marker not in marker_array_msg.markers:
        marker_array_msg.markers.append(marker)        

        triples.append(ORIGIN_name + ',' + RELATUM_name + ' ' + value + ' ' + referents_names[i])

    pub_markers.publish(marker_array_msg)
    pub_markers_orients.publish(marker_array_orients_msg)

    I += 1
    # print triples
    if I == 100:
        #print('\n\n')
        #printTriples(triples)
        I = 0

    #reason(states_msg)

def defineQsrCalculus(qsr_choice):
    global tpcc_dict, tpcc_dict_inv, R

    if qsr_choice == 0:
        # left -- right dichotomy in a relative refence system
        # from 'moratz2008qualitative'
        # used for getting semantic costmap
        tpcc_dict = {
            'left': 0,
            'right': 1
        }

    elif qsr_choice == 1:
        # single cross calculus from 'moratz2008qualitative'
        # used for getting semantic costmap
        tpcc_dict = {
            'left/front': 0,
            'right/front': 1,
            'left/back': 2,
            'right/back': 3
        }

    elif qsr_choice == 2:
        # TPCC reference system
        # my modified version from 'moratz2008qualitative'
        # used for getting semantic costmap
        if R == 0:
            tpcc_dict = {
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
            tpcc_dict = {
                'csb': 0,
                'clb': 1,
                'cbl': 2,
                'csl': 3,
                'cfl': 4,
                'clf': 5,
                'csf': 6,
                'crf': 7,
                'cfr': 8,
                'csr': 9,
                'cbr': 10,
                'crb': 11,
                'dsb': 12,
                'dlb': 13,
                'dbl': 14,
                'dsl': 15,
                'dfl': 16,
                'dlf': 17,
                'dsf': 18,
                'drf': 19,
                'dfr': 20,
                'dsr': 21,
                'dbr': 22,
                'drb': 23
            }

    elif qsr_choice == 3:
        # A model [(Herrmann, 1990),(Hernandez, 1994)] from 'moratz2002spatial'
        # used for getting semantic costmap
        tpcc_dict = {
            'front': 0,
            'left': 1,
            'back': 2,
            'right': 3
        }

    elif qsr_choice == 4:
        # Model for combined expressions from 'moratz2002spatial'
        # used for getting semantic costmap
        tpcc_dict = {
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
    tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}

def ORIGIN_callback(msg):
    print('\nreceived ORIGIN string:' + msg.data)
    global ORIGIN_name#, marker_array_msg
    #marker_array_msg.markers = []
    ORIGIN_name = copy.deepcopy(msg.data)
   
def RELATUM_callback(msg):
    print('\nreceived RELATUM string: ' + msg.data)
    global RELATUM_name#, marker_array_msg
    #marker_array_msg.markers = []
    RELATUM_name = copy.deepcopy(msg.data)
   
def triple_callback(msg):
    print('\nreceived triple string: ' + msg.data)
    global origin_name, relatum_name, referent_name
    strings = msg.data.split(',', -1)
    origin_name = strings[0]
    print('origin_name = ', origin_name)
    relatum_name = strings[1]
    print('relatum_name = ', relatum_name)
    referent_name = strings[2]
    print('referent_name = ', referent_name)

def lime_callback(msg):
    global lime_exp
    for i in range(0, int(len(msg.data)/3)):
        lime_exp.append([msg.data[3*i],msg.data[3*i+1],msg.data[3*i+2]])


# choose qsr calculus [0,4]
qsr_choice = 0 
defineQsrCalculus(qsr_choice)

# Initialize the ROS Node named 'qsr', allow multiple nodes to be run with this name
rospy.init_node('qsr', anonymous=True)

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, model_state_callback)

pub_markers = rospy.Publisher('/semantic_labels', MarkerArray, queue_size=10)
pub_markers_orients = rospy.Publisher('/orientations', MarkerArray, queue_size=10)

sub_ORIGIN = rospy.Subscriber("/ORIGIN", String, ORIGIN_callback)
sub_RELATUM = rospy.Subscriber("/RELATUM", String, RELATUM_callback)
#rostopic pub /ORIGIN std_msgs/String 'cabinet'
#rostopic pub /RELATUM std_msgs/String 'tiago'
sub_triple = rospy.Subscriber("/triple", String, triple_callback)
#rostopic pub /triple std_msgs/String 'cabinet,tiago,wall_1_model'

sub_lime = rospy.Subscriber("/lime_exp", Float32MultiArray, lime_callback)


# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()