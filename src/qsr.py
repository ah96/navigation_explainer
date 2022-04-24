#!/usr/bin/env python3

from gazebo_msgs.msg import ModelStates, ModelState
import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray


PI = math.pi

pose = Pose()
twist = Twist()
models = []
init = True

R = -1

# choose qsr calculus
qsr_choice = 0

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

origin = [0.0,0.0]
origin_name = 'human'
relatum = [0.0,0.0]
relatum_name = 'tiago'
referent = [0.0,0.0]
referents = []
referents_names = []

# Define a callback for the ModelStates message
def model_state_callback(states_msg):
    global init, models, pose, twist, origin, origin_name, relatum, relatum_name, referent, referents, tpcc_dict, tpcc_dict_inv, PI, pub_markers

    if init == True:
        init = False
        models = [ModelState()] * len(states_msg.name)

    referents = []
    referents_names = []

    marker_array_msg = MarkerArray()

    for i in range(0, len(states_msg.name)):
        pose = states_msg.pose[i]
        twist = states_msg.twist[i]
        models[i].model_name = states_msg.name[i]
        models[i].pose = pose
        models[i].twist = twist
        models[i].reference_frame = 'map' # 'map', 'world', etc.
        #print('\nmodel: ', models[i])

        if models[i].model_name == 'tiago':
            relatum = [models[i].pose.position.x, models[i].pose.position.y]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = models[i].pose.position.x
            marker.pose.position.y = models[i].pose.position.y
            marker.pose.position.z = 2.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            #marker.frame_locked = False
            marker.text = models[i].model_name
            marker.ns = "my_namespace"
            marker_array_msg.markers.append(marker)
        elif models[i].model_name == 'citizen_extras_female_02':
            origin = [models[i].pose.position.x, models[i].pose.position.y]
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = models[i].pose.position.x
            marker.pose.position.y = models[i].pose.position.y
            marker.pose.position.z = 2.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            #marker.frame_locked = False
            marker.text = 'human'
            marker.ns = "my_namespace"
            marker_array_msg.markers.append(marker)
        else:
            if models[i].model_name == 'ground_plane':
                continue
            referents_names.append(models[i].model_name)
            referents.append([models[i].pose.position.x, models[i].pose.position.y])
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.pose = Pose()
            marker.pose.position.x = models[i].pose.position.x
            marker.pose.position.y = models[i].pose.position.y
            marker.pose.position.z = 2.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            #marker.frame_locked = False
            marker.text = models[i].model_name
            marker.ns = "my_namespace"
            marker_array_msg.markers.append(marker)

        '''
        marker0 = Marker()
        marker0.header.frame_id = 'map'
        marker0.id = 55
        marker0.type = marker.TEXT_VIEW_FACING
        marker0.action = marker.ADD
        marker0.pose = Pose()
        marker0.pose.position.x = 0.0
        marker0.pose.position.y = 0.0
        marker0.pose.position.z = 2.0
        marker0.color.r = 1.0
        marker0.color.g = 0.0
        marker0.color.b = 0.0
        marker0.color.a = 1.0
        marker0.scale.x = 0.5
        marker0.scale.y = 0.5
        marker0.scale.z = 0.5
        #marker.frame_locked = False
        marker0.text = 'ORIGIN'
        marker0.ns = "my_namespace"
        marker_array_msg.markers.append(marker0)

        marker1 = Marker()
        marker1.header.frame_id = 'map'
        marker1.id = 56
        marker1.type = marker.TEXT_VIEW_FACING
        marker1.action = marker.ADD
        marker1.pose = Pose()
        marker1.pose.position.x = 1.0
        marker1.pose.position.y = 0.0
        marker1.pose.position.z = 2.0
        marker1.color.r = 1.0
        marker1.color.g = 0.0
        marker1.color.b = 0.0
        marker1.color.a = 1.0
        marker1.scale.x = 0.5
        marker1.scale.y = 0.5
        marker1.scale.z = 0.5
        #marker.frame_locked = False
        marker1.text = 'X'
        marker1.ns = "my_namespace"
        marker_array_msg.markers.append(marker1)

        marker2 = Marker()
        marker2.header.frame_id = 'map'
        marker2.id = 57
        marker2.type = marker.TEXT_VIEW_FACING
        marker2.action = marker.ADD
        marker2.pose = Pose()
        marker2.pose.position.x = 0.0
        marker2.pose.position.y = 1.0
        marker2.pose.position.z = 2.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.color.a = 1.0
        marker2.scale.x = 0.5
        marker2.scale.y = 0.5
        marker2.scale.z = 0.5
        #marker.frame_locked = False
        marker2.text = 'Y'
        marker2.ns = "my_namespace"
        marker_array_msg.markers.append(marker2)
        '''

    print('\n\n')

    # reference direction == direction between relatum and origin orientations
    d_x = relatum[0] - origin[0]
    d_y = relatum[1] - origin[1]
    R = math.sqrt((relatum[0] - origin[0])**2+(relatum[1] - origin[1])**2)
    angle_ref = np.arctan2(d_y, d_x)
    #print('angle_ref = ', angle_ref)
    #print('angle_ref (in deg) = ', angle_ref * 180 / PI)

    for i in range(0, len(referents)):
        d_x = referents[i][0] - relatum[0]
        d_y = referents[i][1] - relatum[1]
        r = math.sqrt(d_x**2+d_y**2)
        angle = np.arctan2(d_y, d_x)
        #print('\nreferent_name = ', referents_names[i])
        #print('angle = ', angle)
        #print('angle (in deg) = ', angle * 180 / PI)
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

        print(origin_name + ',' + relatum_name + ' ' + value + ' ' + referents_names[i]) 

    pub_markers.publish(marker_array_msg)      



# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('get_model_state', anonymous=True)

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, model_state_callback)

pub_markers = rospy.Publisher('/qsr_markers', MarkerArray, queue_size=10)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()