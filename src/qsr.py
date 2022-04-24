#!/usr/bin/env python3

from gazebo_msgs.msg import ModelStates, ModelState
import rospy
from geometry_msgs.msg import Pose, Twist
import numpy as np
import math

PI = math.pi

pose = Pose()
twist = Twist()
models = []
init = True

tpcc_dict = {
    'left': 0,
    'right': 1
}

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
    global init, models, pose, twist, origin, origin_name, relatum, relatum_name, referent, referents, tpcc_dict, tpcc_dict_inv, PI

    if init == True:
        init = False
        models = [ModelState()] * len(states_msg.name)

    referents = []
    referents_names = []

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
        elif models[i].model_name == 'citizen_extras_female_02':
            origin = [models[i].pose.position.x, models[i].pose.position.y]
        else:
            referents_names.append(models[i].model_name)
            referents.append([models[i].pose.position.x, models[i].pose.position.y])

    # reference direction == direction between relatum and origin orientations
    d_x = relatum[0] - origin[0]
    if d_x == 0:
        d_x = 1
    d_y = relatum[1] - origin[1]
    #k = d_y / d_x
    angle_ref = np.arctan2(d_x, d_y)

    print('\n\n')
    value = ''

    for i in range(0, len(referents)):
        d_x = referents[i][0] - relatum[0]
        d_y = -1 * (referents[i][1] - relatum[1])
        angle = np.arctan2(d_y, np.sign(d_x) * max(1, abs(d_x)))
        angle -= angle_ref
        if angle > PI:
            angle -= 2*PI
        elif angle < -PI:
            angle += 2*PI

        if -PI <= angle < 0:
            value = 'right'
        else:
            value = 'left'

        print(origin_name + ',' + relatum_name + ' ' + value + ' ' + referents_names[i])    



# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('get_model_state', anonymous=True)

# Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, model_state_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    rospy.spin()