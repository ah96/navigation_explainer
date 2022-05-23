#!/usr/bin/env python3

import rospy
import numpy as np
import pandas as pd
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import os
import struct

costmap_size = 160

fields = [PointField('x', 0, PointField.FLOAT32, 1),
PointField('y', 4, PointField.FLOAT32, 1),
PointField('z', 8, PointField.FLOAT32, 1),
PointField('rgba', 12, PointField.UINT32, 1)]

header = Header()

# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('explanation_layer_pub', anonymous=True)

pub_exp_pointcloud = rospy.Publisher("/local_explanation_layer", PointCloud2)

dirCurr = os.getcwd()
dirName = 'lime_rt_data'

file_path_1 = dirName + '/costmap_info_tmp.csv'
file_path_B = dirName + '/output_B.csv'
file_path_G = dirName + '/output_G.csv'
file_path_R = dirName + '/output_R.csv'
       

def publish():
    if os.path.getsize(file_path_1) == 0 or os.path.exists(file_path_1) == False:
        return
    costmap_info_tmp = pd.read_csv(dirCurr + '/' + file_path_1)
    if costmap_info_tmp.empty or costmap_info_tmp.shape != (7,1):
        return

    localCostmapOriginX = costmap_info_tmp.iloc[3][0]
    localCostmapOriginY = costmap_info_tmp.iloc[4][0]
    localCostmapResolution = costmap_info_tmp.iloc[0][0]

    if os.path.getsize(file_path_B) == 0 or os.path.exists(file_path_B) == False:
        return
    output_B = pd.read_csv(dirCurr + '/' + file_path_B)
    if output_B.empty or output_B.shape != (costmap_size, costmap_size):
        return

    if os.path.getsize(file_path_G) == 0 or os.path.exists(file_path_G) == False:
        return
    output_G = pd.read_csv(dirCurr + '/' + file_path_G)
    if output_G.empty or output_G.shape != (costmap_size, costmap_size):
        return

    if os.path.getsize(file_path_R) == 0 or os.path.exists(file_path_R) == False:
        return
    output_R = pd.read_csv(dirCurr + '/' + file_path_R)
    if output_R.empty or output_R.shape != (costmap_size, costmap_size):
        return

    # publish explanation layer
    #points_start = time.time()
    z = 0.0
    a = 255                    
    points = []
    for i in range(0, costmap_size):
        print(i)
        for j in range(0, costmap_size):
            if output_R.iloc[j, i] == output_G.iloc[j, i] == output_B.iloc[j, i]:
                continue
            #if output_R.iloc[j, i] <= 50 and output_G.iloc[j, i] <= 50 and output_B.iloc[j, i] <= 50:
            #    continue
            x = localCostmapOriginX + i * localCostmapResolution
            y = localCostmapOriginY + j * localCostmapResolution
            r = int(output_R.iloc[j, i])
            g = int(output_G.iloc[j, i])
            b = int(output_B.iloc[j, i])
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, rgb]
            points.append(pt)
    header.frame_id = 'odom'
    pc2 = point_cloud2.create_cloud(header, fields, points)
    pc2.header.stamp = rospy.Time.now()
    pub_exp_pointcloud.publish(pc2)


#rate = rospy.Rate(5)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning explanation_layer_pub')
    publish()
    #print('after publish')
    #rate.sleep()
    #rospy.spin()