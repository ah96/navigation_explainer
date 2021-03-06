#!/usr/bin/env python3

# lc -- local costmap

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
import copy
import tf2_ros
import os
from gazebo_msgs.msg import ModelStates
import math
from skimage.measure import regionprops

from typing import NamedTuple
from point2d import Point2D
# obstacle point class
class obstaclePoints(NamedTuple):
    c: Point2D
    tl: Point2D
    tr: Point2D
    bl: Point2D
    br: Point2D


class lime_rt_sub(object):
    # Constructor
    def __init__(self):
        # plans' variables
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.local_plan_tmp = [] 
        self.plan_tmp = [] 
        self.global_plan_tmp = [] 
        self.global_plan_empty = True
        self.local_plan_empty = True
        
        # poses' variables
        self.footprint_tmp = []  
        self.amcl_pose_tmp = [] 
        self.odom_tmp = []
        self.odom_x = 0
        self.odom_y = 0

        # tf variables
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 

        # costmap variables
        self.image_rgb = np.array([]) 
        self.segments = np.array([])
        self.data = np.array([]) 
        self.image = np.array([])
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.costmap_size = 160
        self.costmap_info_tmp = []

        # deviation
        self.original_deviation = 0
        self.local_costmap_empty = True

        # samples        
        self.num_samples = 0
        self.n_features = 0

        # directory
        self.dirCurr = os.getcwd()
        self.dirName = 'lime_rt_data'
        try:
            os.mkdir(self.dirName)
        except FileExistsError:
            pass

        # semantic
        self.semantic_tags = []
        #self.semantic_global_map = []
        self.static_names = []
        self.gazebo_tags = []

        # plotting
        self.plot_data = False
        self.plot_segments = False
        
        # gazebo
        self.gazebo_names = []
        self.gazebo_poses = []

    # Declare subscribers
    def main_(self):
        # subscribers
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)
        
        # semantic part
        semantic_worlds_names = ['world_movable_chair_1', 'world_movable_chair_2', 'world_movable_chair_3', 'world_no_openable_door', 'world_openable_door']
        idx = 2
        
        # load semantic tags
        self.semantic_tags = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '_tags.csv')
        # populate static_names
        for i in range(0, self.semantic_tags.shape[0]):
            self.static_names.append(self.semantic_tags.iloc[i][1])
        #print(self.static_names)    

        # load gazebo tags
        gazebo_tags_pd = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '_gazebo_tags.csv')
        # populate gazebo_tags
        for i in range(0, gazebo_tags_pd.shape[0]):
            self.gazebo_tags.append(gazebo_tags_pd.iloc[i][0])
        #print(self.gazebo_tags)

        # load semantic_global_map
        #self.semantic_global_map = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '.csv', index_col=None, header=None))
        
        # load semantic_global_info
        self.semantic_global_map_info = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/worlds/' + semantic_worlds_names[idx] + '/' + semantic_worlds_names[idx] + '_info.csv', index_col=None, header=None)
        self.semantic_global_map_info = self.semantic_global_map_info.transpose()
       
        # populate some semantic_global_map variables
        self.semantic_global_map_resolution = float(self.semantic_global_map_info.iloc[1][1])
        print(self.semantic_global_map_resolution)
        self.semantic_global_map_origin_x = float(self.semantic_global_map_info.iloc[4][1])
        print(self.semantic_global_map_origin_x)
        self.semantic_global_map_origin_y = float(self.semantic_global_map_info.iloc[5][1])
        print(self.semantic_global_map_origin_y)

    # Create data based on segments
    def create_data(self):
        # create data -- N+1 perturbations
        self.n_features = np.unique(self.segments).shape[0] # (N+1) - probably includes N normal segments + 1 free space segment
        self.num_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.num_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.num_samples, self.n_features))

    # Segmentation algorithm
    # Append to the closest centroid
    def segment_local_costmap_semantic_1(self):
        start = time.time()
        
        # tf from map to odom
        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())

        # static(=known+unknown) objects
        centroids_static_map = []
        centroids_static_pixel = []
        values_static = []
        labels_static = []

        # lc objects
        centroids_in_map = []
        centroids_in_lc = []
        values_in_lc = []
        labels_in_lc = []

        # known obstacles
        for i in range(0, self.semantic_tags.shape[0]):   
            # centroids of a known object in a /map frame
            cx = float(self.semantic_tags.iloc[i][3])
            cy = float(self.semantic_tags.iloc[i][2])
            px = cx*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
            py = cy*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
            # transform centroids from /map to /odom frame
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            # centroids of a known object in /odom frame
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)

            # add it also to the list of static objects, as it is not a moving object -- human
            centroids_static_map.append([cx, cy])
            centroids_static_pixel.append([cx_odom, cy_odom])
            values_static.append(self.semantic_tags.iloc[i][0])
            labels_static.append(self.semantic_tags.iloc[i][1])

            # test if the centroid of this known object is in the lc
            # if it is in lc, then add it to the lists
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                label = self.semantic_tags.iloc[i][1]
                dx = self.semantic_tags.iloc[i][5]
                dy = self.semantic_tags.iloc[i][4]
                v = self.semantic_tags.iloc[i][0]

                centroids_in_map.append([cx, cy])
                centroids_in_lc.append([cx_odom, cy_odom])
                values_in_lc.append(v)
                labels_in_lc.append(label)

        # unknown obstacles
        unknown_obstacles = []
        # find unknown objects from gazebo
        for i in range(0, len(self.gazebo_names)):
            # human is not an unknown object -- only observer
            # ground_plane is not an unknown object
            if 'citizen' in self.gazebo_names[i] or 'tiago' in self.gazebo_names[i] or self.gazebo_names[i] in self.gazebo_tags:
                continue

            # if an object is not in the list of static objects, nor a robot, nor a human
            #print('found - ' + self.gazebo_names[i])

            # centroids of an unknown object in a /map frame
            px = float(self.gazebo_poses[i].position.x)
            py = float(self.gazebo_poses[i].position.y)
            # transform centroids from /map to /odom frame
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            # centroids of an unknown object in /odom frame
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)

            # add it also to the list of static objects, as it is not a moving object -- human
            label = self.gazebo_names[i]
            labels_static.append(label)
            v = len(self.static_names) + len(unknown_obstacles) + 1
            values_static.append(v)
            centroids_static_pixel.append([cx_odom, cy_odom])
            centroids_static_map.append([px, py])

            # test if this unknown object is in the lc
            # if it is in lc, then add it to the lists
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                centroids_in_map.append([px, py])
                centroids_in_lc.append([cx_odom, cy_odom])
                values_in_lc.append(v)
                labels_in_lc.append(label)

            # add the unknown object to the list of the unknown objects
            unknown_obstacles.append([v, label, px, py, cx_odom, cy_odom])

        # save unknown objects to the
        # [ID, label, x_map, y_map, x_pixel, y_pixel]
        pd.DataFrame(unknown_obstacles).to_csv(self.dirCurr + '/' + self.dirName + '/unknown_objects.csv', index=False)#, header=False)
        
        ###### SEGMENTATION PART ######
        # first segment image is a lc with 99s and 100s
        self.segments = copy.deepcopy(self.image_99s_100s)
        # num of all static centroids--objects
        N_centroids_all = len(labels_static)
        # change 99s and 100s with the numeric label of the closest static(known or unknown) object
        for i in range(0, self.segments.shape[0]):
            for j in range(0, self.segments.shape[1]):
                if self.segments[i, j] == 99 or self.segments[i, j] == 100:
                    # calculate distances of this point to the centroids of static objects
                    # populate the point with the numeric label of the object with the centroid closest to this point
                    distances_to_centroids = []
                    for k in range(0, N_centroids_all):
                        dx = abs(j - centroids_static_pixel[k][0])
                        dy = abs(i - centroids_static_pixel[k][1])
                        distances_to_centroids.append(dx + dy) # L1
                        #distances_to_centroids.append(math.sqrt(dx**2 + dy**2)) # L2
                    idx = distances_to_centroids.index(min(distances_to_centroids))
                    self.segments[i, j] = values_static[idx]

        # find centroids_in_LC of the objects areas when real objects centroids are not in LC
        lc_regions = regionprops(self.segments.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            if v not in values_in_lc:
                idx = values_static.index(v)
                centroids_in_map.append(centroids_static_map[idx])
                centroids_in_lc.append([cx, cy])
                values_in_lc.append(v)
                labels_in_lc.append(labels_static[idx])
        
        objects_in_lc = pd.DataFrame(
        {'values': values_in_lc,
        'labels': labels_in_lc,
        })        
        # save labels of objects in the local costmap
        pd.DataFrame(objects_in_lc).to_csv(self.dirCurr + '/' + self.dirName + '/objects_in_lc.csv', index=False)#, header=False)
        
        end = time.time()
        print('\nsemantic_segmentation_time = ', end-start)

        # num of centroids--objects in lc
        N_centroids_in_lc = len(centroids_in_lc)
        if self.plot_segments == True:
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')

            for i in range(0, N_centroids_in_lc):
                ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
                ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')

            fig.savefig('segments.png', transparent=False)
            fig.clf()
        
        return self.segments    

    # Segmentation algorithm
    # Map the objects to the LC and append remaining points to the closest centroid
    def segment_local_costmap_semantic_2(self):
        start = time.time()

        # tf from map to odom
        t_mo = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r_mo = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_mo = np.asarray(r_mo.as_matrix())

        # tf from odom to map
        t_om = np.asarray([self.tf_odom_map_tmp[0],self.tf_odom_map_tmp[1],self.tf_odom_map_tmp[2]])
        r_om = R.from_quat([self.tf_odom_map_tmp[3],self.tf_odom_map_tmp[4],self.tf_odom_map_tmp[5],self.tf_odom_map_tmp[6]])
        r_om = np.asarray(r_om.as_matrix())

        # convert LC points from /odom to /map
        c_odom_x = self.localCostmapOriginX + 0.5 * self.costmap_size * self.localCostmapResolution
        c_odom_y = self.localCostmapOriginY + 0.5 * self.costmap_size * self.localCostmapResolution
        p_odom = np.array([c_odom_x, c_odom_y, 0.0])
        p_map = p_odom.dot(r_om) + t_om
        c_map_x = p_map[0]
        c_map_y = p_map[1]

        tl_odom_x = self.localCostmapOriginX
        tl_odom_y = self.localCostmapOriginY
        p_odom = np.array([tl_odom_x, tl_odom_y, 0.0])
        p_map = p_odom.dot(r_om) + t_om
        tl_map_x = p_map[0]
        tl_map_y = p_map[1]

        tr_odom_x = self.localCostmapOriginX + self.costmap_size * self.localCostmapResolution
        tr_odom_y = self.localCostmapOriginY
        p_odom = np.array([tr_odom_x, tr_odom_y, 0.0])
        p_map = p_odom.dot(r_om) + t_om
        tr_map_x = p_map[0]
        tr_map_y = p_map[1]

        bl_odom_x = self.localCostmapOriginX
        bl_odom_y = self.localCostmapOriginY + self.costmap_size * self.localCostmapResolution
        p_odom = np.array([bl_odom_x, bl_odom_y, 0.0])
        p_map = p_odom.dot(r_om) + t_om
        bl_map_x = p_map[0]
        bl_map_y = p_map[1]

        br_odom_x = self.localCostmapOriginX + self.costmap_size * self.localCostmapResolution
        br_odom_y = self.localCostmapOriginY + self.costmap_size * self.localCostmapResolution        
        p_odom = np.array([br_odom_x, br_odom_y, 0.0])
        p_map = p_odom.dot(r_om) + t_om
        br_map_x = p_map[0]
        br_map_y = p_map[1]

        LC_points_map = obstaclePoints((c_map_x,c_map_y),(tl_map_x,tl_map_y),(tr_map_x,tr_map_y),(bl_map_x,bl_map_y),(br_map_x,br_map_y))

        # static(=known+unknown) objects
        centroids_static_map = []
        centroids_static_pixel = []
        values_static = []
        labels_static = []

        # lc objects
        centroids_in_map = []
        centroids_in_lc = []
        values_in_lc = []
        labels_in_lc = []

        ###### SEGMENTATION PART ######
        # first segment image is a lc with 99s and 100s
        self.segments = copy.deepcopy(self.image_99s_100s)

        # known obstacles
        # populate LC with known objects that have centroids in LC
        for i in range(0, self.semantic_tags.shape[0]):
            c_map_x = self.semantic_tags.iloc[i, 6]
            c_map_y = self.semantic_tags.iloc[i, 7]
            # transform centroids from /map to /odom frame
            p_map = np.array([c_map_x, c_map_y, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo
            # centroids of a known object in /lc frame
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)

            # add it also to the list of static objects, as it is not a moving object -- human
            centroids_static_map.append([cx_odom, cy_odom])
            centroids_static_pixel.append([cx_odom, cy_odom])
            values_static.append(self.semantic_tags.iloc[i][0])
            labels_static.append(self.semantic_tags.iloc[i][1])

            if LC_points_map.tl[0] < c_map_x < LC_points_map.tr[0] and LC_points_map.tl[1] < c_map_y < LC_points_map.bl[1]:
                # centroid of the current object is in the LC
                label = self.semantic_tags.iloc[i][1]
                v = self.semantic_tags.iloc[i][0]
                centroids_in_map.append([c_map_x, c_map_y])
                centroids_in_lc.append([cx_odom, cy_odom])
                values_in_lc.append(v)
                labels_in_lc.append(label)

                # objects vertices from /map to /odom
                tl_map_x = self.semantic_tags.iloc[i, 8]
                tl_map_y = self.semantic_tags.iloc[i, 9]
                p_map = np.array([tl_map_x, tl_map_y, 0.0])
                p_odom = p_map.dot(r_mo) + t_mo
                tl_odom_x = p_odom[0]
                tl_odom_y = p_odom[1]
                tl_odom_x = int((tl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                tl_odom_y = int((tl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

                tr_map_x = self.semantic_tags.iloc[i, 10]
                tr_map_y = self.semantic_tags.iloc[i, 11]
                p_map = np.array([tr_map_x, tr_map_y, 0.0])
                p_odom = p_map.dot(r_mo) + t_mo
                tr_odom_x = p_odom[0]
                tr_odom_y = p_odom[1]
                tr_odom_x = int((tr_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                tr_odom_y = int((tr_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

                bl_map_x = self.semantic_tags.iloc[i, 12]
                bl_map_y = self.semantic_tags.iloc[i, 13]
                p_map = np.array([bl_map_x, bl_map_y, 0.0])
                p_odom = p_map.dot(r_mo) + t_mo
                bl_odom_x = p_odom[0]
                bl_odom_y = p_odom[1]
                bl_odom_x = int((bl_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                bl_odom_y = int((bl_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

                br_map_x = self.semantic_tags.iloc[i, 14]
                br_map_y = self.semantic_tags.iloc[i, 15]
                p_map = np.array([br_map_x, br_map_y, 0.0])
                p_odom = p_map.dot(r_mo) + t_mo
                br_odom_x = p_odom[0]
                br_odom_y = p_odom[1]
                br_odom_x = int((br_odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
                br_odom_y = int((br_odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

                y_2 = max(0,tl_odom_y)
                y_1 = min(self.costmap_size-1,bl_odom_y)
                x_1 = max(0,tl_odom_x)
                x_2 = min(self.costmap_size-1,tr_odom_x)
                #print('(y_1, y_2) = ', (y_1, y_2))
                #print('(x_1, x_2) = ', (x_1, x_2))
                self.segments[y_1:y_2,x_1:x_2] = self.semantic_tags.iloc[i, 0]

        # find unknown objects from gazebo
        unknown_obstacles = []
        for i in range(0, len(self.gazebo_names)):
            # human is not an unknown object -- only observer
            # ground_plane is not an unknown object
            if 'citizen' in self.gazebo_names[i] or 'tiago' in self.gazebo_names[i] or self.gazebo_names[i] in self.gazebo_tags:
                continue

            print('found unknown object - ' + self.gazebo_names[i])

            # centroids of an unknown object in a /map frame
            px = float(self.gazebo_poses[i].position.x)
            py = float(self.gazebo_poses[i].position.y)
            # transform centroids from /map to /odom frame
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_mo) + t_mo
            # centroids of an unknown object in /odom frame
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)

            # add it also to the list of static objects, as it is not a moving object -- human
            label = self.gazebo_names[i]
            labels_static.append(label)
            v = len(self.static_names) + len(unknown_obstacles) + 1
            values_static.append(v)
            centroids_static_pixel.append([cx_odom,cy_odom])
            centroids_static_map.append([px, py])
            
            # test if centroid of this unknown object is in the lc
            # if it is in lc, then add it to the lists
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:                
                centroids_in_map.append([px, py])
                centroids_in_lc.append([cx_odom, cy_odom])
                values_in_lc.append(v)
                labels_in_lc.append(label)
            
            # add the unknown object to the list of the unknown objects
            unknown_obstacles.append([v, label, px, py, cx_odom, cy_odom])

        # save unknown objects 
        # [ID, label, x_map, y_map, x_pixel, y_pixel]
        pd.DataFrame(unknown_obstacles).to_csv(self.dirCurr + '/' + self.dirName + '/unknown_objects.csv', index=False)#, header=False)
        #'''
        # num of all centroids--objects
        N_centroids_static = len(labels_static)
        # append remaining inflated area to the static object with the closest centroid
        for i in range(0, self.costmap_size):
            for j in range(0, self.costmap_size):
                if self.segments[i, j] >= 99:
                    distances_to_centroids = []
                    distances_indices = []
                    for k in range(0, N_centroids_static):
                        dx = abs(j - centroids_static_pixel[k][0])
                        dy = abs(i - centroids_static_pixel[k][1])
                        distances_to_centroids.append(dx + dy) # L1
                        #distances_to_centroids.append(math.sqrt(dx**2 + dy**2)) # L2
                        distances_indices.append(k)
                    idx = distances_to_centroids.index(min(distances_to_centroids))
                    #self.segments[i, j] = values_static[distances_indices[idx]]
                    self.segments[i, j] = values_static[idx]
        #'''
        
        #'''
        # find centroids_in_LC of the objects areas when real objects centroids are not in LC
        lc_regions = regionprops(self.segments.astype(int))
        #print('\nlen(lc_regions) = ', len(lc_regions))
        for lc_region in lc_regions:
            v = lc_region.label
            cy, cx = lc_region.centroid
            if v not in values_in_lc:
                idx = values_static.index(v)
                centroids_in_map.append(centroids_static_map[idx])
                centroids_in_lc.append([cx, cy])
                values_in_lc.append(v)
                labels_in_lc.append(labels_static[idx])
        #'''

        objects_in_lc = pd.DataFrame(
        {'values': values_in_lc,
        'labels': labels_in_lc,
        })        
        # save labels of objects in the local costmap
        pd.DataFrame(objects_in_lc).to_csv(self.dirCurr + '/' + self.dirName + '/objects_in_lc.csv', index=False)#, header=False)

        end = time.time()
        print('\nsemantic_segmentation_time = ', end-start)

        # plot segments
        N_centroids_in_lc = len(labels_in_lc)
        #print('\nN_centroids_in_lc = ', N_centroids_in_lc)
        if self.plot_segments == True:
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')
            #'''
            for i in range(0, N_centroids_in_lc):
                ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
                ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')
            #'''
            fig.savefig('segments.png', transparent=False)
            fig.clf()
        
        return self.segments

    # Define a callback for the local costmap
    def local_costmap_callback(self, msg):
        #print('\nlocal_costmap_callback')

        # catch map_odom and odom_map tf
        try:
            # catch transform from /map to /odom and vice versa
            transf = self.tfBuffer.lookup_transform('map', 'odom', rospy.Time())
            #t = np.asarray([transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z])
            #r = R.from_quat([transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w])
            #r_ = np.asarray(r.as_matrix())

            transf_ = self.tfBuffer.lookup_transform('odom', 'map', rospy.Time())

            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
        # if tf not catched do not go further
        except:
            print('tf except!!!')
            #pass    
            return

        # save tf data
        pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_odom_map_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_map_odom_tmp.csv', index=False)#, header=False)
        
        # save costmap data
        self.localCostmapOriginX = msg.info.origin.position.x
        self.localCostmapOriginY = msg.info.origin.position.y
        self.localCostmapResolution = msg.info.resolution
        self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

        # create np.array image object
        self.image = np.asarray(msg.data)
        self.image.resize((msg.info.height,msg.info.width))

        self.image_99s_100s = copy.deepcopy(self.image)
        self.image_99s_100s[self.image_99s_100s < 99] = 0        
        if self.plot_data == True:
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_99s_100s.astype('float64'), aspect='auto')
            fig.savefig('local_costmap_99s_100s.png', transparent=False)
            fig.clf()
            
            self.image_original = copy.deepcopy(self.image)
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_original.astype('float64'), aspect='auto')
            fig.savefig('local_costmap_original.png', transparent=False)
            fig.clf()
            
            self.image_100s = copy.deepcopy(self.image)
            self.image_100s[self.image_100s != 100] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_100s.astype('float64'), aspect='auto')
            fig.savefig('local_costmap_100s.png', transparent=False)
            fig.clf()
            
            self.image_99s = copy.deepcopy(self.image)
            self.image_99s[self.image_99s != 99] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_99s.astype('float64'), aspect='auto')
            fig.savefig('local_costmap_99s.png', transparent=False)
            fig.clf()
            
            self.image_less_than_99 = copy.deepcopy(self.image)
            self.image_less_than_99[self.image_less_than_99 >= 99] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_less_than_99.astype('float64'), aspect='auto')
            fig.savefig('local_costmap_less_than_99.png', transparent=False)
            fig.clf()
            
        # Turn inflated area to free space and 100s to 99s
        self.image[self.image == 100] = 99
        self.image[self.image <= 98] = 0

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        #self.image = self.image * 1.0

        # my change - to return grayscale to classifier_fn
        self.fudged_image = self.image.copy()
        self.fudged_image[:] = 0 #hide_color = 0

        # save indices of robot's odometry location in local costmap to class variables
        self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

        # find segments
        #self.segments = self.segment_local_costmap_semantic_1()
        self.segments = self.segment_local_costmap_semantic_2()

        self.create_data()

        self.local_costmap_empty = False

        pd.DataFrame(self.costmap_info_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/costmap_info_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.image).to_csv(self.dirCurr + '/' + self.dirName + '/image.csv', index=False) #, header=False)
        pd.DataFrame(self.fudged_image).to_csv(self.dirCurr + '/' + self.dirName + '/fudged_image.csv', index=False)#, header=False)
        pd.DataFrame(self.segments).to_csv(self.dirCurr + '/' + self.dirName + '/segments.csv', index=False)#, header=False)
        pd.DataFrame(self.data).to_csv(self.dirCurr + '/' + self.dirName + '/data.csv', index=False)#, header=False)

    # Define a callback for the local plan
    def odom_callback(self, msg):
        #print('odom_callback!!!')
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_tmp = [self.odom_x, self.odom_y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        
    # Define a callback for the global plan
    def global_plan_callback(self, msg):
        #print('\nglobal_plan_callback!')
        self.global_plan_xs = [] 
        self.global_plan_ys = []
        self.global_plan_tmp = []
        self.plan_tmp = []
        self.transformed_plan_xs = []
        self.transformed_plan_ys = []

        for i in range(0,len(msg.poses)):
            self.global_plan_xs.append(msg.poses[i].pose.position.x) 
            self.global_plan_ys.append(msg.poses[i].pose.position.y)
            self.global_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])
            self.plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

        self.global_plan_empty = False

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/global_plan_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/plan_tmp.csv', index=False)#, header=False)
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')  
        try:
            self.local_plan_x_list = [] 
            self.local_plan_y_list = [] 
            self.local_plan_tmp = []

            for i in range(0,len(msg.poses)):
                self.local_plan_tmp.append([msg.poses[i].pose.position.x,msg.poses[i].pose.position.y,msg.poses[i].pose.orientation.z,msg.poses[i].pose.orientation.w,5])

                x_temp = int((msg.poses[i].pose.position.x - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((msg.poses[i].pose.position.y - self.localCostmapOriginY) / self.localCostmapResolution)
                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    self.local_plan_x_list.append(x_temp)
                    self.local_plan_y_list.append(y_temp)

            self.local_plan_empty = False

            pd.DataFrame(self.local_plan_x_list).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_x_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_y_list).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_y_list.csv', index=False)#, header=False)
            pd.DataFrame(self.local_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/local_plan_tmp.csv', index=False)#, header=False)
            pd.DataFrame(self.odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/odom_tmp.csv', index=False)#, header=False)

        except:
            pass
    
    # Define a callback for the footprint
    def footprint_callback(self, msg):
        self.footprint_tmp = []
        for i in range(0,len(msg.polygon.points)):
            self.footprint_tmp.append([msg.polygon.points[i].x,msg.polygon.points[i].y,msg.polygon.points[i].z,5])
        pd.DataFrame(self.footprint_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/footprint_tmp.csv', index=False)#, header=False)
    
    # Define a callback for the amcl pose
    def amcl_callback(self, msg):
        self.amcl_pose_tmp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pd.DataFrame(self.amcl_pose_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/amcl_pose_tmp.csv', index=False)#, header=False)

    # Gazebo callback
    def model_state_callback(self, states_msg):
        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose


# ----------main-----------
# main function
# define lime_rt_sub object
lime_rt_obj = lime_rt_sub()
# call main to initialize subscribers
lime_rt_obj.main_()

# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('lime_rt_sub_semantic', anonymous=True)

# declare transformation buffer
lime_rt_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_obj.tf_listener = tf2_ros.TransformListener(lime_rt_obj.tfBuffer)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()