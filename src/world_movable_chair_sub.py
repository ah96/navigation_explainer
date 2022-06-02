#!/usr/bin/env python3

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
import rospy
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from skimage.segmentation import slic
from skimage.color import gray2rgb
from scipy.spatial.transform import Rotation as R
import copy
import tf2_ros
import math
from skimage.measure import regionprops
import os
from gazebo_msgs.msg import ModelStates


class lime_rt_sub(object):
    # Constructor
    def __init__(self):
        self.global_plan_xs = []
        self.global_plan_ys = []
        self.transformed_plan_xs = [] 
        self.transformed_plan_ys = []
        self.local_plan_x_list = [] 
        self.local_plan_y_list = []
        self.footprint_tmp = [] 
        self.local_plan_tmp = [] 
        self.plan_tmp = [] 
        self.global_plan_tmp = [] 
        self.costmap_info_tmp = [] 
        self.amcl_pose_tmp = [] 
        self.tf_odom_map_tmp = [] 
        self.tf_map_odom_tmp = [] 
        self.odom_tmp = []
        self.image_rgb = np.array([]) 
        self.segments = np.array([])
        self.data = np.array([]) 
        self.image = np.array([])
        self.odom_x = 0
        self.odom_y = 0
        self.localCostmapOriginX = 0 
        self.localCostmapOriginY = 0 
        self.localCostmapResolution = 0
        self.original_deviation = 0
        self.costmap_size = 160
        self.global_plan_empty = True
        self.local_costmap_empty = True
        self.local_plan_empty = True
        self.num_samples = 0
        self.n_features = 0
        #self.start = time.time()
        #self.end = time.time()

        self.dirCurr = os.getcwd()

        self.dirName = 'lime_rt_data'
        try:
            os.mkdir(self.dirName)
        except FileExistsError:
            pass

        self.semantic_tags = []
        self.semantic_global_map = []

        self.plot_data = False
        
        self.gazebo_names = []
        self.gazebo_poses = []

        self.static_names = []

    # Segmentation algorithm
    def segment_local_costmap(self, image, img_rgb):
        #print('segmentation algorithm')
        # show original image
        #img = copy.deepcopy(image)

        # Find segments_slic
        segments_slic = slic(img_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                                multichannel=True, convert2lab=True,
                                enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                                start_label=1, mask=None)

        '''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_slic.astype('float64'), aspect='auto')
        fig.savefig('segments_slic.png', transparent=False)
        fig.clf()
        '''

        self.segments = np.zeros(image.shape, np.uint8)

        # make one free space segment
        ctr = 0
        self.segments[:, :] = ctr
        ctr = ctr + 1

        # add obstacle segments
        num_of_obstacles = 0        
        for i in np.unique(segments_slic):
            temp = image[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            #print('count: ', count)
            #print('temp: ', temp)
            #print('len(temp): ', temp.shape[0])
            if np.all(image[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                #print('obstacle')
                self.segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1
        #print('num_of_obstacles: ', num_of_obstacles)        

        '''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments_final.png', transparent=False)
        fig.clf()
        '''

        return self.segments

    # Segmentation algorithm
    def segment_local_costmap_semantic(self):
        #print('segmentation algorithm')
       
        #self.segments = np.zeros(self.image.shape, np.uint8)
        self.segments = copy.deepcopy(self.image_99s_100s)

        if self.plot_data == True:
            segs = copy.deepcopy(self.segments)
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segs.astype('float64'), aspect='auto')
            fig.savefig('segments_beginning.png', transparent=False)
            fig.clf()
            pd.DataFrame(segs).to_csv('SEGMENTS_BEGINNING.csv')

        tiago_inscribed_radius = 11

        
        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())
        print('\n')
        for i in range(0, self.semantic_tags.shape[0]):
            cx = float(self.semantic_tags.iloc[i][3])
            cy = float(self.semantic_tags.iloc[i][2])
            px = cx*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
            py = cy*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                print('Found ' + self.semantic_tags.iloc[i][1] + ' in LC')
                dx_map = self.semantic_tags.iloc[i][5]
                dy_map = self.semantic_tags.iloc[i][4]
                v = self.semantic_tags.iloc[i][0]
                print('(cx, cy, dx_map, dy_map, v) = ', (cx, cy, dx_map, dy_map, v))
                for row in range(int(cy-dy_map-1), int(cy+dy_map+1)):
                    for column in range(int(cx-dx_map-1), int(cx+dx_map+1)):
                        px = column*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
                        py = row*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
                        p_map = np.array([px, py, 0.0])
                        p_odom = p_map.dot(r_) + t
                        cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
                        if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                            if self.segments[cx_odom,cx_odom] >= 99:
                                self.segments[cy_odom,cx_odom] = v

        if self.plot_data == True:
            segs = copy.deepcopy(self.segments)
            segs[segs >= 99] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segs.astype('float64'), aspect='auto')
            fig.savefig('segments_mid.png', transparent=False)
            fig.clf()
            pd.DataFrame(segs).to_csv('SEGMENTS_MID.csv')

        self.segments[self.segments > 99] = 99
        # from odom to map
        t = np.asarray([self.tf_odom_map_tmp[0],self.tf_odom_map_tmp[1],self.tf_odom_map_tmp[2]])
        r = R.from_quat([self.tf_odom_map_tmp[3],self.tf_odom_map_tmp[4],self.tf_odom_map_tmp[5],self.tf_odom_map_tmp[6]])
        r_ = np.asarray(r.as_matrix()) 
        for i in range(0, self.segments.shape[0]):
            for j in range(0, self.segments.shape[1]):
                if self.segments[i, j] == 99:
                    px = j*self.localCostmapResolution + self.localCostmapOriginX
                    py = i*self.localCostmapResolution + self.localCostmapOriginY
                    p_odom = np.array([px, py, 0.0])
                    p_map = p_odom.dot(r_) + t
                    cx = int((p_map[0] - self.semantic_global_map_origin_x) / self.semantic_global_map_resolution)
                    cy = int((p_map[1] - self.semantic_global_map_origin_y) / self.semantic_global_map_resolution)
                    if 0 <= cx < self.semantic_global_map.shape[1] and 0 <= cy < self.semantic_global_map.shape[0]:
                        if self.semantic_global_map[cy, cx] > 0:
                            self.segments[i, j] = self.semantic_global_map[cy, cx]

        if self.plot_data == True:
            segs = copy.deepcopy(self.segments)
            segs[segs >= 99] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segs.astype('float64'), aspect='auto')
            fig.savefig('segments_mid_2.png', transparent=False)
            fig.clf()
            pd.DataFrame(segs).to_csv('SEGMENTS_MID_2.csv')
       
        for ctr in range(1, tiago_inscribed_radius):
            for row in range(0, self.segments.shape[0]):
                for column in range(0, self.segments.shape[1]):
                    if 0 < self.segments[row, column] == 99:
                        for k in range(1, 2*tiago_inscribed_radius):
                            if row >= k and row < self.costmap_size - k and column >= k and column < self.costmap_size - k:
                                '''
                                if 99 >  self.segments[row+k,column]  > 0:
                                    self.segments[row, column] =  self.segments[row+k,column]
                                    break
                                '''
                                if  99 > self.segments[row+k,column+k]  > 0:
                                    self.segments[row, column] = self.segments[row+k,column+k]
                                    break
                                elif  99 > self.segments[row,column+k]  > 0:
                                    self.segments[row, column] = self.segments[row,column+k]
                                    break
                                '''    
                                elif  99 > self.segments[row-k,column+k]  > 0:
                                    self.segments[row, column] = self.segments[row-k,column+k]
                                    break
                                elif  99 > self.segments[row-k,column]  > 0:
                                    self.segments[row, column] = self.segments[row-k,column]
                                    break
                                elif 99 > self.segments[row-k,column-k] > 0:
                                    self.segments[row, column] = self.segments[row-k,column-k]
                                    break
                                elif 99 > self.segments[row,column-k]  > 0:
                                    self.segments[row, column] = self.segments[row,column-k]
                                    break
                                elif 99 > self.segments[row+k,column-k]  > 0:
                                    self.segments[row, column] = self.segments[row+k,column-k]
                                    break
                                '''

       
        '''
        for i in range(0, self.segments.shape[0]):
            for j in range(0, self.segments.shape[1]):
                if 0 < self.segments[i, j] < 99 and self.image_original[i, j] == 100:
                    dx = min(tiago_inscribed_radius, j, self.costmap_size - j)
                    dy = min(tiago_inscribed_radius, i, self.costmap_size - i)
                    v = self.segments[i, j]
                    for row in range(i-dy, i+dy):
                        for column in range(j-dx, j+dx):
                            if self.segments[row, column] == 99:
                                self.segments[row, column] = self.segments[i, j]
        '''

        if self.plot_data:
            self.segments[self.segments == 99] = 0
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')
            fig.savefig('segments_final.png', transparent=False)
            fig.clf()
            #pd.DataFrame(self.segments).to_csv('SEGMENTS_FINAL.csv')


        # from map to odom
        '''
        centroids_in_lc = []
        vals = []

        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())
        for i in range(0, self.semantic_tags.shape[0]):
            cx = float(self.semantic_tags.iloc[i][3])
            cy = float(self.semantic_tags.iloc[i][2])
            px = cx*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
            py = cy*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            cx = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= cx < self.costmap_size and 0 <= cy < self.costmap_size:
                print('Found ' + self.semantic_tags.iloc[i][1] + ' in LC')
                dx_map = self.semantic_tags.iloc[i][5]
                dy_map = self.semantic_tags.iloc[i][4]
                v = self.semantic_tags.iloc[i][0]
                centroids_in_lc.append([cx,cy,v])
                vals.append(v)
                for row in range(0, self.segments.shape[0]):
                    for column in range(0, self.segments.shape[1]):
                        if self.segments[row, column] == 99:
                            dx = abs(column - cx)
                            dy = abs(row - cy)
                            if dx <= dx_map and dy <= dy_map:
                                #print('PROMJENA')
                                self.segments[row, column] = v

        print(centroids_in_lc)

        for row in range(0, self.segments.shape[0]):
            for column in range(0, self.segments.shape[1]):
                if self.segments[row, column] == 99:
                    if row != 0 and row != self.costmap_size - 1 and column != 0 and column != self.costmap_size - 1:
                        if self.segments[row-1,column-1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row-1,column-1])]
                        if self.segments[row,column-1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row,column-1])]
                        if self.segments[row+1,column-1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row+1,column-1])]
                        if self.segments[row+1,column] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row+1,column])]
                        if self.segments[row+1,column+1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row+1,column+1])]
                        if self.segments[row,column+1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row,column+1])]    
                        if self.segments[row-1,column+1] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row-1,column+1])]
                        if self.segments[row-1,column] in vals:
                            self.segments[row, column] = vals[vals.index(self.segments[row-1,column])]
                        else:
                            distances = []
                            for i in range(0, len(centroids_in_lc)):
                                dx = abs(column - centroids_in_lc[i][0])
                                dy = abs(row - centroids_in_lc[i][1])
                                distances.append(dx + dy)
                            if len(distances) > 0:    
                                idx = distances.index(min(distances))
                                self.segments[row, column] = centroids_in_lc[idx][2]    
                    else:
                        distances = []
                        for i in range(0, len(centroids_in_lc)):
                            dx = abs(column - centroids_in_lc[i][0])
                            dy = abs(row - centroids_in_lc[i][1])
                            distances.append(dx + dy)
                        if len(distances) > 0:    
                            idx = distances.index(min(distances))
                            self.segments[row, column] = centroids_in_lc[idx][2]    

        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments.astype('float64'), aspect='auto')
        fig.savefig('segments_final.png', transparent=False)
        fig.clf()
        
        pd.DataFrame(self.segments).to_csv('SEGMENTS.csv')
        '''

        print('self.segments.unique = ', np.unique(self.segments))

        return self.segments

    # Segmentation algorithm
    def segment_local_costmap_semantic_new(self):
        # transformacija iz map u odom
        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())

        centroids_in_lc = []
        edges_in_lc = []
        values_in_lc = []
        labels_in_lc = []
        
        # preslikavanje centroida iz map u odom
        for i in range(0, self.semantic_tags.shape[0]):
            # centroidi u map
            cx = float(self.semantic_tags.iloc[i][3])
            cy = float(self.semantic_tags.iloc[i][2])
            px = cx*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
            py = cy*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
            # preslikavanje iz map u odom
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            # centroidi u odom
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            # da li je centroid u lc?
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                label = self.semantic_tags.iloc[i][1]
                dx = self.semantic_tags.iloc[i][5]
                dy = self.semantic_tags.iloc[i][4]
                v = self.semantic_tags.iloc[i][0]

                centroids_in_lc.append([cx_odom, cy_odom])
                edges_in_lc.append([[max(cx_odom-dx,0),max(cy_odom-dy,0)],[min(cx_odom+dx,159),max(cy_odom-dy,0)],
                [max(cx_odom-dx,0),min(cy_odom+dy,159)],[min(cx_odom+dx,159),min(cy_odom+dy,159)]])
                values_in_lc.append(v)
                labels_in_lc.append(label)

        N_centroids = len(centroids_in_lc)

        if N_centroids > 1:
            print('\ncentroids in lc')
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.image_99s_100s.astype('float64'), aspect='auto')

            for i in range(0, N_centroids):
                ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
                ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')
                for j in range(0, 4):
                    ax.scatter(edges_in_lc[i][j][0], edges_in_lc[i][j][1], c='black', marker='o')

            fig.savefig('costmap_new.png', transparent=False)
            fig.clf()

            start = time.time()
            self.segments = copy.deepcopy(self.image_99s_100s)
            for i in range(0, self.segments.shape[0]):
                for j in range(0, self.segments.shape[1]):
                    if self.segments[i, j] == 99 or self.segments[i, j] == 100:
                        distances_to_centroids = []
                        for k in range(0, N_centroids):
                            dx = abs(j - centroids_in_lc[k][0])
                            dy = abs(i - centroids_in_lc[k][1])
                            distances_to_centroids.append(dx + dy)
                        idx = distances_to_centroids.index(min(distances_to_centroids))
                        self.segments[i, j] = values_in_lc[idx]
            #'''
            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')

            for i in range(0, N_centroids):
                ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
                ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')

            fig.savefig('segments.png', transparent=False)
            fig.clf()
            #'''
            end = time.time()
            print('\ntime = ', end-start)

            i = 0
            for s in np.unique(self.segments):
                self.segments[self.segments == s] = i
                i += 1

        # if there are no centroids in the lc
        else:
            print('\nno centroids in lc')
            # Find segments_slic
            segments_slic = slic(self.image_rgb, n_segments=8, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                                    multichannel=True, convert2lab=True,
                                    enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                                    start_label=1, mask=None)



            self.segments = np.zeros(self.image.shape, np.uint8)

            # make one free space segment
            ctr = 0
            self.segments[:, :] = ctr
            ctr = ctr + 1

            # add obstacle segments
            num_of_obstacles = 0        
            for i in np.unique(segments_slic):
                temp = self.image[segments_slic == i]
                count_of_99_s = np.count_nonzero(temp == 99)
                #print('count: ', count)
                #print('temp: ', temp)
                #print('len(temp): ', temp.shape[0])
                if np.all(self.image[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                    #print('obstacle')
                    self.segments[segments_slic == i] = ctr
                    ctr = ctr + 1
                    num_of_obstacles += 1
            #print('num_of_obstacles: ', num_of_obstacles)

            fig = plt.figure(frameon=False)
            w = 1.6 * 3
            h = 1.6 * 3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments.astype('float64'), aspect='auto')

            for i in range(0, N_centroids):
                ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
                ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')
                for j in range(0, 4):
                    ax.scatter(edges_in_lc[i][j][0], edges_in_lc[i][j][1], c='black', marker='o')

            fig.savefig('segments.png', transparent=False)
            fig.clf()


        '''
        start = time.time()
        self.segments_ = copy.deepcopy(self.image_99s_100s)
        for i in range(0, self.segments_.shape[0]):
            for j in range(0, self.segments_.shape[1]):
                if self.segments_[i, j] == 99 or self.segments_[i, j] == 100:
                    distances_to_centroids = []
                    for k in range(0, N_centroids):
                        dx = abs(j - centroids_in_lc[k][0])
                        dy = abs(i - centroids_in_lc[k][1])
                        distances_to_centroids.append(dx + dy)
                        for q in range(0, 4):
                            dx = abs(j - edges_in_lc[k][q][0])
                            dy = abs(i - edges_in_lc[k][q][1])
                            distances_to_centroids.append(dx + dy)
                    idx = int(distances_to_centroids.index(min(distances_to_centroids)) / 5)
                    self.segments_[i, j] = values_in_lc[idx]

        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments_.astype('float64'), aspect='auto')

        for i in range(0, N_centroids):
            ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
            ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')
            for j in range(0, 4):
                ax.scatter(edges_in_lc[i][j][0], edges_in_lc[i][j][1], c='black', marker='o')

        fig.savefig('segments_new_edges.png', transparent=False)
        fig.clf()
        end = time.time()
        print('time = ', end-start)
        '''

        return self.segments

    # Segmentation algorithm
    def segment_local_costmap_semantic_new_2(self):
        # transformacija iz map u odom
        t = np.asarray([self.tf_map_odom_tmp[0],self.tf_map_odom_tmp[1],self.tf_map_odom_tmp[2]])
        r = R.from_quat([self.tf_map_odom_tmp[3],self.tf_map_odom_tmp[4],self.tf_map_odom_tmp[5],self.tf_map_odom_tmp[6]])
        r_ = np.asarray(r.as_matrix())

        centroids_in_lc = []
        #edges_in_lc = []
        values_in_lc = []
        labels_in_lc = []

        centroids_static = []
        values_static = []
        labels_static = []

        for i in range(0, len(self.gazebo_names)):
            if self.gazebo_names[i] not in self.static_names and ('citizen' in self.gazebo_names[i] or 'chair' in self.gazebo_names[i]):
                #print('found - ' + self.gazebo_names[i])

                # centroidi u map
                px = float(self.gazebo_poses[i].position.x)
                py = float(self.gazebo_poses[i].position.y)
                # preslikavanje iz map u odom
                p_map = np.array([px, py, 0.0])
                p_odom = p_map.dot(r_) + t
                # centroidi u odom
                cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
                cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
                # da li je centroid u lc?
                #'''
                #print('[cx_odom,cy_odom] = ', [cx_odom,cy_odom])
                if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                    label = self.gazebo_names[i]
                    v = len(self.static_names) + 1

                    centroids_in_lc.append([cx_odom, cy_odom])
                    values_in_lc.append(v)
                    labels_in_lc.append(label)
                #'''

                labels_static.append(self.gazebo_names[i])
                values_static.append(len(self.static_names) + 1)
                centroids_static.append([cx_odom,cy_odom])

        # preslikavanje centroida iz map u odom
        for i in range(0, self.semantic_tags.shape[0]):
            # centroidi u map
            cx = float(self.semantic_tags.iloc[i][3])
            cy = float(self.semantic_tags.iloc[i][2])
            px = cx*self.semantic_global_map_resolution + self.semantic_global_map_origin_x
            py = cy*self.semantic_global_map_resolution + self.semantic_global_map_origin_y
            # preslikavanje iz map u odom
            p_map = np.array([px, py, 0.0])
            p_odom = p_map.dot(r_) + t
            # centroidi u odom
            cx_odom = int((p_odom[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            cy_odom = int((p_odom[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            # da li je centroid u lc?
            #'''
            if 0 <= cx_odom < self.costmap_size and 0 <= cy_odom < self.costmap_size:
                label = self.semantic_tags.iloc[i][1]
                dx = self.semantic_tags.iloc[i][5]
                dy = self.semantic_tags.iloc[i][4]
                v = self.semantic_tags.iloc[i][0]

                centroids_in_lc.append([cx_odom, cy_odom])
                #edges_in_lc.append([[max(cx_odom-dx,0),max(cy_odom-dy,0)],[min(cx_odom+dx,159),max(cy_odom-dy,0)],
                #[max(cx_odom-dx,0),min(cy_odom+dy,159)],[min(cx_odom+dx,159),min(cy_odom+dy,159)]])
                values_in_lc.append(v)
                labels_in_lc.append(label)
            #'''

            centroids_static.append([cx_odom, cy_odom])
            values_static.append(self.semantic_tags.iloc[i][0])
            labels_static.append(self.semantic_tags.iloc[i][1])

        N_centroids = len(centroids_in_lc)
        N_centroids_all = len(labels_static)
        print('\nlabels_in_lc:', labels_in_lc)

        start = time.time()
        self.segments = copy.deepcopy(self.image_99s_100s)
        for i in range(0, self.segments.shape[0]):
            for j in range(0, self.segments.shape[1]):
                if self.segments[i, j] == 99 or self.segments[i, j] == 100:
                    distances_to_centroids = []
                    for k in range(0, N_centroids_all):
                        dx = abs(j - centroids_static[k][0])
                        dy = abs(i - centroids_static[k][1])
                        distances_to_centroids.append(dx + dy)
                    idx = distances_to_centroids.index(min(distances_to_centroids))
                    self.segments[i, j] = values_static[idx]
        #'''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments.astype('float64'), aspect='auto')

        for i in range(0, N_centroids):
            ax.scatter(centroids_in_lc[i][0], centroids_in_lc[i][1], c='white', marker='o')   
            ax.text(centroids_in_lc[i][0], centroids_in_lc[i][1], labels_in_lc[i], c='white')

        fig.savefig('segments.png', transparent=False)
        fig.clf()
        #'''

        '''
        i = 0
        for s in np.unique(self.segments):
            self.segments[self.segments == s] = i
            i += 1
        '''    
        end = time.time()
        print('\ntime = ', end-start)

        return self.segments    


    # Create data based on segments
    def create_data(self):
        # create data
        self.n_features = np.unique(self.segments).shape[0]
        self.num_samples = self.n_features
        lst = [[1]*self.n_features]
        for i in range(1, self.num_samples):
            lst.append([1]*self.n_features)
            lst[i][self.n_features-i] = 0    
        self.data = np.array(lst).reshape((self.num_samples, self.n_features))
        #print(self.data)
        #print(np.unique(self.segments))

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

            self.tf_odom_map_tmp = [transf_.transform.translation.x,transf_.transform.translation.y,transf_.transform.translation.z,transf_.transform.rotation.x,transf_.transform.rotation.y,transf_.transform.rotation.z,transf_.transform.rotation.w]
            self.tf_map_odom_tmp = [transf.transform.translation.x,transf.transform.translation.y,transf.transform.translation.z,transf.transform.rotation.x,transf.transform.rotation.y,transf.transform.rotation.z,transf.transform.rotation.w]
        except:
            print('tf except!!!')
            #pass    
            return

        pd.DataFrame(self.tf_odom_map_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_odom_map_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.tf_map_odom_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/tf_map_odom_tmp.csv', index=False)#, header=False)
        

        # save costmap in a right image format
        self.localCostmapOriginX = msg.info.origin.position.x
        self.localCostmapOriginY = msg.info.origin.position.y
        self.localCostmapResolution = msg.info.resolution
        self.costmap_info_tmp = [self.localCostmapResolution, msg.info.width, msg.info.height, self.localCostmapOriginX, self.localCostmapOriginY, msg.info.origin.orientation.z, msg.info.origin.orientation.w]

        # create np.array image object
        self.image = np.asarray(msg.data)
        self.image.resize((msg.info.height,msg.info.width))
        
        if self.plot_data == True:
            self.image_original = copy.deepcopy(self.image)
            #'''
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
            #'''

            self.image_100s = copy.deepcopy(self.image)
            self.image_100s[self.image_100s != 100] = 0
            #'''
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
            #'''

            self.image_99s = copy.deepcopy(self.image)
            self.image_99s[self.image_99s != 99] = 0
            #'''
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
            #'''

        self.image_99s_100s = copy.deepcopy(self.image)
        self.image_99s_100s[self.image_99s_100s < 99] = 0
        if self.plot_data == True:
            #'''
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
            #'''

        if self.plot_data == True:
            self.image_less_than_99 = copy.deepcopy(self.image)
            self.image_less_than_99[self.image_less_than_99 >= 99] = 0
            #'''
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
            #'''

        # Turn inflated area to free space and 100s to 99s
        self.image[self.image == 100] = 99
        self.image[self.image <= 98] = 0

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        self.image = self.image * 1.0

        # find image_rgb
        #self.image_rgb = gray2rgb(self.image)
        #image = np.stack(3 * (image,), axis=-1)

        # my change - to return grayscale to classifier_fn
        self.fudged_image = self.image.copy()
        self.fudged_image[:] = hide_color = 0

        # save indices of robot's odometry location in local costmap to class variables
        self.x_odom_index = round((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        self.y_odom_index = round((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)

        # find segments
        #self.segments = self.segment_local_costmap(self.image, self.image_rgb)
        #self.segments = self.segment_local_costmap_semantic()
        #self.segments = self.segment_local_costmap_semantic_new()
        self.segments = self.segment_local_costmap_semantic_new_2()

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
        #pd.DataFrame(self.odom_tmp).to_csv('~/amar_ws/lime_rt_data/odom_tmp.csv', index=False)#, header=False)
        
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
            '''
            p = np.array([self.global_plan_xs[-1], self.global_plan_ys[-1], 0.0])
            pnew = p.dot(r_) + t
            x_temp = round((pnew[0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = round((pnew[1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)
            '''

        self.global_plan_empty = False

        pd.DataFrame(self.global_plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/global_plan_tmp.csv', index=False)#, header=False)
        pd.DataFrame(self.plan_tmp).to_csv(self.dirCurr + '/' + self.dirName + '/plan_tmp.csv', index=False)#, header=False)
        
    # Define a callback for the local plan
    def local_plan_callback(self, msg):
        #print('\nlocal_plan_callback')
        
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

    def model_state_callback(self, states_msg):
        self.gazebo_names = states_msg.name
        self.gazebo_poses = states_msg.pose

    # Declare subscribers
    def main_(self):
        self.sub_local_plan = rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", Path, self.local_plan_callback)

        self.sub_global_plan = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.global_plan_callback)

        self.sub_footprint = rospy.Subscriber("/move_base/local_costmap/footprint", PolygonStamped, self.footprint_callback)

        # Initalize a subscriber to the odometry
        self.sub_odom = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odom_callback)

        #self.sub_amcl = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        # Initalize a subscriber to the local costmap
        self.sub_local_costmap = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)
        
        # semantic part
        self.semantic_tags = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/world_movable_chair_3/world_movable_chair_3_tags.csv')
        #print(self.semantic_tags)
        for i in range(0, self.semantic_tags.shape[0]):
            self.static_names.append(self.semantic_tags.iloc[i][1])

        self.semantic_global_map = np.array(pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/world_movable_chair_3/world_movable_chair_3.csv',index_col=None,header=None))
        #print(self.semantic_global_map.shape)
        self.semantic_global_map_info = pd.read_csv(self.dirCurr + '/src/navigation_explainer/src/world_movable_chair_3/world_movable_chair_3_info.csv',index_col=None,header=None)
        self.semantic_global_map_info = self.semantic_global_map_info.transpose()
        #print(self.semantic_global_map_info)
        self.semantic_global_map_resolution = float(self.semantic_global_map_info.iloc[1][1])
        print(self.semantic_global_map_resolution)
        self.semantic_global_map_origin_x = float(self.semantic_global_map_info.iloc[4][1])
        print(self.semantic_global_map_origin_x)
        self.semantic_global_map_origin_y = float(self.semantic_global_map_info.iloc[5][1])
        print(self.semantic_global_map_origin_y)

        # Initalize a subscriber to the "/gazebo/model_states" topic with the function "model_state_callback" as a callback
        self.sub_state = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)

        # find centroids
        '''
        regions = regionprops(self.semantic_global_map)
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            print('(v,cx,cy) = ', (v,cx,cy))
        '''

lime_rt_obj = lime_rt_sub()

lime_rt_obj.main_()

# Initialize the ROS Node named 'get_model_state', allow multiple nodes to be run with this name
rospy.init_node('world_movable_chair_sub', anonymous=True)

lime_rt_obj.tfBuffer = tf2_ros.Buffer()
lime_rt_obj.tf_listener = tf2_ros.TransformListener(lime_rt_obj.tfBuffer)


# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    #print('spinning')
    rospy.spin()