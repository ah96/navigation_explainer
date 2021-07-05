#!/usr/bin/env python3

import lime
import lime.lime_tabular

# lime image
from lime_explainer import lime_image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pokretanje ROS node-a
import shlex
from psutil import Popen

import rospy

from skimage import *
from skimage.color import gray2rgb, rgb2gray, rgb2lab
from skimage.util import regular_grid
from skimage.segmentation import mark_boundaries, felzenszwalb, slic, quickshift
from skimage.segmentation._slic import (_slic_cython, _enforce_label_connectivity_cython)
from skimage.segmentation.slic_superpixels import _get_grid_centroids, _get_mask_centroids
from skimage.measure import regionprops

from collections.abc import Iterable

import math
import copy

# important global variables
perturb_hide_color = 50

class ExplainRobotNavigation:

    def __init__(self, cmd_vel, odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data,
                 local_costmap_info,
                 amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train, X_test, modeParam, explanation_mode,
                 expID, num_samples, output_class_name, numOfFirstRowsToDelete, footprints):
        self.cmd_vel = cmd_vel
        self.odom = odom
        self.plan = plan
        self.global_plan = teb_global_plan
        self.local_plan = teb_local_plan
        self.current_goal = current_goal
        self.costmap_data = local_costmap_data
        self.costmap_info = local_costmap_info
        self.amcl_pose = amcl_pose
        self.tf_odom_map = tf_odom_map
        self.tf_map_odom = tf_map_odom
        self.map_data = map_data
        self.map_info = map_info
        self.X_train = X_train
        self.X_test = X_test
        self.mode = modeParam
        self.explanationMode = explanation_mode
        self.expID = expID
        self.num_samples = num_samples
        self.offset = numOfFirstRowsToDelete
        self.footprints = footprints

        # print important information
        print('self.mode: ', self.mode)
        print('self.explanationMode: ', self.explanationMode)
        print('self.expID: ', self.expID)
        print('self.num_samples: ', self.num_samples)
        print('output_class_name: ', output_class_name)
        print('self.offset: ', self.offset)

        # if mode is 'image'
        if self.explanationMode == 'image':
            self.explainer = lime_image.LimeImageExplainer(verbose=True)

            self.index = self.expID

            # Get local costmap
            self.local_costmap_original = self.costmap_data.iloc[(self.index) * 160:(self.index + 1) * 160, :] # srediti ovo 160
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            #'''
            # Turn inflated area to free space
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):
                    if 99 > self.image[i, j] > 0:
                        self.image[i, j] = 0
                    elif self.image[i, j] == 100:
                        self.image[i, j] = 99
            #'''

            # '''
            # Turn point free space to point obstacle
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):
                    if self.image[i, j] == 0:
                        # in the middle
                        if self.image.shape[0]-1 > i > 0 and self.image.shape[1]-1 > j > 0:
                            if self.image[i-1, j] == 99 and self.image[i+1, j] == 99 and self.image[i, j-1] == 99 and self.image[i, j+1] == 99:
                                self.image[i,j] = 99
                        # top left corner
                        elif i == 0 and j == 0:
                            if self.image[0, 1] == 99 and self.image[1, 0] == 99 and self.image[1, 1] == 99:
                                self.image[i,j] = 99
                        # top edge
                        if i == 0 and self.image.shape[1]-1 > j > 0:
                            if self.image[i, j-1] == 99 and self.image[i+1, j] == 99 and self.image[i, j+1] == 99:
                                self.image[i,j] = 99
                        # top right corner
                        elif i == 0 and j == self.image.shape[1]-1:
                            if self.image[0, self.image.shape[1]-2] == 99 and self.image[1, self.image.shape[1]-1] == 99 and self.image[1, self.image.shape[1]-2] == 99:
                                self.image[i,j] = 99
                        # bottom left corner
                        elif i == self.image.shape[0]-1 and j == 0:
                            if self.image[self.image.shape[0]-2, 0] == 99 and self.image[self.image.shape[0]-1, 1] == 99 and self.image[self.image.shape[0]-2, 1] == 99:
                                self.image[i,j] = 99
                        # bottom edge
                        if i == self.image.shape[0]-1 and self.image.shape[1]-1 > j > 0:
                            if self.image[i, j-1] == 99 and self.image[i-1, j] == 99 and self.image[i, j+1] == 99:
                                self.image[i,j] = 99
                        # bottom right corner
                        elif i == self.image.shape[0]-1 and j == self.image.shape[1]-1:
                            if self.image[self.image.shape[0]-2, self.image.shape[1]-2] == 99 and self.image[self.image.shape[0]-1, self.image.shape[1]-2] == 99 and self.image[self.image.shape[0]-2, self.image.shape[1]-1] == 99:
                                self.image[i,j] = 99
                        # left edge
                        if self.image.shape[0]-1 > i > 0 and j == 0:
                            if self.image[i - 1, j] == 99 and self.image[i + 1, j] == 99 and self.image[i, j + 1] == 99:
                                self.image[i, j] = 99
                        # right edge
                        if self.image.shape[0]-1 > i > 0 and j == self.image.shape[1]-1:
                            if self.image[i - 1, j] == 99 and self.image[i + 1, j] == 99 and self.image[i, j - 1] == 99:
                                self.image[i, j] = 99
            # '''

            self.image = self.image * 1.0

            # Saving data to .csv files for C++ node

            # Save footprint instance to a file
            self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
            self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
            self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False,
                                      header=False)

            # Save local plan instance to a file
            self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
            self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
            self.local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False,
                                       header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
            self.plan_tmp = self.plan_tmp.iloc[:, 1:]
            self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp = self.global_plan.loc[self.global_plan['ID'] == self.index + self.offset]
            self.global_plan_tmp = self.global_plan_tmp.iloc[:, 1:]
            self.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False,
                                        header=False)

            # Save costmap_info to file
            self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
            self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
            self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
            self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False,
                                         header=False)

            # Save amcl_pose to file
            self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
            self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
            self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
            self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False,
                                      header=False)

            # Save tf_odom_map to file
            self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
            self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
            self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False,
                                        header=False)

            # Save tf_map_odom to file
            self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
            self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
            self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False,
                                        header=False)

            # Save sampled odom to file
            self.odom_tmp = self.odom.iloc[self.index, :]
            self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
            self.odom_tmp = self.odom_tmp.iloc[:, 2:]
            self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

            # save costmap info
            self.localCostmapOriginX = self.costmap_info_tmp.iloc[0, 3]
            # print('self.localCostmapOriginX: ', self.localCostmapOriginX)
            self.localCostmapOriginY = self.costmap_info_tmp.iloc[0, 4]
            # print('self.localCostmapOriginY: ', self.localCostmapOriginY)
            self.localCostmapResolution = self.costmap_info_tmp.iloc[0, 0]
            # print('self.localCostmapResolution: ', self.localCostmapResolution)
            self.localCostmapHeight = self.costmap_info_tmp.iloc[0, 2]
            # print('self.localCostmapHeight: ', self.localCostmapHeight)
            self.localCostmapWidth = self.costmap_info_tmp.iloc[0, 1]
            # print('self.localCostmapWidth: ', self.localCostmapWidth)

            # save robot odometry location
            self.odom_x = self.odom_tmp.iloc[0, 0]
            # print('self.odom_x: ', self.odom_x)
            self.odom_y = self.odom_tmp.iloc[0, 1]
            # print('self.odom_y: ', self.odom_y)

            # indices of robot's odometry location in local costmap
            self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
            # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
            self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
            # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

            # indices of robot's odometry location in local costmap in a list - suitable for plotting
            self.x_odom_index = [self.localCostmapIndex_x_odom]
            self.y_odom_index = [self.localCostmapIndex_y_odom]

            # robot odometry orientation
            self.odom_z = self.odom_tmp.iloc[0, 2]
            self.odom_w = self.odom_tmp.iloc[0, 3]
            [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
            # print('roll: ', roll_odom)
            # print('pitch: ', pitch_odom)
            # print('yaw: ', yaw_odom)
            self.yaw_odom_x = math.cos(self.yaw_odom)
            self.yaw_odom_y = math.sin(self.yaw_odom)

            '''
            # indices of footprint's poses in local costmap
            self.footprint_x_list = []
            self.footprint_y_list = []
            for j in range(0, self.footprint_tmp.shape[0]):
                self.footprint_x_list.append(
                    int((self.footprint_tmp.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
                self.footprint_y_list.append(
                    int((self.footprint_tmp.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
            '''



        elif self.explanationMode == 'tabular':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.X_train),
                                                                    feature_names=self.X_train.columns, mode=self.mode,
                                                                    class_names=[output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False,
                                                                    sample_around_instance=False, random_state=None)

        elif self.explanationMode == 'tabular_costmap':
            self.index = self.expID;
            img = self.costmap_data.iloc[(self.index) * 160:(self.index + 1) * 160, :]
            lista = []
            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    lista.append(img.iloc[i, j])
            self.tabular_costmap = pd.DataFrame(lista)
            self.tabular_costmap = pd.DataFrame(self.tabular_costmap).transpose()
            # print(self.tabular_costmap.shape)
            # print(self.tabular_costmap)
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.tabular_costmap),
                                                                    feature_names=self.tabular_costmap.columns,
                                                                    mode=modeParam, class_names=[output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False)

    def explain_instance(self, expID):
        # redni broj instance
        self.expID = expID
        # print('self.expID: ', self.expID)

        # if mode is 'image'
        if self.explanationMode == 'image':

            # Use new variable in the algorithm
            img = copy.deepcopy(self.image)

            # my custom segmentation func
            segm_fn = 'custom_segmentation'

            self.explanation = self.explainer.explain_instance(img, self.classifier_fn_image, hide_color=perturb_hide_color, num_samples=self.num_samples, batch_size=128, segmentation_fn=segm_fn)

            #print('self.explanation: ', self.explanation)

            self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=True,
                                                                           negative_only=False, num_features=1,
                                                                           hide_rest=False,
                                                                           min_weight=0.0)  # min_weight=0.1 - default

            self.plotExplanation()


        elif self.explanationMode == 'tabular':
            # trazenje indexa reda instance (originalno ime reda instance u haman ulaznim datafrejmovima)
            self.index = self.X_train.index.values[self.expID];
            print('self.index: ', self.index)

            self.explanation = self.explainer.explain_instance(data_row=np.array(self.X_train.iloc[self.expID]),
                                                               predict_fn=self.classifier_fn_tabular,
                                                               num_samples=self.num_samples,
                                                               num_features=self.X_train.shape[1])

            print(self.explanation.as_list())
            fig = self.explanation.as_pyplot_figure()
            plt.savefig('explanation.png')


        elif self.explanationMode == 'tabular_costmap':
            self.explanation = self.explainer.explain_instance(data_row=self.tabular_costmap,
                                                               predict_fn=self.classifier_fn_tabular_costmap,
                                                               num_samples=self.num_samples,
                                                               num_features=self.tabular_costmap.shape[1])
            # print(self.explanation.as_list())
            fig = self.explanation.as_pyplot_figure()
            plt.savefig('explanation.png')

    def classifier_fn_image(self, sampled_instance):

        print('classifier_fn_image started')

        # sampled_instance info
        #print('sampled_instance: ', sampled_instance)
        #print('sampled_instance.shape: ', sampled_instance.shape)
        
        #'''
        # I will use channel 0 from sampled_instance as actual perturbed data
        # Perturbed pixel intensity is perturb_hide_color
        # Convert perturbed free space to obstacle (99), and perturbed obstacles to free space (0) in all perturbations
        for i in range(0, sampled_instance.shape[0]):
            for j in range(0, sampled_instance[i].shape[0]):
                for k in range(0, sampled_instance[i].shape[1]):
                    if sampled_instance[i][j, k, 0] == perturb_hide_color:
                        if self.image[j, k] == 0:
                            sampled_instance[i][j, k, 0] = 99
                            #print('free space')
                        elif self.image[j, k] == 99:
                            sampled_instance[i][j, k, 0] = 0
                            #print('obstacle')
        #'''

        #'''
        # Save perturbed costmap_data to file
        #sampled_instance = sampled_instance.astype(int)
        self.costmap_tmp = pd.DataFrame(sampled_instance[0][:, :, 0])
        for i in range(1, sampled_instance.shape[0]):
            self.costmap_tmp = pd.concat([self.costmap_tmp, pd.DataFrame(sampled_instance[i][:, :, 0])], join='outer', axis=0, sort=False)
        self.costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)
        # print('self.costmap_tmp.shape: ', self.costmap_tmp.shape)
        # self.costmap_tmp.to_csv('~/amar_ws/costmap_data.csv', index=False, header=False)
        #'''

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        # load command velocities
        cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('cmd_vel: ', cmd_vel)
        #print('cmd_vel.shape: ', cmd_vel.shape)

        # load local plans
        #local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')
        #print('local_plans: ', local_plans)
        #print('local_plans.shape: ', local_plans.shape)

        # load transformed plan
        #transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('transformed_plan: ', transformed_plan)
        #print('transformed_plan.shape: ', transformed_plan.shape)


        '''
        # Visualise last 10 perturbations and last 100 perturbations separately
        self.perturbations_visualization = sampled_instance[0][:, :, 0]
        for i in range(1, sampled_instance.shape[0]):
            if i == 10:
                self.perturbations_visualization_final = self.perturbations_visualization
                self.perturbations_visualization = sampled_instance[i][:, :, 0]
            elif i % 10 == 0 & i != 10:
                self.perturbations_visualization_final = np.concatenate(
                    (self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
                self.perturbations_visualization = sampled_instance[i][:, :, 0]
            else:
                self.perturbations_visualization = np.concatenate(
                    (self.perturbations_visualization, sampled_instance[i][:, :, 0]), axis=1)
        self.perturbations_visualization_final = np.concatenate(
            (self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
        '''
        '''
        # Save perturbations as .csv file
        for i in range(0, sampled_instance.shape[0]):
            pd.DataFrame(sampled_instance[i][:, :, 0]).to_csv('~/amar_ws/perturbation_' + str(i) + '.csv', index=False,
                                                              header=False)
        '''


        '''
        # indices of transformed plan's poses in local costmap
        transformed_plan_x_list = []
        transformed_plan_y_list = []
        for j in range(0, transformed_plan.shape[0]):
            transformed_plan_x_list.append(
                int((transformed_plan.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            transformed_plan_y_list.append(
                int((transformed_plan.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        # print('i: ', i)
        # print('local_plan_x_list.size(): ', len(local_plan_x_list))
        # print('local_plan_y_list.size(): ', len(local_plan_y_list))

        # plot every perturbation
        for i in range(0, sampled_instance.shape[0]):

            # plot perturbed local costmap
            plt.imshow(sampled_instance[i][:, :, 0])

            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []
            for j in range(0, local_plans.shape[0]):
                if local_plans.iloc[j, -1] == i:
                    local_plan_x_list.append(
                        int((local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
                    local_plan_y_list.append(
                        int((local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
            # print('i: ', i)
            # print('local_plan_x_list.size(): ', len(local_plan_x_list))
            # print('local_plan_y_list.size(): ', len(local_plan_y_list))

            # plot transformed plan
            plt.scatter(transformed_plan_x_list, transformed_plan_y_list, c='blue', marker='x')

            # plot footprint
            plt.scatter(self.footprint_x_list, self.footprint_y_list, c='green', marker='x')

            # plot local plan
            plt.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='x')

            # plot local plan last point
            if len(local_plan_x_list) != 0:
                plt.scatter([local_plan_x_list[-1]], [local_plan_y_list[-1]], c='black', marker='x')

            # plot robot's location and orientation
            plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

            # plot command velocities as text
            plt.text(0.0, -5.0, 'lin_x=' + str(round(cmd_vel.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(cmd_vel.iloc[i, 2], 2)))

            # save figure
            plt.savefig('perturbation_' + str(i) + '.png')
            plt.clf()
        '''

        # classification
        conditions = [
            ((cmd_vel['cmd_vel_ang_z'] >= 0.05) & (cmd_vel['cmd_vel_lin_x'] > 0)),
            ((cmd_vel['cmd_vel_ang_z'] <= -0.05) & (cmd_vel['cmd_vel_lin_x'] > 0)),
            ((cmd_vel['cmd_vel_ang_z'] < 0.05) & (cmd_vel['cmd_vel_ang_z'] > -0.05) & (cmd_vel['cmd_vel_lin_x'] > 0)),
            ((cmd_vel['cmd_vel_ang_z'] >= 0.05) & (cmd_vel['cmd_vel_lin_x'] < 0)),
            ((cmd_vel['cmd_vel_ang_z'] <= -0.05) & (cmd_vel['cmd_vel_lin_x'] < 0)),
            ((cmd_vel['cmd_vel_ang_z'] < 0.05) & (cmd_vel['cmd_vel_ang_z'] > -0.05) & (cmd_vel['cmd_vel_lin_x'] < 0)),
            (abs(cmd_vel['cmd_vel_lin_x']) < 0.01) # & (cmd_vel['cmd_vel_lin_x'] > -0.01))
        ]

        valuesLeftAhead = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cmd_vel['left_ahead'] = np.select(conditions, valuesLeftAhead)

        valuesRightAhead = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cmd_vel['right_ahead'] = np.select(conditions, valuesRightAhead)

        valuesStraightAhead = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        cmd_vel['straight_ahead'] = np.select(conditions, valuesStraightAhead)

        valuesLeftBack = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        cmd_vel['left_back'] = np.select(conditions, valuesLeftBack)

        valuesRightBack = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        cmd_vel['right_back'] = np.select(conditions, valuesRightBack)

        valuesStraightBack = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        cmd_vel['straight_back'] = np.select(conditions, valuesStraightBack)

        valuesStop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        cmd_vel['stop'] = np.select(conditions, valuesStop)

        print('classifier_fn_image ended')

        return np.array(cmd_vel.iloc[:, 3:4])

    def plotExplanation(self):
        print('plotExplanation starts')

        # print important information
        print('self.mode: ', self.mode)
        print('self.explanationMode: ', self.explanationMode)
        print('self.expID: ', self.expID)
        print('self.num_samples: ', self.num_samples)
        print('self.offset: ', self.offset)
        print('self.costmap_info.shape[0]: ', self.costmap_info.shape[0])

        # plot local costmap
        # plot robot odometry location
        plt.scatter(self.x_odom_index, self.y_odom_index, c='blue', marker='o')

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        # print('self.local_plan_tmp.shape: ', self.local_plan_tmp.shape)
        for i in range(0, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(
                int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(
                int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        plt.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')

        # robot's odometry orientation
        plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

        # plot costmap
        img = np.array(self.image)
        plt.imshow(img)
        plt.savefig('LIME_image_local_costmap.png')
        plt.clf()

        # plot global map
        # map info
        self.mapOriginX = self.map_info.iloc[0, 4]
        #print('self.mapOriginX: ', self.mapOriginX)
        self.mapOriginY = self.map_info.iloc[0, 5]
        #print('self.mapOriginY: ', self.mapOriginY)
        self.mapResolution = self.map_info.iloc[0, 1]
        #print('self.mapResolution: ', self.mapResolution)
        self.mapHeight = self.map_info.iloc[0, 3]
        #print('self.mapHeight: ', self.mapHeight)
        self.mapWidth = self.map_info.iloc[0, 2]
        #print('self.mapWidth: ', self.mapWidth)

        # robot amcl location
        self.amcl_x = self.amcl_pose_tmp.iloc[0, 0]
        #print('self.amcl_x: ', self.amcl_x)
        self.amcl_y = self.amcl_pose_tmp.iloc[0, 1]
        #print('self.amcl_y: ', self.amcl_y)

        # indices of robot's odometry location in map
        self.mapIndex_x_amcl = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        #print('self.mapIndex_x_amcl: ', self.mapIndex_x_amcl)
        self.mapIndex_y_amcl = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        #print('self.mapIndex_y_amcl: ', self.mapIndex_y_amcl)

        # indices of robot's amcl location in map in a list - suitable for plotting
        self.x_amcl_index = [self.mapIndex_x_amcl]
        self.y_amcl_index = [self.mapIndex_y_amcl]
        plt.scatter(self.x_amcl_index, self.y_amcl_index, c='yellow', marker='o')

        # robot amcl orientation
        self.amcl_z = self.amcl_pose_tmp.iloc[0, 2]
        self.amcl_w = self.amcl_pose_tmp.iloc[0, 3]
        [self.yaw_amcl, pitch_amcl, roll_amcl] = self.quaternion_to_euler(0.0, 0.0, self.amcl_z, self.amcl_w)
        #print('roll_amcl: ', roll_amcl)
        #print('pitch_amcl: ', pitch_amcl)
        #print('yaw_amcl: ', self.yaw_amcl)
        self.yaw_amcl_x = math.cos(self.yaw_amcl)
        self.yaw_amcl_y = math.sin(self.yaw_amcl)
        plt.quiver(self.x_amcl_index, self.y_amcl_index, self.yaw_amcl_x, self.yaw_amcl_y, color='white')

        # plan from global planner
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(19, self.plan_tmp.shape[0], 20):
            self.plan_x_list.append(int((self.plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.plan_y_list.append(int((self.plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        plt.scatter(self.plan_x_list, self.plan_y_list, c='red', marker='<')

        # global plan from teb algorithm
        self.global_plan_x_list = []
        self.global_plan_y_list = []
        for i in range(19, self.global_plan_tmp.shape[0], 20):
            self.global_plan_x_list.append(
                int((self.global_plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.global_plan_y_list.append(
                int((self.global_plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        plt.scatter(self.global_plan_x_list, self.global_plan_y_list, c='yellow', marker='>')

        # plot robot's location in the map
        x_map = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        y_map = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        plt.scatter(x_map, y_map, c='red', marker='o')

        # plot map, fill -1 with 100
        map_tmp = self.map_data
        for i in range(0, map_tmp.shape[0]):
            for j in range(0, map_tmp.shape[1]):
                if map_tmp.iloc[i, j] == -1:
                    map_tmp.iloc[i, j] = 100
        plt.imshow(map_tmp)
        plt.savefig('LIME_image_map.png')
        plt.clf()

        # plot image_temp
        plt.imshow(self.temp_img)
        plt.savefig('LIME_image_temp_img.png')
        plt.clf()

        # plot mask
        plt.imshow(self.mask)
        plt.savefig('LIME_image_mask.png')
        plt.clf()

        # plot last 100 perturbations
        #plt.imshow(self.perturbations_visualization_final)
        #plt.savefig('LIME_perturbations.png')
        #plt.clf()

        # plot last 10 perturbations
        #plt.imshow(self.perturbations_visualization)
        #plt.savefig('LIME_perturbations_last_row.png')
        #plt.clf()

        # plt.imshow(mark_boundaries(self.temp_img / 2 + 0.5, self.mask))
        # plot explanation
        plt.imshow(mark_boundaries(self.temp_img, self.mask))
        plt.scatter(self.x_odom_index, self.y_odom_index, c='blue', marker='o')
        plt.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')
        plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')
        plt.savefig('LIME_image_explanation.png')
        plt.close()

        print('plotExplanation ends')

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return [yaw, pitch, roll]

    def testSegmentation(self):

        print('Test segmentation function beginning')

        img_ = copy.deepcopy(self.image)

        # Save local costmap to .csv file
        # pd.DataFrame(img_).to_csv('~/amar_ws/local_costmap_gray_segmentation_test.csv', index=False, header=False)

        # Save local costmap as gray image
        plt.imshow(img_)
        plt.savefig('testSegmentation_image_gray.png')
        plt.clf()

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # Save local costmap as rgb image
        plt.imshow(rgb)
        plt.savefig('testSegmentation_image_rgb.png')
        plt.clf()

        # Superpixel segmentation with skimage functions

        # felzenszwalb
        # segments = felzenszwalb(rgb, scale=100, sigma=5, min_size=30, multichannel=True)
        # segments = felzenszwalb(rgb, scale=1, sigma=0.8, min_size=20, multichannel=True)  # default

        # quickshift
        # segments = quickshift(rgb, ratio=0.0001, kernel_size=8, max_dist=10, return_tree=False, sigma=0.0, convert2lab=True, random_seed=42)
        # segments = quickshift(rgb, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, random_seed=42) # default

        # slic
        # segments = slic(rgb, n_segments=6, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=5, slic_zero=False, start_label=None, mask=None)
        # segments = slic(rgb, n_segments=100, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=None, mask=None) # default

        # Turn segments gray image to rgb image
        #segments_rgb = gray2rgb(segments)

        # Generate segments - superpixels with my slic function
        segments = self.mySlic(rgb)

        # Save segments to .csv file
        #pd.DataFrame(segments).to_csv('~/amar_ws/segments_segmentation_test.csv', index=False, header=False)

        print('Test segmentation function ending')

    def mySlic(self, img_rgb):

        print('mySlic for testSegmentation starts')

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = img_rgb[:, :, 0]
        # Save segments_1 as a picture
        plt.imshow(img)
        plt.savefig('testSegmentation_img.png')
        plt.clf()

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # plot segments_1 with centroids and labels
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_1 as a picture
        plt.imshow(segments_1)
        plt.savefig('testSegmentation_segments_1.png')
        plt.clf()
        # find segments_unique_1
        segments_unique_1 = np.unique(segments_1)
        print('segments_unique_1: ', segments_unique_1)
        print('segments_unique_1.shape: ', segments_unique_1.shape)

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)
        # make obstacles on segments_2 nice - not needed
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 99:
                   segments_2[i, j] = segments_1[i, j] + segments_unique_2.shape[0]
        '''
        # plot segments_2 with centroids and labels
        regions = regionprops(segments_2)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_2 as a picture
        plt.imshow(segments_2)
        plt.savefig('testSegmentation_segments_2.png')
        plt.clf()
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]

        # plot segments with centroids and labels/weights
        plt.imshow(segments_1)
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_1 as a picture before nice segment numbering
        plt.savefig('testSegmentation_segments_beforeNiceNumbering.png')
        plt.clf()
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique: ', segments_unique)
        print('segments_unique.shape: ', segments_unique.shape)

        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        # find segments_unique after nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique (with nice numbering): ', segments_unique)
        print('segments_unique.shape (with nice numbering): ', segments_unique.shape)

        # print explanation
        print('self.exp: ', self.exp)
        print('len(self.exp): ', len(self.exp))

        # plot segments with centroids and labels/weights
        plt.imshow(segments_1)
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            #plt.text(centers[i][0], centers[i][1], str(v))
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    #print('j: ', j)
                    plt.text(centers[i][0], centers[i][1], str(round(self.exp[j][1],4)))   #str(v))
                    break
            i = i + 1

        # Save segments as a picture
        plt.savefig('testSegmentation_segments.png')
        plt.clf()

        print('mySlic for testSegmentation ends')

        return segments_1









    # functions that I currently do not use
    def slic_help(self, image_rgb, n_segments=100, compactness=10., max_iter=10, sigma=0, spacing=None,
                  multichannel=True, convert2lab=None,
                  enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, start_label=None, mask=None, channel_axis=-1):
        slic_zero = False
        image_rgb = img_as_float(image_rgb)
        float_dtype = _supported_float_type(image_rgb.dtype)
        image_rgb = image_rgb.astype(float_dtype, copy=False)

        use_mask = mask is not None
        dtype = image_rgb.dtype

        is_2d = False

        multichannel = channel_axis is not None
        if image_rgb.ndim == 2:
            # 2D grayscale image
            image_rgb = image_rgb[np.newaxis, ..., np.newaxis]
            is_2d = True
        elif image_rgb.ndim == 3 and multichannel:
            # Make 2D multichannel image 3D with depth = 1
            image_rgb = image_rgb[np.newaxis, ...]
            is_2d = True
        elif image_rgb.ndim == 3 and not multichannel:
            # Add channel as single last dimension
            image_rgb = image_rgb[..., np.newaxis]

        if multichannel and (convert2lab or convert2lab is None):
            if image_rgb.shape[channel_axis] != 3 and convert2lab:
                raise ValueError("Lab colorspace conversion requires a RGB image.")
            elif image_rgb.shape[channel_axis] == 3:
                image_rgb = rgb2lab(image_rgb)

        if start_label is None:
            if use_mask:
                start_label = 1
            else:
                import warnings
                warnings.warn("skimage.measure.label's indexing starts from 0. " +
                              "In future version it will start from 1. " +
                              "To disable this warning, explicitely " +
                              "set the `start_label` parameter to 1.",
                              FutureWarning, stacklevel=2)
                start_label = 0

        if start_label not in [0, 1]:
            raise ValueError("start_label should be 0 or 1.")

        # initialize cluster centroids for desired number of segments
        update_centroids = False
        if use_mask:
            mask = np.ascontiguousarray(mask, dtype=bool).view('uint8')
            if mask.ndim == 2:
                mask = np.ascontiguousarray(mask[np.newaxis, ...])
            if mask.shape != image_rgb.shape[:3]:
                raise ValueError("image and mask should have the same shape.")
            centroids, steps = _get_mask_centroids(mask, n_segments, multichannel)
            update_centroids = True
        else:
            centroids, steps = _get_grid_centroids(image_rgb, n_segments)

        if spacing is None:
            spacing = np.ones(3, dtype=dtype)
        elif isinstance(spacing, (list, tuple)):
            spacing = np.ascontiguousarray(spacing, dtype=dtype)

        if not isinstance(sigma, Iterable):
            sigma = np.array([sigma, sigma, sigma], dtype=dtype)
            sigma /= spacing.astype(dtype)
        elif isinstance(sigma, (list, tuple)):
            sigma = np.array(sigma, dtype=dtype)
        if (sigma > 0).any():
            # add zero smoothing for multichannel dimension
            sigma = list(sigma) + [0]
            from scipy import ndimage as ndi
            image = ndi.gaussian_filter(image_rgb, sigma)

        n_centroids = centroids.shape[0]
        segments = np.ascontiguousarray(np.concatenate(
            [centroids, np.zeros((n_centroids, image_rgb.shape[3]))],
            axis=-1), dtype=dtype)

        # Scaling of ratio in the same way as in the SLIC paper so the
        # values have the same meaning
        step = max(steps)
        ratio = 1.0 / compactness

        image = np.ascontiguousarray(image_rgb * ratio, dtype=dtype)

        if update_centroids:
            # Step 2 of the algorithm [3]_
            _slic_cython(image_rgb, mask, segments, step, max_iter, spacing,
                         slic_zero, ignore_color=True,
                         start_label=start_label)

        labels = _slic_cython(image_rgb, mask, segments, step, max_iter,
                              spacing, slic_zero, ignore_color=False,
                              start_label=start_label)

        if enforce_connectivity:
            if use_mask:
                segment_size = mask.sum() / n_centroids
            else:
                segment_size = np.prod(image.shape[:3]) / n_centroids
            min_size = int(min_size_factor * segment_size)
            max_size = int(max_size_factor * segment_size)
            labels = _enforce_label_connectivity_cython(
                labels, min_size, max_size, start_label=start_label)

        if is_2d:
            labels = labels[0]

        return labels, centroids

    def matrixflip(self, m, d):
        myl = np.array(m)
        if d == 'v':
            return np.flip(myl, axis=0)
        elif d == 'h':
            return np.flip(myl, axis=1)

    def classifier_fn_tabular(self, sampled_instance):

        print('classifier_fn_tabular started')

        # Save sampled_instance to self.sampled_instance
        self.sampled_instance = sampled_instance

        # Save local plan instance to a file
        local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index]
        local_plan_tmp = local_plan_tmp.iloc[:, 1:]
        local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan instance to a file
        plan_tmp = self.plan.loc[self.plan['ID'] == self.index]
        plan_tmp = plan_tmp.iloc[:, 1:]
        plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save costmap_data to file
        costmap_tmp = self.costmap_data.iloc[(self.index - self.offset) * 160:(self.index - self.offset + 1) * 160,
                      :]  # doraditi ovo 160
        costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)

        # Save costmap_info to file
        costmap_info_tmp = self.costmap_info.iloc[self.index - self.offset, :]
        costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
        costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]
        costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose to file
        amcl_pose_tmp = self.amcl_pose.iloc[self.index - self.offset, :]
        amcl_pose_tmp = pd.DataFrame(amcl_pose_tmp).transpose()
        amcl_pose_tmp = amcl_pose_tmp.iloc[:, 1:]
        amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map to file
        tf_odom_map_tmp = self.tf_odom_map.iloc[self.index - self.offset, :]
        tf_odom_map_tmp = pd.DataFrame(tf_odom_map_tmp).transpose()
        tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom to file
        tf_map_odom_tmp = self.tf_map_odom.iloc[self.index - self.offset, :]
        tf_map_odom_tmp = pd.DataFrame(tf_map_odom_tmp).transpose()
        tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save sampled odom to file
        odom_tmp = pd.concat([self.odom.iloc[self.index - self.offset, :], self.odom.iloc[self.index - self.offset, :]],
                             join='outer', axis=1, sort=False)
        for i in range(0, self.num_samples - 2):
            odom_tmp = pd.concat([odom_tmp, self.odom.iloc[self.index - self.offset, :]], join='outer', axis=1,
                                 sort=False)
        odom_tmp = odom_tmp.transpose()
        odom_tmp = odom_tmp.iloc[:, 2:]
        for i in range(0, self.num_samples):
            odom_tmp.iloc[i, -2] = sampled_instance[i, 0]
            odom_tmp.iloc[i, -1] = sampled_instance[i, 1]

        odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        # start ROS node
        node_process = Popen(shlex.split('rosrun teb_local_planner perturb_node_tabular'))

        rospy.sleep(7)

        # stop ROS node
        node_process.terminate()

        cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')

        print('classifier_fn_tabular ended')

        return np.array(cmd_vel.iloc[:, 2])  # srediti index ovdje

    # Treba sample input_data pretvoriti u ispravne formate i publishati na ispravne topice da bi ih local_planner node mogao koristiti
    def classifier_fn_tabular_costmap(self, sampled_instance):

        print('classifier_fn_tabular_costmap started')
        print("Broj perturbacija: ", sampled_instance.shape[0])

        # Save sampled_instance to self.sampled_instance
        self.sampled_instance = sampled_instance

        # Save local plan instance to a file
        local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
        local_plan_tmp = local_plan_tmp.iloc[:, 1:]
        local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan instance to a file
        plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
        plan_tmp = plan_tmp.iloc[:, 1:]
        plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save costmap_info to file
        costmap_info_tmp = self.costmap_info.iloc[self.index, :]
        costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
        costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]
        costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose to file
        amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
        amcl_pose_tmp = pd.DataFrame(amcl_pose_tmp).transpose()
        amcl_pose_tmp = amcl_pose_tmp.iloc[:, 1:]
        amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map to file
        tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
        tf_odom_map_tmp = pd.DataFrame(tf_odom_map_tmp).transpose()
        tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom to file
        tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
        tf_map_odom_tmp = pd.DataFrame(tf_map_odom_tmp).transpose()
        tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save sampled odom to file
        odom_tmp = self.odom.iloc[self.index, :]
        odom_tmp = pd.DataFrame(odom_tmp).transpose()
        odom_tmp = odom_tmp.iloc[:, 2:]
        odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        # Save costmap_data to file - srediti ove 160-ke
        costmap_tmp = pd.DataFrame(sampled_instance[0][0:160]).transpose()
        for j in range(1, 160):
            costmap_tmp = pd.concat(
                [costmap_tmp, pd.DataFrame(sampled_instance[0][160 * j: 160 * (j + 1)]).transpose()], join='outer',
                axis=0, sort=False)
        for i in range(1, self.num_samples):
            if i % 100 == 0:
                print('Current sample: ', i)
            img = pd.DataFrame(sampled_instance[i][0:160]).transpose()
            for j in range(1, 160):
                img = pd.concat([img, pd.DataFrame(sampled_instance[i][160 * j: 160 * (j + 1)]).transpose()],
                                join='outer', axis=0, sort=False)
            costmap_tmp = pd.concat([costmap_tmp, img], join='outer', axis=0, sort=False)
        print(costmap_tmp.shape)
        costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)

        # start ROS node
        node_process = Popen(shlex.split('rosrun teb_local_planner perturb_node_tabular_costmap'))

        rospy.sleep(20)

        # stop ROS node
        node_process.terminate()

        cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        print(cmd_vel)

        '''
        # classification with one-hot encoding
        conditions = [
            ((cmd_vel['cmd_vel_ang_z'] < 0) & (cmd_vel['cmd_vel_lin_x'] >= 0)) | ((cmd_vel['cmd_vel_ang_z'] >= 0) & (cmd_vel['cmd_vel_lin_x'] < 0)),
            ((cmd_vel['cmd_vel_ang_z'] >= 0) & (cmd_vel['cmd_vel_lin_x'] >= 0)) | ((cmd_vel['cmd_vel_ang_z'] < 0) & (cmd_vel['cmd_vel_lin_x'] < 0))
            ]

        valuesRight = [1.0, 0.0]

        cmd_vel['right'] = np.select(conditions, valuesRight)
        
        valuesLeft = [0.0, 1.0]

        cmd_vel['left'] = np.select(conditions, valuesLeft)

        print(cmd_vel.iloc[:,3:5])
        '''

        print('classifier_fn_tabular_costmap ended')

        return np.array(cmd_vel.iloc[:, 2])

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}

def _supported_float_type(input_dtype, allow_complex=False):
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


'''
# Other code for dealing with calling other ROS node from this one
import roslaunch
package = 'teb_local_planner'
executable = 'perturb_node_image'
node = roslaunch.core.Node(package, executable)
launch = roslaunch.scriptapi.ROSLaunch()
launch.start()

#process = launch.launch(node)
#print(process.is_alive())
# Ovdje uvesti ros services
#rospy.sleep(20)
#process.stop()
# stop ROS node
#node_process.terminate()
# finish killing node proces
#node_process.terminate()

#node_process = Popen(shlex.split('rosnode kill /perturb_node_image'))
'''


''' 
    
    # Plot segments with legend
    from matplotlib.patches import Rectangle
    list_with_data = []
    for i in range(0, segments_unique.shape[0]):
        list_temp = [segments_unique[i], [200, 0.0, 0.0, 200.0],
                     str(segments_unique[i])]  # str(self.explanation[i][1])]
        list_with_data.append(list_temp)
    df_legend = pd.DataFrame(list_with_data, columns=['key', 'color', 'weight'])

    handles_1 = [Rectangle((0, 0), 0.8, 0.8, color=[float(c) / 255 for c in color_list]) for color_list in
                 df_legend['color']]
    labels = df_legend['weight']
    plt.figure()
    plt.subplots_adjust(hspace=0)  # plt.tight_layout()
    plt.rcParams.update({'legend.fontsize': 20})
    plt.rc(('xtick', 'ytick'), color=(1, 1, 1, 0))
    plt.subplot(2, 1, 1), plt.imshow(segments, aspect='auto')
    plt.subplot(2, 1, 2), plt.legend(handles_1, labels, mode='expand', ncol=3)
    plt.savefig('segmentation_test_segments_with_weights.png')
    plt.clf()

    # Generate segments - superpixels with centroids
    segments_help, centroids_help = self.slic_help(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0,
                                              spacing=None, multichannel=True, convert2lab=True,
                                              enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5,
                                              start_label=1, mask=None)
    print('centroids_help: ', centroids_help)
    print('centroids_help.shape: ', centroids_help.shape)
    segments_help_unique = np.unique(segments_help)
    print('np.unique(segments_help): ', segments_help_unique)
    # Save segments as an image
    for i in range(0, centroids_help.shape[0]):
        plt.scatter(centroids_help[i][1], centroids_help[i][2], c='white', marker='o')
    plt.imshow(segments_help)
    plt.savefig('segmentation_test_segments_help.png')
    plt.clf()



    np.array_equal(fn_result, original_result)



    # creating a image object (main image) 
    #im1 = Image.open(r"/home/robolab/amar_ws/explanation.png")
    #print('Ucitana slika')
    #print(im1)
    #print(type(im1))
    
    amcl_w = self.amcl_pose_tmp.iloc[:, 3]
    amcl_z = self.amcl_pose_tmp.iloc[:, 2]
    yaw = math.atan2(2.0*(0.0*amcl_z + amcl_w*0.0), amcl_w*amcl_w - 0.0*0.0 - 0.0*0.0 + amcl_z*amcl_z);
    print(yaw)
    pitch = math.asin(-2.0*(0.0*amcl_z - amcl_w*0.0));
    print(pitch)
    roll = math.atan2(2.0*(0.0*0.0 + amcl_w*amcl_z), amcl_w*amcl_w + 0.0*0.0 - 0.0*0.0 - amcl_z*amcl_z);
    print(roll)


    # from classifier_fn_image function
    # saving sampling costmaps as pics
    #for i in range(1, 10):                                 
        #img =  sampled_instance[i][:,:,0]
        #img =  img * 2.55
        #img = img.astype(int)            
        #img = Image.fromarray((img).astype(np.uint8))
        #img.save("sampled_instance_" + str(i) + ".png")
        
    #    self.final = np.concatenate((self.final, sampled_instance[i][:,:,0] * 2.55), axis = 1)            
    #   self.final_frame = cv.hconcat((self.final_frame, sampled_instance[i][:,:,0] * 2.55))
              
    #self.final_final = self.final
    #self.final_frame_final = self.final_frame

    # saving sampling costmaps as pics
    #for i in range(10, sampled_instance.shape[0]):

        ## zakljucak -- sve tri dimenzije sampled_instance arraya su jednake
        #comparison01 = sampled_instance[i][:,:,0] == sampled_instance[i][:,:,1]
        #equal_arrays01 = comparison01.all()
        #comparison12 = sampled_instance[i][:,:,1] == sampled_instance[i][:,:,2]
        #equal_arrays12 = comparison12.all()
        #comparison02 = sampled_instance[i][:,:,0] == sampled_instance[i][:,:,2]
        #equal_arrays02 = comparison02.all()
        #if equal_arrays01 and equal_arrays12 and equal_arrays02:
            #print('Equal: ' + str(i))

        #print(sampled_instance[i])
        #img = rgb2gray(img)
        #img = img.astype(int)
        #print("sampled_instance_" + str(i) + " gray:")
        #print(type(img))
        #print(img.shape)
        #print(img)
        
        #img =  sampled_instance[i][:,:,0]
        #img =  img * 2.55
        #img = img.astype(int)            
        #img = Image.fromarray((img).astype(np.uint8))
        #img.save("sampled_instance_" + str(i) + ".png")

    #    if i % 10 == 0:
    #        self.final_final = np.concatenate((self.final_final, self.final), axis = 0)
    #        self.final_frame_final = cv.vconcat((self.final_frame_final, self.final_frame))    
    #        self.final = sampled_instance[i][:,:,0] * 2.55            
    #        self.final_frame = sampled_instance[i][:,:,0] * 2.55
    #        continue
    #                                
    #    self.final = np.concatenate((self.final, sampled_instance[i][:,:,0] * 2.55), axis = 1)
    #    self.final_frame = cv.hconcat((self.final_frame, sampled_instance[i][:,:,0] * 2.55))

        #print("sampled_instance_" + str(i) + ":")
        #img = rgb2gray(sampled_instance[i])
        #img = sampled_instance[i][:,:,0]
        #df = pd.DataFrame(img)
        #print(df)
    
    #self.final = self.final.astype(int)        
    #self.final = Image.fromarray((self.final).astype(np.uint8))
    #self.final.save("final.png")
    
    #self.final_final = self.final_final.astype(int)        
    #self.final_final = Image.fromarray((self.final_final).astype(np.uint8))
    #self.final_final.save("final_final.png")
    
    #cv.imwrite("final_frame.png", self.final_frame)

    #cv.imwrite("final_frame_final.png", self.final_frame_final)    
'''
