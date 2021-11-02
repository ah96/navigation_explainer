#!/usr/bin/env python3

# import time tabular
import lime
import lime.lime_tabular

import time

import os

# lime image - my implementation
from lime_explainer import lime_image

# for managing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculations
import math
import copy

# for _supported_float_type function
from collections.abc import Iterable

# for running ROS node
import shlex
from psutil import Popen
import rospy

# for managing image
from skimage import *
from skimage.color import gray2rgb, rgb2gray, rgb2lab
from skimage.util import regular_grid
from skimage.segmentation import mark_boundaries, felzenszwalb, slic, quickshift
from skimage.segmentation._slic import (_slic_cython, _enforce_label_connectivity_cython)
from skimage.segmentation.slic_superpixels import _get_grid_centroids, _get_mask_centroids
from skimage.measure import regionprops

# important global variables
perturb_hide_color_value = 50

class ExplainRobotNavigation:

    def __init__(self, cmd_vel, odom, plan, global_plan, local_plan, current_goal, local_costmap_data,
                 local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train, X_test, tabular_mode, explanation_mode,
                 num_samples, output_class_name, num_of_first_rows_to_delete, footprints, test_type, costmap_size):
        print('Constructor starting\n')

        # save variables as class variables
        self.cmd_vel_original = cmd_vel
        self.odom = odom
        self.plan = plan
        self.global_plan = global_plan
        self.local_plan = local_plan
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
        self.tabular_mode = tabular_mode
        self.explanation_mode = explanation_mode
        self.num_samples = num_samples
        self.output_class_name = output_class_name
        self.offset = num_of_first_rows_to_delete
        self.footprints = footprints
        self.test_type = test_type
        self.costmap_size = costmap_size

        # manually modified LIME image
        if self.explanation_mode == 'image':
            self.explainer = lime_image.LimeImageExplainer(verbose=True)


        elif self.explanation_mode == 'tabular':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.X_train),
                                                                    feature_names=self.X_train.columns, mode=self.mode,
                                                                    class_names=[output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False,
                                                                    sample_around_instance=False, random_state=None)

        elif self.explanation_mode == 'tabular_costmap':
            self.index = self.expID
            img = self.costmap_data.iloc[(self.index) * 160:(self.index + 1) * 160, :]
            lista = []
            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    lista.append(img.iloc[i, j])
            self.tabular_costmap = pd.DataFrame(lista)
            self.tabular_costmap = pd.DataFrame(self.tabular_costmap).transpose()
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.tabular_costmap),
                                                                    feature_names=self.tabular_costmap.columns,
                                                                    mode=modeParam, class_names=[output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False)

        print('Constructor ending\n')

    def explain_instance(self, expID):
        print('explain_instance function starting\n')

        self.expID = expID
            
        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.index = self.expID

            self.printImportantInformation()

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

            '''
            # If a custom costmap is used - TO-DO: make custom map loading a separate case in explainer.py
            #self.local_costmap_original.to_csv('~/amar_ws/costmapToChange.csv', index=False, header=True)
            self.local_costmap_original = pd.read_csv('~/amar_ws/costmapToChange.csv')
            '''

            # Make image a np.array deepcopy of local_costmap_original
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            # Turn inflated area to free space and 100s to 99s
            self.inflatedToFree()

            # Turn point free space (that is surrounded by obstacles) to point obstacle - not really needed
            #self.PFP2PO()

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0

            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveData()

            # Use new variable in the algorithm - possible time saving
            img = copy.deepcopy(self.image)

            # my custom segmentation func
            segm_fn = 'custom_segmentation'

            self.explanation = self.explainer.explain_instance(img, self.classifier_fn_image, hide_color=perturb_hide_color_value, num_samples=self.num_samples,
                                                               batch_size=1024, segmentation_fn=segm_fn, top_labels=10)
            
            self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=False,
                                                                           negative_only=False, num_features=100,
                                                                           hide_rest=False,
                                                                           min_weight=0.08)  # min_weight=0.1 - default

            #self.plotExplanationMinimal()
            #self.plotExplanationMinimalFlipped()
            #self.pl#otExplanation()
            #self.plotExplanationFlipped()


        elif self.explanation_mode == 'tabular':
            # search for instance queue index (original instance queue name in almost (haman) input data frames)
            self.index = self.X_train.index.values[self.expID]
            print('self.index: ', self.index)

            self.explanation = self.explainer.explain_instance(data_row=np.array(self.X_train.iloc[self.expID]),
                                                               predict_fn=self.classifier_fn_tabular,
                                                               num_samples=self.num_samples,
                                                               num_features=self.X_train.shape[1])

            print(self.explanation.as_list())
            fig = self.explanation.as_pyplot_figure()
            plt.savefig('explanation.png')


        elif self.explanation_mode == 'tabular_costmap':
            self.explanation = self.explainer.explain_instance(data_row=self.tabular_costmap,
                                                               predict_fn=self.classifier_fn_tabular_costmap,
                                                               num_samples=self.num_samples,
                                                               num_features=self.tabular_costmap.shape[1])
            # print(self.explanation.as_list())
            fig = self.explanation.as_pyplot_figure()
            plt.savefig('explanation.png')

        print('explain_instance function ending')


    def classifier_fn_image(self, sampled_instance):

        print('classifier_fn_image started')

        # sampled_instance info
        #print('sampled_instance: ', sampled_instance)
        #print('sampled_instance.shape: ', sampled_instance.shape)
        
        #'''
        # I will use channel 0 from sampled_instance as actual perturbed data
        # Perturbed pixel intensity is perturb_hide_color_value
        # Convert perturbed free space to obstacle (99), and perturbed obstacles to free space (0) in all perturbations
        for i in range(0, sampled_instance.shape[0]):
            for j in range(0, sampled_instance[i].shape[0]):
                for k in range(0, sampled_instance[i].shape[1]):
                    if sampled_instance[i][j, k, 0] == perturb_hide_color_value:
                        if self.image[j, k] == 0:
                            sampled_instance[i][j, k, 0] = 99
                            #print('free space')
                        elif self.image[j, k] == 99:
                            sampled_instance[i][j, k, 0] = 0
                            #print('obstacle')
        #'''

        #'''
        # Save perturbed costmap_data to file for C++ node
        #sampled_instance = sampled_instance.astype(int)
        self.costmap_tmp = pd.DataFrame(sampled_instance[0][:, :, 0])
        for i in range(1, sampled_instance.shape[0]):
            self.costmap_tmp = pd.concat([self.costmap_tmp, pd.DataFrame(sampled_instance[i][:, :, 0])], join='outer', axis=0, sort=False)
        self.costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)
        # print('self.costmap_tmp.shape: ', self.costmap_tmp.shape)
        # self.costmap_tmp.to_csv('~/amar_ws/costmap_data.csv', index=False, header=False)
        #'''

        print('starting C++ node')

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        #rospy.sleep(1)

        print('C++ node ended')

        # load command velocities
        self.cmd_vel_perturb = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('self.cmd_vel: ', self.cmd_vel_perturb)
        #print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)

        # load local plans
        self.local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')
        #print('self.local_plans: ', self.local_plans)
        #print('self.local_plans.shape: ', self.local_plans.shape)

        # load transformed plan
        self.transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('self.transformed_plan: ', self.transformed_plan)
        #print('self.transformed_plan.shape: ', self.transformed_plan.shape)

        # only needed for classifier_fn_image_plot() function
        self.sampled_instance = sampled_instance

        # plot perturbation of local costmap
        #self.classifier_fn_image_plot()

  
        import math

        # fill the list of local plan coordinates
        #print('self.local_plan_tmp: ', self.local_plan_tmp)
        #print('\nself.local_plan_tmp.shape[0]: ', self.local_plan_tmp.shape[0])
        local_plan_xs_orig = []
        local_plan_ys_orig = []
        for i in range(0, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                local_plan_xs_orig.append(x_temp)
                local_plan_ys_orig.append(y_temp)

        #print('\nlen(local_plan_xs): ', len(local_plan_xs_orig))
        #print('len(local_plan_ys): ', len(local_plan_ys_orig))

        # fill the list of transformed plan coordinates
        #print('\nself.transformed_plan.shape[0]: ', self.transformed_plan.shape[0])
        transformed_plan_xs = []
        transformed_plan_ys = []
        closest_to_robot_index = -100
        min_diff = 100
        for i in range(0, self.transformed_plan.shape[0]):
            x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)

                diff = math.sqrt( (transformed_plan_xs[-1]-local_plan_xs_orig[0])**2 + (transformed_plan_ys[-1]-local_plan_ys_orig[0])**2 )
                if diff < min_diff:
                    min_diff = diff
                    closest_to_robot_index = len(transformed_plan_xs) - 1

        #print('\nlen(transformed_plan_xs): ', len(transformed_plan_xs))
        #print('len(transformed_plan_ys): ', len(transformed_plan_ys))

        #print('\nclosest_to_robot_index_original: ', closest_to_robot_index)                    

        transformed_plan_xs = np.array(transformed_plan_xs)
        transformed_plan_ys = np.array(transformed_plan_ys)


        # a new way of deviation logic
        local_plan_gap_threshold = 11
        deviation_threshold = 12

        local_plan_original_gap = False
        local_plan_gaps = []
        diff = 0
        for j in range(0, len(local_plan_xs_orig) - 1):
            diff = math.sqrt( (local_plan_xs_orig[j]-local_plan_xs_orig[j+1])**2 + (local_plan_ys_orig[j]-local_plan_ys_orig[j+1])**2 )
            local_plan_gaps.append(diff)
        if max(local_plan_gaps) > local_plan_gap_threshold:
            local_plan_original_gap = True

        #print('\nmax(local_plan_original_gaps): ', max(local_plan_gaps))      
        
        if local_plan_original_gap == True:
            deviation_type = 'deviation'
        else:
            diff_x = 0
            diff_y = 0
            
            real_deviation = False
            for j in range( 0, len(local_plan_xs_orig)):
                #diffs = []
                deviation_local = True  
                for k in range(0, len(transformed_plan_xs)):
                    diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    #diffs.append(diff)
                    if diff <= deviation_threshold:
                        deviation_local = False
                        break
                #print('j = ', j)
                #print('min(diffs): ', min(diffs))    
                if deviation_local == True:
                    real_deviation = True
                    break
            
            if real_deviation == False:
                deviation_type = 'no_deviation'
            else:
                deviation_type = 'deviation'    
                    

        print('\nself.expID: ', self.expID)

        print('\ndeviation_type: ', deviation_type)

        '''
        print('\ndeviation_type: ', deviation_type)
        print('local_plan_gap: ', local_plan_original_gap)
        print('command velocities original - lin_x: ' + str(self.cmd_vel_original_tmp.iloc[0]) + ', ang_z: ' + str(self.cmd_vel_original_tmp.iloc[1]) + '\n')
        '''
        

        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)


        # fill in deviation dataframe
        for i in range(0, sampled_instance.shape[0]):
            # test if there is local plan
            local_plan_xs = []
            local_plan_ys = []
            local_plan_found = False
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    x_temp = int((self.local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((self.local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        local_plan_xs.append(x_temp)
                        local_plan_ys.append(y_temp)
                        local_plan_found = True
                
            if local_plan_found == True:
                # test if there is local plan gap
                diff = 0
                local_plan_gap = False
                local_plan_gaps = []
                for j in range(0, len(local_plan_xs) - 1):
                    diff = math.sqrt( (local_plan_xs[j]-local_plan_xs[j+1])**2 + (local_plan_ys[j]-local_plan_ys[j+1])**2 )
                    local_plan_gaps.append(diff)
                
                if max(local_plan_gaps) > local_plan_gap_threshold:
                    local_plan_gap = True
                
                if local_plan_gap == True:
                    if deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    elif deviation_type == 'deviation':
                        self.local_plan_deviation.iloc[i, 0] = 1.0
                else:
                    diff_x = 0
                    diff_y = 0
                    real_deviation = False
                    for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                        #diffs = []
                        deviation_local = True  
                        for k in range(0, len(transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            #diffs.append(diff)
                            if diff <= deviation_threshold:
                                deviation_local = False
                                break
                        #print('j = ', j)
                        #print('min(diffs): ', min(diffs))
                        if deviation_local == True:
                            real_deviation = True
                            break
                    
                    if deviation_type == 'no_deviation':
                        if real_deviation == False:
                            self.local_plan_deviation.iloc[i, 0] = 1.0
                        else:    
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                    elif deviation_type == 'deviation':
                        if real_deviation == False:
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        else:    
                            self.local_plan_deviation.iloc[i, 0] = 1.0                
            else:
                if deviation_type == 'no_deviation':         
                    self.local_plan_deviation.iloc[i, 0] = 0.0
                elif deviation_type == 'deviation':
                    self.local_plan_deviation.iloc[i, 0] = 1.0
            
            '''
            print('\ni: ', i)
            print('local plan found: ', local_plan_found)
            if local_plan_found == True:
                print('local plan length: ', len(local_plan_xs))
                print('local_plan_gap: ', local_plan_gap)
                print('max(local_plan_gaps): ', max(local_plan_gaps))
                if local_plan_gap == False:
                    print('deviation: ', real_deviation)
                    #print('minimal diff: ', min(diffs))
            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
            '''
        
        #print('self.local_plan_deviation: ', self.local_plan_deviation)
        #'''
        

        '''
        # an old way of deviation logic

        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        # fill in deviation dataframe
        for i in range(0, self.sampled_instance.shape[0]):
            local_plan_xs = []
            local_plan_ys = []
            local_plan_found = False
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    local_plan_found = True
                    local_plan_xs.append(self.local_plans.iloc[j, 0])
                    local_plan_ys.append(self.local_plans.iloc[j, 1])
            if local_plan_found == True:
                local_plan_xs = np.array(local_plan_xs)
                local_plan_ys = np.array(local_plan_ys)
                sum_x = 0
                sum_y = 0
                if local_plan_xs.shape <= transformed_plan_xs.shape:
                    for j in range(0, local_plan_xs.shape[0]):
                        sum_x = sum_x + (local_plan_xs[j] - transformed_plan_xs[j]) ** 2
                        sum_y = sum_y + (local_plan_ys[j] - transformed_plan_ys[j]) ** 2
                else:
                    for j in range(0, transformed_plan_xs.shape[0]):
                        sum_x = sum_x + (local_plan_xs[j] - transformed_plan_xs[j]) ** 2
                        sum_y = sum_y + (local_plan_ys[j] - transformed_plan_ys[j]) ** 2
                import math
                sum_final = math.sqrt(sum_x + sum_y)
                #print('i: ', i)
                #print('sum_final: ', sum_final)
                if sum_final < 4.0: # heuristics
                    self.local_plan_deviation.iloc[i, 0] = 0.0
        #print('self.local_plan_deviation: ', self.local_plan_deviation)
        '''

        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        
        # if more outputs wanted
        more_outputs = False
        if more_outputs == True:
            # classification
            stop_list = []
            linear_positive_list = []
            rotate_left_list = []
            rotate_right_list = []
            ahead_straight_list = []
            ahead_left_list = []
            ahead_right_list = []
            for i in range(0, self.cmd_vel_perturb.shape[0]):
                if abs(self.cmd_vel_perturb.iloc[i, 0]) < 0.01:
                    stop_list.append(1.0)
                else:
                    stop_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01:
                    linear_positive_list.append(1.0)
                else:
                    linear_positive_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 2] > 0.0:
                    rotate_left_list.append(1.0)
                else:
                    rotate_left_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 2] < 0.0:
                    rotate_right_list.append(1.0)
                else:
                    rotate_right_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) < 0.01:
                    ahead_straight_list.append(1.0)
                else:
                    ahead_straight_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.0:
                    ahead_left_list.append(1.0)
                else:
                    ahead_left_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] < 0.0:
                    ahead_right_list.append(1.0)
                else:
                    ahead_right_list.append(0.0)

            self.cmd_vel_perturb['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['stop'])
            self.cmd_vel_perturb['linear_positive'] = pd.DataFrame(np.array(linear_positive_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['linear_positive'])
            self.cmd_vel_perturb['rotate_left'] = pd.DataFrame(np.array(rotate_left_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left'])
            self.cmd_vel_perturb['rotate_right'] = pd.DataFrame(np.array(rotate_right_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right'])
            self.cmd_vel_perturb['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_straight'])
            self.cmd_vel_perturb['ahead_left'] = pd.DataFrame(np.array(ahead_left_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_left'])
            self.cmd_vel_perturb['ahead_right'] = pd.DataFrame(np.array(ahead_right_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_right'])

            '''
            print('self.cmd_vel_perturb: ', self.cmd_vel_perturb)
            print('self.local_plan_deviation: ', self.local_plan_deviation)
            print('stop_list: ', stop_list)
            print('linear_positive_list: ', linear_positive_list)
            print('rotate_left_list: ', rotate_left_list)
            print('rotate_right_list: ', rotate_right_list)
            print('ahead_straight_list: ', ahead_straight_list)
            print('ahead_left_list: ', ahead_left_list)
            print('ahead_right_list: ', ahead_right_list)
            '''

        print('classifier_fn_image ended')

        return np.array(self.cmd_vel_perturb.iloc[:, 3:])

    def classifier_fn_image_plot(self):
        '''
        # Visualise last 10 perturbations and last 100 perturbations separately
        self.perturbations_visualization = self.sampled_instance[0][:, :, 0]
        for i in range(1, 120):
            if i == 10:
                self.perturbations_visualization_final = self.perturbations_visualization
                self.perturbations_visualization = self.sampled_instance[i][:, :, 0]
            elif i % 10 == 0 & i != 10:
                self.perturbations_visualization_final = np.concatenate((self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
                self.perturbations_visualization = self.sampled_instance[i][:, :, 0]
            else:
                self.perturbations_visualization = np.concatenate((self.perturbations_visualization, self.sampled_instance[i][:, :, 0]), axis=1)
        self.perturbations_visualization_final = np.concatenate((self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
        '''

        '''
        # Save perturbations as .csv file
        for i in range(0, self.sampled_instance.shape[0]):
            pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('~/amar_ws/perturbation_' + str(i) + '.csv', index=False, header=False)
        '''


        #'''
        # indices of transformed plan's poses in local costmap
        self.transformed_plan_x_list = []
        self.transformed_plan_y_list = []
        for j in range(0, self.transformed_plan.shape[0]):
            self.transformed_plan_x_list.append(int((self.transformed_plan.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.transformed_plan_y_list.append(int((self.transformed_plan.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        # print('i: ', i)
        # print('self.transformed_plan_x_list.size(): ', len(self.transformed_plan_x_list))
        # print('self.transformed_plan_y_list.size(): ', len(self.transformed_plan_y_list))

        # plot every perturbation
        for i in range(0, self.sampled_instance.shape[0]):

            # save current perturbation as .csv file
            #pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('perturbation_' + str(i) + '.csv', index=False, header=False)

            # plot perturbed local costmap
            plt.imshow(self.sampled_instance[i][:, :, 0])

            # indices of local plan's poses in local costmap
            self.local_plan_x_list = []
            self.local_plan_y_list = []
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    index_x = int((self.local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    index_y = int((self.local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
                    self.local_plan_x_list.append(index_x)
                    self.local_plan_y_list.append(index_y)
                    '''
                    [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans.iloc[j, 2], self.local_plans.iloc[j, 3])
                    yaw_x = math.cos(yaw)
                    yaw_y = math.sin(yaw)
                    plt.quiver(index_x, index_y, yaw_x, yaw_y, color='white')
                    '''
            # print('i: ', i)
            # print('self.local_plan_x_list.size(): ', len(self.local_plan_x_list))
            # print('self.local_plan_y_list.size(): ', len(self.local_plan_y_list))

            # plot transformed plan
            plt.scatter(self.transformed_plan_x_list, self.transformed_plan_y_list, c='blue', marker='x')

            # plot footprint
            plt.scatter(self.footprint_x_list, self.footprint_y_list, c='green', marker='x')

            '''
            # plot footprints for first five points of local plan
            # indices of local plan's poses in local costmap
            self.footprint_local_plan_x_list = []
            self.footprint_local_plan_y_list = []
            self.footprint_local_plan_x_list_angle = []
            self.footprint_local_plan_y_list_angle = []
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    for k in range(6, 7):

                        [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans.iloc[j + k, 2], self.local_plans.iloc[j + k, 3])
                        sin_th = math.sin(yaw)
                        cos_th = math.cos(yaw)

                        for l in range(0, self.footprint_tmp.shape[0]):
                            x_new = self.footprint_tmp.iloc[l, 0] + (self.local_plans.iloc[j + k, 0] - self.odom_x)
                            y_new = self.footprint_tmp.iloc[l, 1] + (self.local_plans.iloc[j + k, 1] - self.odom_y)
                            self.footprint_local_plan_x_list.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                            self.footprint_local_plan_y_list.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))

                            x_new = self.local_plans.iloc[j + k, 0] + (self.footprint_tmp.iloc[l, 0] - self.odom_x) * sin_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * cos_th
                            y_new = self.local_plans.iloc[j + k, 1] - (self.footprint_tmp.iloc[l, 0] - self.odom_x) * cos_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * sin_th
                            self.footprint_local_plan_x_list_angle.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                            self.footprint_local_plan_y_list_angle.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))
                    break
            #print('self.footprint_local_plan_x_list: ', self.footprint_local_plan_x_list)
            #print('self.footprint_local_plan_y_list: ', self.footprint_local_plan_y_list)
            # plot footprints
            plt.scatter(self.footprint_local_plan_x_list, self.footprint_local_plan_y_list, c='green', marker='x')
            plt.scatter(self.footprint_local_plan_x_list_angle, self.footprint_local_plan_y_list_angle, c='white', marker='x')
            '''

            # plot local plan
            plt.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')

            # plot local plan last point
            #if len(self.local_plan_x_list) != 0:
            #    plt.scatter([self.local_plan_x_list[-1]], [self.local_plan_y_list[-1]], c='black', marker='x')

            # plot robot's location and orientation
            plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

            # plot command velocities as text
            plt.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_perturb.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_perturb.iloc[i, 2], 2)))

            # save figure
            plt.savefig('perturbation_' + str(i) + '.png')
            plt.clf()
        #'''


    # plot explanation picture and segments
    def plotExplanationMinimal(self):
        path_core = os.getcwd()

        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        # plot segments with centroids and labels/weights
        #print('segments_1.shape: ', segments_1.shape)
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1

        # Save segments with nice numbering as a picture
        fig.savefig(path_core + '/weighted_segments.png')
        fig.clf()


        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        #ax = plt.Axes(fig, [0., 0., 1., 0.95])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        
        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)
        
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
    
        # Get coordinates of the global plan in the local costmap
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
                    
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # plot robots' local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
                
        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        #self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        #self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        #[self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        #self.yaw_odom_x = math.cos(self.yaw_odom)
        #self.yaw_odom_y = math.sin(self.yaw_odom)
        
        # plot robots' location and orientation
        #print('self.x_odom_index: ', self.x_odom_index)
        #print('self.y_odom_index: ', self.y_odom_index)
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        
        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
        #    round(self.cmd_vel_original_tmp.iloc[1], 2)))
        
        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask)
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries, aspect='auto')  # , aspect='auto')
        fig.savefig(path_core + '/explanation.png', transparent=False)
        fig.clf()
        #fig.close()

    # plot explanation picture and segments flipped
    def plotExplanationMinimalFlipped(self):
        path_core = os.getcwd()    

        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        # plot segments with centroids and labels/weights
        #print('segments_1.shape: ', segments_1.shape)
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.matrixFlip(segments_1, 'h').astype('uint8'), aspect='auto')
        
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(160 - centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(160 - centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1

        # Save segments with nice numbering as a picture
        fig.savefig(path_core + '/weighted_segments_flipped.png')
        fig.clf()


        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(160 - int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = 160 - int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)
        
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # plot robots' local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        
        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        #self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        #self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        #[self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        #yaw_sign = math.copysign(1, self.yaw_odom)
        #self.yaw_odom = yaw_sign * (math.pi - abs(self.yaw_odom)) 
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        #self.yaw_odom_x = math.cos(self.yaw_odom)
        #self.yaw_odom_y = math.sin(self.yaw_odom)

        # plot robots' location and orientation
        #print('self.x_odom_index: ', self.x_odom_index)
        #print('self.y_odom_index: ', self.y_odom_index)
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        
        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
        #    round(self.cmd_vel_original_tmp.iloc[1], 2)))
        
        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries_flipped.astype('float64'), aspect='auto')  # , aspect='auto')
        fig.savefig(path_core + '/explanation_flipped.png', transparent=False)
        fig.clf()
        #fig.close()

    # plot explanation picture, segments, global map, mask and temp
    def plotExplanation(self):
        print('plotExplanation starts')

        path_core = os.getcwd()

        # plot local costmap
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
    
        # Get coordinates of the global plan in the local costmap
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(y_temp)
                    
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        # print('self.local_plan_tmp.shape: ', self.local_plan_tmp.shape)
        for i in range(0, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 160 and 0 <= y_temp <= 160:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')

        # plot robot odometry location
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        # robot's odometry orientation
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
        
        # plot local costmap
        ax.imshow(self.image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/local_costmap.png')
        fig.clf()


        # plot global map
        fig = plt.figure(frameon=False)
        w = 4.0
        h = 6.0
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # global plan from teb algorithm
        self.global_plan_x_list = []
        self.global_plan_y_list = []
        for i in range(19, self.global_plan_tmp.shape[0], 20):
            self.global_plan_x_list.append(
                int((self.global_plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.global_plan_y_list.append(
                int((self.global_plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.global_plan_x_list, self.global_plan_y_list, c='blue', marker='>')

        # plan from global planner
        self.plan_map_x_list = []
        self.plan_map_y_list = []
        for i in range(19, self.plan_tmp.shape[0], 20):
            self.plan_map_x_list.append(int((self.plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.plan_map_y_list.append(int((self.plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.plan_map_x_list, self.plan_map_y_list, c='red', marker='<')

        # indices of robot's odometry location in map
        self.mapIndex_x_amcl = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        # print('self.mapIndex_x_amcl: ', self.mapIndex_x_amcl)
        self.mapIndex_y_amcl = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        # print('self.mapIndex_y_amcl: ', self.mapIndex_y_amcl)

        # indices of robot's amcl location in map in a list - suitable for plotting
        self.x_amcl_index = [self.mapIndex_x_amcl]
        self.y_amcl_index = [self.mapIndex_y_amcl]
        ax.scatter(self.x_amcl_index, self.y_amcl_index, c='black', marker='o')

        #self.yaw_amcl_x = math.cos(self.yaw_amcl)
        #self.yaw_amcl_y = math.sin(self.yaw_amcl)
        #ax.quiver(self.x_amcl_index, self.y_amcl_index, self.yaw_amcl_x, self.yaw_amcl_y, color='black')

        # plot robot's location in the map
        x_map = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        y_map = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        #ax.scatter(x_map, y_map, c='red', marker='o')

        # plot map, fill -1 with 100
        map_tmp = self.map_data
        for i in range(0, map_tmp.shape[0]):
            for j in range(0, map_tmp.shape[1]):
                if map_tmp.iloc[i, j] == -1:
                    map_tmp.iloc[i, j] = 100
        ax.imshow(map_tmp.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/global_map.png')
        fig.clf()

        # plot image_temp
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.temp_img, aspect='auto')
        fig.savefig(path_core + '/temp_img.png')
        fig.clf()

        # plot mask
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.mask.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/mask.png')
        fig.clf()

        # plot explanation - srediti nekad granice
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask)
        ax.imshow(marked_boundaries, aspect='auto')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
        fig.savefig(path_core + '/explanation.png')
        fig.clf()

        print('plotExplanation ends')

    # plot explanation picture, segments, global map, mask and temp all flipped
    def plotExplanationFlipped(self):
        print('plotExplanationFlipped starts')

        path_core = os.getcwd()

        # make a deepcopy of local costmap and flip it
        local_costmap = copy.deepcopy(self.image)
        local_costmap_flipped = self.matrixFlip(local_costmap, 'h')

        # plot flipped local costmap
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        #print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        #print('r_array: ', r_array)
        #print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        #print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array([self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        #'''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(y_temp)
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = 160 - int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 150 and 0 <= x_temp <= 150:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')

        # plot robot odometry location
        self.x_odom_index = [160 - self.localCostmapIndex_x_odom]
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')

        # find yaw angles projections on x and y axes and save them to class variables
        #yaw_sign = math.copysign(1, self.yaw_odom)
        #self.yaw_odom = -1 * yaw_sign * (math.pi - abs(self.yaw_odom))
        #self.yaw_odom_x = math.cos(self.yaw_odom)
        #self.yaw_odom_y = math.sin(self.yaw_odom)

        # robot's odometry orientation
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        ax.imshow(local_costmap_flipped, aspect='auto')
        fig.savefig(path_core + '/local_costmap_flipped.png')
        fig.clf()


        # plot flipped map
        fig = plt.figure(frameon=False)
        w = 4.0
        h = 6.0
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # indices of robot's odometry location in map
        self.mapIndex_x_amcl = 160 - int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        # print('self.mapIndex_x_amcl: ', self.mapIndex_x_amcl)
        self.mapIndex_y_amcl = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        # print('self.mapIndex_y_amcl: ', self.mapIndex_y_amcl)

        # plan from global planner
        self.plan_map_x_list = []
        self.plan_map_y_list = []
        for i in range(19, self.plan_tmp.shape[0], 20):
            self.plan_map_x_list.append(160 - int((self.plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.plan_map_y_list.append(int((self.plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.plan_map_x_list, self.plan_map_y_list, c='blue', marker='<')

        # global plan from teb algorithm
        self.global_plan_x_list = []
        self.global_plan_y_list = []
        for i in range(19, self.global_plan_tmp.shape[0], 20):
            self.global_plan_x_list.append(
                160 - int((self.global_plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.global_plan_y_list.append(
                int((self.global_plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.global_plan_x_list, self.global_plan_y_list, c='red', marker='>')

        # indices of robot's amcl location in map in a list - suitable for plotting
        self.x_amcl_index = [self.mapIndex_x_amcl]
        self.y_amcl_index = [self.mapIndex_y_amcl]
        ax.scatter(self.x_amcl_index, self.y_amcl_index, c='black', marker='o')

        # find yaw angles projections on x and y axes and save them to class variables
        #yaw_sign = math.copysign(1, self.yaw_amcl)
        #self.yaw_amcl = -1 * yaw_sign * (math.pi - abs(self.yaw_amcl))
        #self.yaw_amcl_x = math.cos(self.yaw_amcl)
        #self.yaw_amcl_y = math.sin(self.yaw_amcl)
        #ax.quiver(self.x_amcl_index, self.y_amcl_index, self.yaw_amcl_x, self.yaw_amcl_y, color='black')

        # plot robot's location in the map
        x_map = 160 - int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        y_map = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        #ax.scatter(x_map, y_map, c='red', marker='o')

        # plot map, fill -1 with 100
        map_tmp = self.map_data
        for i in range(0, map_tmp.shape[0]):
            for j in range(0, map_tmp.shape[1]):
                if map_tmp.iloc[i, j] == -1:
                    map_tmp.iloc[i, j] = 100
        map_tmp_flipped = self.matrixFlip(map_tmp, 'h')
        ax.imshow(map_tmp_flipped, aspect='auto')
        fig.savefig('global_map_flipped.png')
        fig.clf()

        # plot image_temp
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        temp_img_flipped = self.matrixFlip(self.temp_img, 'h')
        ax.imshow(temp_img_flipped, aspect='auto')
        fig.savefig(path_core + '/temp_img_flipped.png')
        fig.clf()
        #pd.DataFrame(self.temp_img_flipped[:,:,0]).to_csv('~/amar_ws/temp_img_flipped.csv', index=False, header=False)

        # plot mask
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        mask_flipped = self.matrixFlip(self.mask, 'h')
        plt.imshow(mask_flipped, aspect='auto')
        plt.savefig(path_core + '/mask_flipped.png')
        plt.clf()
        #pd.DataFrame(self.mask_flipped[:,:,0]).to_csv('~/amar_ws/mask_flipped.csv', index=False, header=False)

        # plot explanation
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
     
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # plot robots' location, orientation and local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
          
        #'''
        # plot explanation
        #print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_original_tmp.iloc[1], 2)))
        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries_flipped) #, aspect='auto')
        fig.savefig(path_core + '/explanation_flipped.png', transparent=False)
        fig.clf()

        print('plotExplanationFlipped ends')


        '''
        # plot for the HARL Workshop 2021 paper
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=(37.00, 52.00), width=30, height=70, edgecolor='orange', fc='None', lw=2, alpha=80.0)
        ax.add_patch(ellipse)
        from matplotlib.patches import Circle
        circle = Circle(xy=(160 - self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom),
                        radius=round(0.275 / self.localCostmapResolution) - 4, edgecolor='blue', fc='blue', lw=2,
                        alpha=100.0)
        ax.add_patch(circle)
        ax.text(35, 72, 'H', fontsize=12, fontweight=700)
        ax.text(34, 32, 'PS', fontsize=12, fontweight=700)
        ax.text(34, 10, 'FV', fontsize=12, fontweight=700)
        ax.text(160 - self.localCostmapIndex_x_odom - 1, self.localCostmapIndex_y_odom + 2, 'R', fontsize=12,
                fontweight=700)
        # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content        
        '''


    def matrixFlip(self, m, d):
        myl = np.array(m)
        if d == 'v':
            return np.flip(myl, axis=0)
        elif d == 'h':
            return np.flip(myl, axis=1)

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

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        #qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        #qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        
        return [qx, qy, qz, qw]

    def printImportantInformation(self):
        # print important information

        if self.explanation_mode == 'image':
            #'''
            print('self.explanation_mode: ', self.explanation_mode)
            print('self.expID: ', self.expID)
            print('self.offset: ', self.offset)
            print('\n')
            #'''
        else:
            #'''
            print('self.explanation_mode: ', self.explanation_mode)
            print('self.tabular_mode: ', self.tabular_mode)
            print('self.expID: ', self.expID)
            print('self.num_samples: ', self.num_samples)
            print('self.output_class_name: ', self.output_class_name)
            print('self.offset: ', self.offset)
            print('\n')
            #'''    

    def PFP2PO(self):
        # Turn point free space (that is surrounded by obstacles) to point obstacle
        # Helps in some cases
        # Because constructor is called only once, this is not a big computational burden (for now)
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

    def limeImageSaveData(self):
        # Saving data to .csv files for C++ node - local navigation planner
        # Save footprint instance to a file
        self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
        self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
        self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

        # Save local plan instance to a file
        self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
        self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
        self.local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan (from global planner) instance to a file
        self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
        self.plan_tmp = self.plan_tmp.iloc[:, 1:]
        self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save global plan instance to a file
        self.global_plan_tmp = self.global_plan.loc[self.global_plan['ID'] == self.index + self.offset]
        self.global_plan_tmp = self.global_plan_tmp.iloc[:, 1:]
        self.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False,
                                    header=False)

        # Save costmap_info instance to file
        self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
        self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
        self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
        self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False,
                                     header=False)

        # Save amcl_pose instance to file
        self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
        self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
        self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
        self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map instance to file
        self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
        self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
        self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False,
                                    header=False)

        # Save tf_map_odom instance to file
        self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
        self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
        self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False,
                                    header=False)

        # Save odometry instance to file
        self.odom_tmp = self.odom.iloc[self.index, :]
        self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
        self.odom_tmp = self.odom_tmp.iloc[:, 2:]
        self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        # Take original command speed
        self.cmd_vel_original_tmp = self.cmd_vel_original.iloc[self.index, :]
        #self.cmd_vel_original_tmp = pd.DataFrame(self.cmd_vel_original_tmp).transpose()
        #self.cmd_vel_original_tmp = self.cmd_vel_original_tmp.iloc[:, 2:]

        # save costmap info to class variables
        self.localCostmapOriginX = self.costmap_info_tmp.iloc[0, 3]
        #print('self.localCostmapOriginX: ', self.localCostmapOriginX)
        self.localCostmapOriginY = self.costmap_info_tmp.iloc[0, 4]
        #print('self.localCostmapOriginY: ', self.localCostmapOriginY)
        self.localCostmapResolution = self.costmap_info_tmp.iloc[0, 0]
        #print('self.localCostmapResolution: ', self.localCostmapResolution)
        self.localCostmapHeight = self.costmap_info_tmp.iloc[0, 2]
        #print('self.localCostmapHeight: ', self.localCostmapHeight)
        self.localCostmapWidth = self.costmap_info_tmp.iloc[0, 1]
        #print('self.localCostmapWidth: ', self.localCostmapWidth)

        # save robot odometry location to class variables
        self.odom_x = self.odom_tmp.iloc[0, 0]
        # print('self.odom_x: ', self.odom_x)
        self.odom_y = self.odom_tmp.iloc[0, 1]
        # print('self.odom_y: ', self.odom_y)

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2]
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        #print('roll_odom: ', roll_odom)
        #print('pitch_odom: ', pitch_odom)
        #print('self.yaw_odom: ', self.yaw_odom)
        #[qx, qy, qz, qw] = self.euler_to_quaternion(self.yaw_odom, pitch_odom, roll_odom)
        
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)

        # save indices of footprint's poses in local costmap to class variables
        self.footprint_x_list = []
        self.footprint_y_list = []
        for j in range(0, self.footprint_tmp.shape[0]):
            # print(str(self.footprint_tmp.iloc[j, 0]) + '  ' + str(self.footprint_tmp.iloc[j, 1]))
            self.footprint_x_list.append(
                int((self.footprint_tmp.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.footprint_y_list.append(
                int((self.footprint_tmp.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # map info
        self.mapOriginX = self.map_info.iloc[0, 4]
        # print('self.mapOriginX: ', self.mapOriginX)
        self.mapOriginY = self.map_info.iloc[0, 5]
        # print('self.mapOriginY: ', self.mapOriginY)
        self.mapResolution = self.map_info.iloc[0, 1]
        # print('self.mapResolution: ', self.mapResolution)
        self.mapHeight = self.map_info.iloc[0, 3]
        # print('self.mapHeight: ', self.mapHeight)
        self.mapWidth = self.map_info.iloc[0, 2]
        # print('self.mapWidth: ', self.mapWidth)

        # robot amcl location
        self.amcl_x = self.amcl_pose_tmp.iloc[0, 0]
        # print('self.amcl_x: ', self.amcl_x)
        self.amcl_y = self.amcl_pose_tmp.iloc[0, 1]
        # print('self.amcl_y: ', self.amcl_y)

        # robot amcl orientation
        self.amcl_z = self.amcl_pose_tmp.iloc[0, 2]
        self.amcl_w = self.amcl_pose_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_amcl, pitch_amcl, roll_amcl] = self.quaternion_to_euler(0.0, 0.0, self.amcl_z, self.amcl_w)
        #print('roll_amcl: ', roll_amcl)
        #print('pitch_amcl: ', pitch_amcl)
        #print('yaw_amcl: ', self.yaw_amcl)
    
    def inflatedToFree(self):
        #'''
        # Turn inflated area to free space and 100s to 99s
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                if 99 > self.image[i, j] > 0:
                    self.image[i, j] = 0
                elif self.image[i, j] == 100:
                    self.image[i, j] = 99
        #'''


    def testSegmentation(self, expID):

        print('Test segmentation function beginning')

        # Make image a np.array deepcopy of local_costmap_original
        img_ = copy.deepcopy(self.image)

        '''
        # Save local costmap as gray image
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_, aspect='auto')
        fig.savefig('local_costmap_gray_test_segmentation.png')
        fig.clf()
        '''

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        '''
        # Save local costmap as rgb image
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgb, aspect='auto')
        fig.savefig('local_costmap_rgb_test_segmentation.png')
        fig.clf()
        '''

        # Superpixel segmentation with skimage functions

        # felzenszwalb
        #segments = felzenszwalb(rgb, scale=100, sigma=5, min_size=30, multichannel=True)
        #segments = felzenszwalb(rgb, scale=1, sigma=0.8, min_size=20, multichannel=True)  # default

        # quickshift
        #segments = quickshift(rgb, ratio=0.0001, kernel_size=8, max_dist=10, return_tree=False, sigma=0.0, convert2lab=True, random_seed=42)
        #segments = quickshift(rgb, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, random_seed=42) # default

        # slic
        #segments = slic(rgb, n_segments=6, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=5, slic_zero=False, start_label=None, mask=None)
        #segments = slic(rgb, n_segments=100, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=None, mask=None) # default

        # Turn segments gray image to rgb image
        #segments_rgb = gray2rgb(segments)

        # Generate segments - superpixels with my slic function
        segments = self.mySlicTest(rgb)

        # Save segments to .csv file
        #pd.DataFrame(segments).to_csv('~/amar_ws/segments_segmentation_test.csv', index=False, header=False)

        print('Test segmentation function ending')

    def mySlicTest(self, img_rgb):

        print('mySlic for testSegmentation starts')

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = img_rgb[:, :, 0]
        # Save picture for segmenting
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='auto')
        fig.savefig('local_costmap_test_segmentation.png')
        fig.clf()

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # plot segments_1 with centroids and labels
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_1 as a picture
        fig.savefig('segments_1_test_segmentation.png')
        fig.clf()
        # find segments_unique_1
        segments_unique_1 = np.unique(segments_1)
        print('segments_unique_1: ', segments_unique_1)
        print('segments_unique_1.shape: ', segments_unique_1.shape)


        # Find segments_2
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
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
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_2 as a picture
        ax.imshow(segments_2, aspect='auto')
        fig.savefig('segments_2_test_segmentation.png')
        fig.clf()
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        
        # plot segments with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments as a picture before nice segment numbering
        fig.savefig('segments_beforeNiceNumbering_test_segmentation.png')
        fig.clf()
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

        # plot segments with nice numbering with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments as a picture before nice segment numbering
        fig.savefig('segments_afterNiceNumbering_test_segmentation.png')
        fig.clf()
        
        # plot segments with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.imshow(self.matrixFlip(segments_1, 'h'), aspect='auto')
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig('segments_with_explanation_weights_test_segmentation.png')
        fig.clf()

        # print explanation
        #print('self.exp: ', self.exp)
        #print('len(self.exp): ', len(self.exp))

        print('mySlic for testSegmentation ends')

        return segments_1

    



    def plotMinimalDataset(self, iteration_ID):
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        
        # plot explanation
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
        #    round(self.cmd_vel_original_tmp.iloc[1], 2)))

        # plot robots' location, orientation, global and local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
        
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        
        ax.imshow(marked_boundaries.astype('float64'), aspect='auto')
        fig.savefig(str(iteration_ID) + '_output' + '.png', transparent=False)
        fig.clf()


        # plot costmap
        fig = plt.figure(figsize=[1.6, 1.6], frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        fig.set_size_inches(1.6, 1.6)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Plot coordinates of the global plan in the local costmap
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        # plot robots' location, orientation and local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        ax.imshow(self.image.astype('uint8'), aspect='auto')        
        fig.savefig(str(iteration_ID) + '_input' + '.png')
        fig.clf()

        
    def plotMinimalFlippedDataset(self, iteration_ID):
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = 160 - int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 160 and 0 <= y_temp <= 160:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)

                with open('local_plan_coordinates.csv', "a") as myfile:
                     myfile.write(str(iteration_ID) + ',' + str(self.local_plan_tmp.iloc[i, 0]) + ',' + str(self.local_plan_tmp.iloc[i, 1]) + '\n')


        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(y_temp)

                with open('global_plan_coordinates.csv', "a") as myfile:
                    myfile.write(str(iteration_ID) + ',' + str(plan_tmp_tmp.iloc[i, 0]) + ',' + str(plan_tmp_tmp.iloc[i, 1]) + '\n')

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = 160 - int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        #znak = 1
        #if self.odom_z < 0:
        #    znak = -1
        #self.odom_z = znak * (1 - abs(self.odom_z))        
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        yaw_sign = math.copysign(1, self.yaw_odom)
        self.yaw_odom = -1 * yaw_sign * (math.pi - abs(self.yaw_odom))
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        
        # plot explanation
        fig = plt.figure(frameon=False)
        w = 6.4
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
        #    round(self.cmd_vel_original_tmp.iloc[1], 2)))

        # plot robots' location, orientation, global and local plan
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
        
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        
        ax.imshow(marked_boundaries_flipped.astype('float64'), aspect='auto')
        fig.savefig(str(iteration_ID) + '_output' + '.png', transparent=False)
        fig.clf()


        # plot costmap
        w = 6.4
        h = 4.8
        fig = plt.figure(figsize=[w, h], frameon=False)
        fig.set_size_inches(w, h)
        #fig.set_size_inches(1.6, 1.6)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Plot coordinates of the global plan in the local costmap
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        # plot robots' location, orientation and local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        ax.imshow(self.matrixFlip(self.image, 'h').astype('uint8'), aspect='auto')        
        fig.savefig(str(iteration_ID) + '_input' + '.png')
        fig.clf()

        with open('costmap_data.csv', "a") as myfile:
                #myfile.write('picture_ID,width,heigth,origin_x,origin_y,resolution\n')
                myfile.write(str(iteration_ID) + ',' + str(self.localCostmapWidth) + ',' + str(self.localCostmapHeight) + ',' + str(self.localCostmapOriginX) + ',' + str(self.localCostmapOriginY) + ',' + str(self.localCostmapResolution) + '\n')

        with open('robot_coordinates.csv', "a") as myfile:
                #myfile.write('picture_ID,position_x,position_y\n')
                myfile.write(str(iteration_ID) + ',' + str(self.odom_x) + ',' + str(self.odom_y) + '\n')        

    
    def explain_instance_dataset(self, expID, iteration_ID):
        print('explain_instance_dataset function starting\n')

        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.expID = expID
            self.index = expID

            self.printImportantInformation()

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            self.local_costmap_original = self.costmap_data.iloc[
                                          (self.index) * 160:(self.index + 1) * self.costmap_size, :]

            '''
            # If a custom costmap is used - TO-DO: razdvojiti na custom i non-custom map 
            self.local_costmap_original.to_csv('~/amar_ws/costmapToChange.csv', index=False, header=True)
            self.local_costmap_original = pd.read_csv('~/amar_ws/costmapToChange.csv')
            '''

            # Make image a np.array deepcopy of local_costmap_original
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            # '''
            # Turn inflated area to free space and 100s to 99s
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):
                    if 99 > self.image[i, j] > 0:
                        self.image[i, j] = 0
                    elif self.image[i, j] == 100:
                        self.image[i, j] = 99
            # '''

            # Turn point free space (that is surrounded by obstacles) to point obstacle
            # self.PFP2PO()

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0

            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveData()

            # Use new variable in the algorithm
            img = copy.deepcopy(self.image)

            # my custom segmentation func
            segm_fn = 'custom_segmentation'

            self.explanation = self.explainer.explain_instance(img, self.classifier_fn_image,
                                                               hide_color=perturb_hide_color_value,
                                                               num_samples=self.num_samples,
                                                               batch_size=1024, segmentation_fn=segm_fn,
                                                               top_labels=10)
            # print('self.explanation: ', self.explanation)

            self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=False,
                                                                                     negative_only=False,
                                                                                     num_features=100,
                                                                                     hide_rest=False,
                                                                                     min_weight=0.1)  # min_weight=0.1 - default

            #self.plotMinimalDataset(iteration_ID)
            self.plotMinimalFlippedDataset(iteration_ID)

    
        '''
        # Turn gray image to rgb image
        img_rgb = gray2rgb(img)

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_1
        #segments_unique_1 = np.unique(segments_1)
        #print('segments_unique_1: ', segments_unique_1)
        #print('segments_unique_1.shape: ', segments_unique_1.shape)

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        #print('segments_unique_2: ', segments_unique_2)
        #print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Creating segments using segments_1 and segments_2
        # Add/Sum segments_1 and segments_2
        #''#
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #''#
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        return 2 ** segments_unique.shape[0], segments_unique.shape[0]
        '''


    def calculateNumOfSamples(self, img):
        # Turn gray image to rgb image
        img_rgb = gray2rgb(img)

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_1
        #segments_unique_1 = np.unique(segments_1)
        #print('segments_unique_1: ', segments_unique_1)
        #print('segments_unique_1.shape: ', segments_unique_1.shape)

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        #print('segments_unique_2: ', segments_unique_2)
        #print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Creating segments using segments_1 and segments_2
        # '''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        # '''
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        return 2 ** segments_unique.shape[0], segments_unique.shape[0]

    def classifier_fn_image_evaluation(self, sampled_instance):

        print('classifier_fn_image started')

        # sampled_instance info
        # print('sampled_instance: ', sampled_instance)
        # print('sampled_instance.shape: ', sampled_instance.shape)

        # '''
        # I will use channel 0 from sampled_instance as actual perturbed data
        # Perturbed pixel intensity is perturb_hide_color_value
        # Convert perturbed free space to obstacle (99), and perturbed obstacles to free space (0) in all perturbations
        for i in range(0, sampled_instance.shape[0]):
            for j in range(0, sampled_instance[i].shape[0]):
                for k in range(0, sampled_instance[i].shape[1]):
                    if sampled_instance[i][j, k, 0] == perturb_hide_color_value:
                        if self.image[j, k] == 0:
                            sampled_instance[i][j, k, 0] = 99
                            # print('free space')
                        elif self.image[j, k] == 99:
                            sampled_instance[i][j, k, 0] = 0
                            # print('obstacle')
        # '''

        # '''
        # Save perturbed costmap_data to file for c++ local planner node
        # sampled_instance = sampled_instance.astype(int)
        self.costmap_tmp = pd.DataFrame(sampled_instance[0][:, :, 0])
        for i in range(1, sampled_instance.shape[0]):
            self.costmap_tmp = pd.concat([self.costmap_tmp, pd.DataFrame(sampled_instance[i][:, :, 0])],
                                         join='outer', axis=0, sort=False)
        self.costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False,
                                header=False)
        # print('self.costmap_tmp.shape: ', self.costmap_tmp.shape)
        # self.costmap_tmp.to_csv('~/amar_ws/costmap_data.csv', index=False, header=False)
        # '''

        start = time.time()

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        # print('perturb_node_image finished from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        end = time.time()
        planner_time = end - start

        # load command velocities
        self.cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        # print('cmd_vel: ', cmd_vel)
        # print('cmd_vel.shape: ', cmd_vel.shape)

        # load local plans
        self.local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')
        # print('local_plans: ', local_plans)
        # print('local_plans.shape: ', local_plans.shape)

        # load transformed plan
        self.transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        # print('transformed_plan: ', transformed_plan)
        # print('transformed_plan.shape: ', transformed_plan.shape)

        self.sampled_instance = sampled_instance

        # self.classifier_fn_image_plot()

        # new output - deviation of the local plan compared to the global plan
        self.local_plan_deviation = d = pd.DataFrame(1.0, index=np.arange(self.sampled_instance.shape[0]),
                                                     columns=['deviate'])
        # print(self.local_plan_deviation)

        transformed_plan_xs = []
        transformed_plan_ys = []
        for i in range(0, self.transformed_plan.shape[0]):
            transformed_plan_xs.append(self.transformed_plan.iloc[i, 0])
            transformed_plan_ys.append(self.transformed_plan.iloc[i, 1])
        transformed_plan_xs = np.array(transformed_plan_xs)
        transformed_plan_ys = np.array(transformed_plan_ys)

        for i in range(0, self.sampled_instance.shape[0]):
            local_plan_xs = []
            local_plan_ys = []
            local_plan_found = False
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    local_plan_found = True
                    local_plan_xs.append(self.local_plans.iloc[j, 0])
                    local_plan_ys.append(self.local_plans.iloc[j, 1])
            if local_plan_found == True:
                local_plan_xs = np.array(local_plan_xs)
                local_plan_ys = np.array(local_plan_ys)
                sum_x = 0
                sum_y = 0
                if local_plan_xs.shape <= transformed_plan_xs.shape:
                    for j in range(0, local_plan_xs.shape[0]):
                        sum_x = sum_x + (local_plan_xs[j] - transformed_plan_xs[j]) ** 2
                        sum_y = sum_y + (local_plan_ys[j] - transformed_plan_ys[j]) ** 2
                else:
                    for j in range(0, transformed_plan_xs.shape[0]):
                        sum_x = sum_x + (local_plan_xs[j] - transformed_plan_xs[j]) ** 2
                        sum_y = sum_y + (local_plan_ys[j] - transformed_plan_ys[j]) ** 2
                import math
                sum_final = math.sqrt(sum_x + sum_y)
                # print('i: ', i)
                # print('sum_final: ', sum_final)
                if sum_final < 4.0:
                    self.local_plan_deviation.iloc[i, 0] = 0.0

        # print(self.local_plan_deviation)

        # classification

        stop_list = []
        linear_positive_list = []
        rotate_left_list = []
        rotate_right_list = []
        ahead_straight_list = []
        ahead_left_list = []
        ahead_right_list = []
        for i in range(0, self.cmd_vel.shape[0]):
            if abs(self.cmd_vel.iloc[i, 0]) < 0.01:
                stop_list.append(1.0)
            else:
                stop_list.append(0.0)

            if self.cmd_vel.iloc[i, 0] > 0.01:
                linear_positive_list.append(1.0)
            else:
                linear_positive_list.append(0.0)

            if self.cmd_vel.iloc[i, 2] > 0.0:
                rotate_left_list.append(1.0)
            else:
                rotate_left_list.append(0.0)

            if self.cmd_vel.iloc[i, 2] < 0.0:
                rotate_right_list.append(1.0)
            else:
                rotate_right_list.append(0.0)

            if self.cmd_vel.iloc[i, 0] > 0.01 and abs(self.cmd_vel.iloc[i, 2]) < 0.01:
                ahead_straight_list.append(1.0)
            else:
                ahead_straight_list.append(0.0)

            if self.cmd_vel.iloc[i, 0] > 0.01 and self.cmd_vel.iloc[i, 2] > 0.0:
                ahead_left_list.append(1.0)
            else:
                ahead_left_list.append(0.0)

            if self.cmd_vel.iloc[i, 0] > 0.01 and self.cmd_vel.iloc[i, 2] < 0.0:
                ahead_right_list.append(1.0)
            else:
                ahead_right_list.append(0.0)

        self.cmd_vel['deviate'] = self.local_plan_deviation
        self.cmd_vel['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel.shape[0]),
                                            columns=['stop'])
        self.cmd_vel['linear_positive'] = pd.DataFrame(np.array(linear_positive_list),
                                                       index=np.arange(self.cmd_vel.shape[0]),
                                                       columns=['linear_positive'])
        self.cmd_vel['rotate_left'] = pd.DataFrame(np.array(rotate_left_list),
                                                   index=np.arange(self.cmd_vel.shape[0]), columns=['rotate_left'])
        self.cmd_vel['rotate_right'] = pd.DataFrame(np.array(rotate_right_list),
                                                    index=np.arange(self.cmd_vel.shape[0]),
                                                    columns=['rotate_right'])
        self.cmd_vel['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list),
                                                      index=np.arange(self.cmd_vel.shape[0]),
                                                      columns=['ahead_straight'])
        self.cmd_vel['ahead_left'] = pd.DataFrame(np.array(ahead_left_list), index=np.arange(self.cmd_vel.shape[0]),
                                                  columns=['ahead_left'])
        self.cmd_vel['ahead_right'] = pd.DataFrame(np.array(ahead_right_list),
                                                   index=np.arange(self.cmd_vel.shape[0]), columns=['ahead_right'])

        # print('self.cmd_vel: ', self.cmd_vel)

        print('classifier_fn_image ended')

        return np.array(self.cmd_vel.iloc[:, 3:]), planner_time

    def plotMinimalEvaluation(self, num_samples):
        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        # '''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        # '''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot segments with centroids and labels/weights
        ax.imshow(segments_1.astype('uint8'), aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(centers[i][0], centers[i][1],
                             str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig('segments_' + str(num_samples) + '.png')
        fig.clf()

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(int(
                (self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(
                int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # plot explanation
        fig = plt.figure(frameon=True)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 0.95])
        ax.set_axis_off()
        fig.add_axes(ax)
        # plot robots' location, orientation and local plan
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
            round(self.cmd_vel_original_tmp.iloc[1], 2)))
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0),
                                            outline_color=(0, 0, 0), mode='outer', background_label=0)
        ax.imshow(marked_boundaries.astype('float64'), aspect='auto')
        fig.savefig('explanation_' + str(num_samples) + '.png', transparent=False)
        fig.clf()

    def plotMinimalEvaluationFlipped(self, num_samples):
        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        # '''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        # '''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot segments with centroids and labels/weights
        ax.imshow(self.matrixFlip(segments_1, 'h').astype('uint8'), aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(160 - centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(160 - centers[i][0], centers[i][1],
                             str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig('flipped_segments_' + str(num_samples) + '.png')
        fig.clf()

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(160 - int(
                (self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(
                int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))


        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = 160 - int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        znak = 1
        if self.odom_z < 0:
            znak = -1
        self.odom_z = znak * (1 - abs(self.odom_z))        
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)        

        # plot explanation
        fig = plt.figure(frameon=True)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 0.95])
        ax.set_axis_off()
        fig.add_axes(ax)
        # plot robots' location, orientation and local plan
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')    
        
        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
            round(self.cmd_vel_original_tmp.iloc[1], 2)))
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0),
                                            outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries_flipped.astype('float64'), aspect='auto')
        fig.savefig('flipped_explanation_' + str(num_samples) + '.png', transparent=False)
        fig.clf()

    def explain_instance_evaluation(self, expID, ID):
        print('explain_instance function starting\n')

        self.expID = expID
            
        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.index = self.expID

            self.printImportantInformation()

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

            '''
            # If a custom costmap is used - TO-DO: make custom map loading a separate case in explainer.py
            self.local_costmap_original.to_csv('~/amar_ws/costmapToChange.csv', index=False, header=True)
            self.local_costmap_original = pd.read_csv('~/amar_ws/costmapToChange.csv')
            '''

            # Make image a np.array deepcopy of local_costmap_original
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            #'''
            # Turn inflated area to free space and 100s to 99s
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):
                    if 99 > self.image[i, j] > 0:
                        self.image[i, j] = 0
                    elif self.image[i, j] == 100:
                        self.image[i, j] = 99
            #'''

            # Turn point free space (that is surrounded by obstacles) to point obstacle - not really needed
            #self.PFP2PO()

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0

            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveData()

            # Use new variable in the algorithm - possible time saving
            img = copy.deepcopy(self.image)

            samples_num, segments_num = self.calculateNumOfSamples(img)
            #print(samples_num)
            #print(segments_num)
            
            # my custom segmentation func
            segm_fn = 'custom_segmentation'
 
            import time

            with open('explanations' + str(ID) + '.csv', "w") as myfile:
                myfile.write('num_samples,segmentation_time,classifier_fn_time,planner_time,explanation_time,explanation_pics_time,plotting_time,weight_0,weight_1,weight_2,weight_3,weight_4,weight_5\n')
                
                for i in range(0, segments_num + 1):
                    num_of_iterations_for_one_segment_size = 1 #30
                    for j in range(0, num_of_iterations_for_one_segment_size): 
                        start = time.time()
                        self.explanation, segmentation_time, classifier_fn_time, planner_time = self.explainer.explain_instance_evaluation(
                            img, self.classifier_fn_image_evaluation,
                            hide_color=perturb_hide_color_value,
                            num_segments=segments_num,
                            num_segments_current=i,
                            batch_size=1024, segmentation_fn=segm_fn,
                            top_labels=10)
                        end = time.time()
                        explanation_time = round(end - start, 3)
                        # print('Explanation time: ', explanation_time)

                        start = time.time()
                        self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0,
                                                                                                positive_only=False,
                                                                                                negative_only=False,
                                                                                                num_features=100,
                                                                                                hide_rest=False,
                                                                                                min_weight=0.1)  # min_weight=0.1 - default
                        end = time.time()
                        explanation_pics_time = round(end - start, 3)
                        # print('Getting explanation pics time: ', explanation_pics_time)

                        start = time.time()
                        self.plotMinimalEvaluation(2 ** i)
                        self.plotMinimalEvaluationFlipped(2 ** i)
                        end = time.time()
                        plotting_time = round(end - start, 3)
                        # print('Plotting time: ', round(plotting_time, 3))

                        # print(self.exp)

                        segmentation_time = round(segmentation_time, 3)
                        classifier_fn_time = round(classifier_fn_time, 3)
                        planner_time = round(planner_time, 3)

                        myfile.write(
                            str(2 ** i) + ',' + str(segmentation_time) + ',' + str(classifier_fn_time) + ',' + str(
                                planner_time) + ',' + str(explanation_time) + ',' + str(
                                explanation_pics_time) + ',' + str(plotting_time) + ',')
                        for k in range(0, len(self.exp)):
                            for l in range(0, len(self.exp)):
                                if k == self.exp[l][0]:
                                    if k != len(self.exp) - 1:
                                        myfile.write(str(self.exp[l][1]) + ',')
                                    else:
                                        myfile.write(str(self.exp[l][1]))
                                    break
                        myfile.write('\n')
                    myfile.write('\n')



    def mySlicEval(self, img_rgb):

        print('mySlic for evaluation starts')

        img = img_rgb[:, :, 0]

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_1
        segments_unique_1 = np.unique(segments_1)
        print('segments_unique_1: ', segments_unique_1)
        print('segments_unique_1.shape: ', segments_unique_1.shape)

        
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique: ', segments_unique)
        print('segments_unique.shape: ', segments_unique.shape)

        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k
                        break
        # find segments_unique after nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique (with nice numbering): ', segments_unique)
        print('segments_unique.shape (with nice numbering): ', segments_unique.shape)
        
        # print explanation
        #print('self.exp: ', self.exp)
        #print('len(self.exp): ', len(self.exp))

        print('mySlic for evaluation ends')

        return segments_1

    def getSegmentsForEval(self, image):

        print('Test segmentation function beginning')

        # Make image a np.array deepcopy of local_costmap_original
        img_ = copy.deepcopy(image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # Generate segments - superpixels with my slic function
        segments = self.mySlicEval(rgb)

        print('Test segmentation function ending')

        return segments



    # functions that I currently do not use:
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