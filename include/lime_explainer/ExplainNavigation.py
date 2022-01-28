#!/usr/bin/env python3

# import time tabular
from selectors import EpollSelector
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
    # constructor
    def __init__(self, cmd_vel, odom, plan, global_plan, local_plan, current_goal, local_costmap_data,
                 local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, tabular_mode, explanation_mode,
                 num_of_first_rows_to_delete, footprints, output_class_name, X_train, X_test, y_train, y_test, num_samples):
        print('\nConstructor starting')

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
        self.tabular_mode = tabular_mode
        self.explanation_mode = explanation_mode
        self.offset = num_of_first_rows_to_delete
        self.footprints = footprints
        self.costmap_size = local_costmap_info.iloc[0, 2]
        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test
        self.num_samples = num_samples
        self.output_class_name = output_class_name

        # manually modified LIME image
        if self.explanation_mode == 'image':
            self.explainer = lime_image.LimeImageExplainer(verbose=True)

        elif self.explanation_mode == 'tabular':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.X_train),
                                                                    feature_names=self.X_train.columns, mode=self.tabular_mode,
                                                                    class_names=[self.output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False,
                                                                    sample_around_instance=False, random_state=None)
        elif self.explanation_mode == 'tabular_costmap':
            pass
        
        print('\nConstructor ending')

    # explain instance
    def explain_instance(self, expID):
        print('\nexplain_instance starting')

        self.expID = expID
            
        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.index = self.expID

            self.manual_instance_loading = False
            self.manually_make_semantic_map = False
            self.test_segmentation = False 

            if self.manual_instance_loading == False:
                # Get local costmap
                # Original costmap will be saved to self.local_costmap_original
                self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

                # Make image a np.array deepcopy of local_costmap_original
                self.image = np.array(copy.deepcopy(self.local_costmap_original))

                # Turn inflated area to free space and 100s to 99s
                self.inflatedToFree()

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
                self.image = self.image * 1.0

            elif self.manual_instance_loading == True:
                #pd.DataFrame(self.image).to_csv('costmap_new.csv', index=False)
                self.image = np.array(pd.read_csv('costmap_new.csv')) * 1.0

                # Save footprint instance to a file
                #self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
                #self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
                #self.footprint_tmp.to_csv('footprint_new.csv', index=False, header=True)
                self.footprint_tmp = pd.read_csv('footprint_new.csv')
                #print(self.footprint_tmp)
                
                # Save local plan instance to a file
                #self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
                #self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
                #self.local_plan_tmp.to_csv('local_plan_new.csv', index=False, header=True)
                self.local_plan_tmp = pd.read_csv('local_plan_new.csv')

                # Save plan (from global planner) instance to a file
                #self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
                #self.plan_tmp = self.plan_tmp.iloc[:, 1:]
                #self.plan_tmp.to_csv('plan_new.csv', index=False, header=True)
                self.plan_tmp = pd.read_csv('plan_new.csv')

                # Save global plan instance to a file
                #self.global_plan_tmp = self.global_plan.loc[self.global_plan['ID'] == self.index + self.offset]
                #self.global_plan_tmp = self.global_plan_tmp.iloc[:, 1:]
                #self.global_plan_tmp.to_csv('global_plan_new.csv', index=False, header=True)
                self.global_plan_tmp = pd.read_csv('global_plan_new.csv')

                # Save costmap_info instance to file
                #self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
                #self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
                #self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
                #self.costmap_info_tmp.to_csv('costmap_info_new.csv', index=False, header=True)
                self.costmap_info_tmp = pd.read_csv('costmap_info_new.csv')

                # Save amcl_pose instance to file
                #self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
                #self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
                #self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
                #self.amcl_pose_tmp.to_csv('amcl_pose_new.csv', index=False, header=True)
                self.amcl_pose_tmp = pd.read_csv('amcl_pose_new.csv')

                # Save tf_odom_map instance to file
                #self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
                #self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
                #self.tf_odom_map_tmp.to_csv('tf_odom_map_new.csv', index=False, header=True)
                self.tf_odom_map_tmp = pd.read_csv('tf_odom_map_new.csv')

                # Save tf_map_odom instance to file
                #self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
                #self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
                #self.tf_map_odom_tmp.to_csv('tf_map_odom_new.csv', index=False, header=True)
                self.tf_map_odom_tmp = pd.read_csv('tf_map_odom_new.csv')

                # Save odometry instance to file
                #self.odom_tmp = self.odom.iloc[self.index, :]
                #self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
                #self.odom_tmp = self.odom_tmp.iloc[:, 2:]
                #self.odom_tmp.to_csv('odom_new.csv', index=False, header=True)
                self.odom_tmp = pd.read_csv('odom_new.csv')
            
            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveDataForLocalPlanner()

            # Saving important data to class variables
            self.saveImportantData2ClassVars()

            # Use new variable in the algorithm - possible time saving
            img = copy.deepcopy(self.image)

            self.semantic_seg = False
            if self.semantic_seg == True:
                segm_fn = 'semantic_segmentation'
            elif self.semantic_seg == False:
                segm_fn = 'custom_segmentation'

            devDistance_x, sum_x, devDistance_y, sum_y, devDistance = self.findDevDistance()
            #print('self.findDevDistance(): ', devDistance)
            #print('sum: ', sum)

            if self.manually_make_semantic_map == True:
                # manually make semantic map
                self.costmap2Map()
            else:
                if self.test_segmentation == True:
                    self.testSegmentation(self.expID)
                else:    
                    self.explanation, self.segments = self.explainer.explain_instance(img, self.classifier_fn_image, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                                                                self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                                                                                self.plan_x_list, self.plan_y_list,
                                                                                hide_color=perturb_hide_color_value, batch_size=2048, segmentation_fn=segm_fn, top_labels=10)
                    
                    self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=False, negative_only=False, num_features=100,
                                                                                hide_rest=False, min_weight=0.0)            
                    
                    self.plotExplanation()

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
            self.index = self.expID
            img = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]
            lista = []
            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    lista.append(img.iloc[i, j])
            self.tabular_costmap = pd.DataFrame(lista)
            self.tabular_costmap = pd.DataFrame(self.tabular_costmap).transpose()
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.tabular_costmap),
                                                                    feature_names=self.tabular_costmap.columns,
                                                                    mode=self.tabular_mode, class_names=[self.output_class_name],
                                                                    verbose=True, feature_selection='none',
                                                                    discretize_continuous=False)

            self.explanation = self.explainer.explain_instance(data_row=self.tabular_costmap,
                                                               predict_fn=self.classifier_fn_tabular_costmap,
                                                               num_samples=self.num_samples,
                                                               num_features=self.tabular_costmap.shape[1])
            # print(self.explanation.as_list())
            fig = self.explanation.as_pyplot_figure()
            plt.savefig('explanation.png')
        
        print('\nexplain_instance ending')

    # helper function for lime image
    def findDevDistance(self):
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)

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

        devs_x = []
        signs_x = []
        devs_y = []
        signs_y = []
        devs = []
        for j in range( 0, len(self.local_plan_x_list)):
            diffs_x = []
            sums_x = []
            diffs_y = []
            sums_y = []
            diffs = []
            for k in range(0, len(self.plan_x_list)):
                diff_x = self.local_plan_x_list[j] - self.plan_x_list[k]
                sums_x.append(diff_x)
                diffs_x.append(abs(diff_x))

                diff_y = self.local_plan_y_list[j] - self.plan_y_list[k]
                sums_y.append(diff_y)
                diffs_y.append(abs(diff_y))

                diff = math.sqrt(diff_x**2 + diff_y**2)
                diffs.append(diff)                
            devs_x.append(min(diffs_x))
            signs_x.append(sums_x[diffs_x.index(devs_x[-1])])

            devs_y.append(min(diffs_y))
            signs_y.append(sums_y[diffs_y.index(devs_y[-1])])

            devs.append(min(diffs))

        return max(devs_x), np.sign(signs_x[devs_x.index(max(devs_x))]), max(devs_y), np.sign(signs_y[devs_y.index(max(devs_y))]), max(devs)    

    # helper function for creating manual semantic maps
    def costmap2Map(self):
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([self.tf_odom_map_tmp.iloc[0, 3], self.tf_odom_map_tmp.iloc[0, 4], self.tf_odom_map_tmp.iloc[0, 5], self.tf_odom_map_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array([self.tf_odom_map_tmp.iloc[0, 0], self.tf_odom_map_tmp.iloc[0, 1], self.tf_odom_map_tmp.iloc[0, 2]])
        # print('t: ', t)

        costmap_segmented = pd.read_csv('costmap_segmented.csv')

        map_sem_1 = np.zeros((304, 201), np.uint8)
        map_sem_2 = np.zeros((304, 201), np.uint8)

        #indices_x = np.zeros((160, 160), np.uint8)
        #indices_y = np.zeros((160, 160), np.uint8)

        map_pairs = []

        br_duplikata = 0
        
        for i in range(0, 160):
            for j in range(0, 160):
                x = j * self.localCostmapResolution + self.localCostmapOriginX
                y = i * self.localCostmapResolution + self.localCostmapOriginY

                p = np.array([x, y, 0])
                # print('p: ', p)
                pnew = p.dot(r_array) + t
                #print('\n(i, j) = ', (i, j))
                #print('pnew: ', pnew)
                x = pnew[0]
                y = pnew[1]

                j_map = int((x - self.mapOriginX) / self.mapResolution + 0.5)
                i_map = int((y - self.mapOriginY) / self.mapResolution + 0.5)

                #print('(i_map, j_map) = ', ((y - self.mapOriginY) / self.mapResolution + 0.5, (x - self.mapOriginX) / self.mapResolution + 0.5))

                if (i_map, j_map) not in map_pairs:
                    map_pairs.append((i_map, j_map))
                    map_sem_1[i_map, j_map] = costmap_segmented.iloc[i, j]
                    #print('(i_map, j_map) = ', (i_map, j_map))
                    #indices_x[i, j] = j_map
                    #indices_y[i, j] = i_map    
                else:
                    map_sem_2[i_map, j_map] = costmap_segmented.iloc[i, j]
                    #br_duplikata += 1   
                    #print('(i_map, j_map) = ', (i_map, j_map))
                    #print('duplikat')    

        pd.DataFrame(map_sem_1).to_csv('semantic_map_1_temporary.csv', index=False, header=False)
        pd.DataFrame(map_sem_2).to_csv('semantic_map_2_temporary.csv', index=False, header=False)
        #pd.DataFrame(indices_x).to_csv('INDICES_X.csv', index=False, header=False)
        #pd.DataFrame(indices_y).to_csv('INDICES_Y.csv', index=False, header=False)

        #print('\nbr_duplikata: ', br_duplikata)

    # classifier function for lime image
    def classifier_fn_image(self, sampled_instance):

        #print('\nclassifier_fn_image started')

        # sampled_instance info
        #print('sampled_instance: ', sampled_instance)
        #print('sampled_instance.shape: ', sampled_instance.shape)

        #pd.DataFrame(self.image).to_csv('image.csv')
        
        #'''
        # I will use channel 0 from sampled_instance as actual perturbed data
        # Perturbed pixel intensity is perturb_hide_color_value
        # Convert perturbed free space to obstacle (99), and perturbed obstacles to free space (0) in all perturbations
        for i in range(0, sampled_instance.shape[0]):
            #pd.DataFrame(sampled_instance[i][:, :, 0]).to_csv('original' + str(i) + '.csv')
            for j in range(0, sampled_instance[i].shape[0]):
                for k in range(0, sampled_instance[i].shape[1]):
                    if sampled_instance[i][j, k, 0] == perturb_hide_color_value:
                        if self.image[j, k] == 0.0:
                            sampled_instance[i][j, k, 0] = 99.0
                            #print('free space')
                        elif self.image[j, k] == 99.0:
                            sampled_instance[i][j, k, 0] = 0.0
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

        # calling ROS C++ node
        #print('\nstarting C++ node')

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        #rospy.sleep(1)

        #print('\nC++ node ended')

        # load command velocities
        self.cmd_vel_perturb = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('self.cmd_vel: ', self.cmd_vel_perturb)
        #print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)
        #self.cmd_vel_perturb.to_csv('cmd_vel.csv')

        # load local plans
        self.local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')
        #print('self.local_plans: ', self.local_plans)
        #print('self.local_plans.shape: ', self.local_plans.shape)
        #self.local_plans.to_csv('local_plans.csv')

        # load transformed plan
        self.transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('self.transformed_plan: ', self.transformed_plan)
        #print('self.transformed_plan.shape: ', self.transformed_plan.shape)


        plot_perturbations = False
        if plot_perturbations == True:
            # only needed for classifier_fn_image_plot() function
            self.sampled_instance = sampled_instance

            # plot perturbation of local costmap
            self.classifier_fn_image_plot()

        print_iterations = False
        
        mode = 'regression' # 'regression' or 'classification'
  
        import math

        # fill the list of the original local plan coordinates
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
        #closest_to_robot_index = -100
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
                    #closest_to_robot_index = len(transformed_plan_xs) - 1
        transformed_plan_xs = np.array(transformed_plan_xs)
        transformed_plan_ys = np.array(transformed_plan_ys)

        # DETERMINE THE DEVIATION TYPE
        # thresholds
        local_plan_gap_threshold = 48 #60 #48 #32
        small_deviation_threshold = 7.0 #5 #7
        big_deviation_threshold = 14
        no_deviation_threshold = 3.0

        # test for the original local plan gap
        local_plan_original_gap = False
        local_plan_gaps = []
        diff = 0
        for j in range(0, len(local_plan_xs_orig) - 1):
            diff = math.sqrt( (local_plan_xs_orig[j]-local_plan_xs_orig[j+1])**2 + (local_plan_ys_orig[j]-local_plan_ys_orig[j+1])**2 )
            local_plan_gaps.append(diff)
        if max(local_plan_gaps) > local_plan_gap_threshold:
            local_plan_original_gap = True

        # local gap too big - stop
        if local_plan_original_gap == True or len(local_plan_xs_orig) == 0:
            deviation_type = 'stop'
            local_plan_gap_threshold = 55
        # no local gap - test further    
        else:        
            diff_x = 0
            diff_y = 0
            
            big_deviation = False
            devs = []
            for j in range( 0, len(local_plan_xs_orig)):
                diffs = []
                deviation_local = True  
                for k in range(0, len(transformed_plan_xs)):
                    diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    diffs.append(diff)
                    if diff <= big_deviation_threshold:
                        deviation_local = False
                        break # commented (comment out) because of big_deviation_threshold = max(devs) * 0.8
                devs.append(min(diffs))    
                if deviation_local == True:
                    big_deviation = True
                    break # commented (comment out) because of big_deviation_threshold = max(devs) * 0.8
            
            if big_deviation == True:
                deviation_type = 'big_deviation'
                local_plan_gap_threshold = 48
                #print('max_dev: ', max(devs))
                big_deviation_threshold = max(devs) * 0.8
                #print('big_deviation_threshold: ', big_deviation_threshold)
            else:
                diff_x = 0
                diff_y = 0
            
                small_deviation = False
                for j in range( 0, len(local_plan_xs_orig)):
                    deviation_local = True  
                    for k in range(0, len(transformed_plan_xs)):
                        diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        if diff <= small_deviation_threshold:
                            deviation_local = False
                            break
                    if deviation_local == True:
                        small_deviation = True
                        break
                if small_deviation == True:
                    deviation_type = 'small_deviation'
                    #print('max_dev: ', max(devs))
                else:
                    deviation_type = 'no_deviation'
                    #print('max_dev: ', max(devs))        

        # PRINTING RESULTS                                       
        #print('\nself.expID: ', self.expID)
        print('deviation_type: ', deviation_type)
        #print('local_plan_original_gap: ', local_plan_original_gap)
        #print('command velocities original - lin_x: ' + str(self.cmd_vel_original_tmp.iloc[0]) + ', ang_z: ' + str(self.cmd_vel_original_tmp.iloc[1]) + '\n')
        '''
        if local_plan_original_gap == True:
            print('\nmax(local_plan_original_gaps): ', max(local_plan_gaps))
            print('len(local_plan_orig): ', len(local_plan_xs_orig))
            print('\n')
        '''
        
        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        #### MAIN PART ####

        if mode == 'regression':
            # fill in deviation dataframe
            dev_original = 0
            # find if there is local plan
            for i in range(0, sampled_instance.shape[0]):
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

                if local_plan_found == False:
                    if deviation_type == 'stop':
                        self.local_plan_deviation.iloc[i, 0] = dev_original
                    elif deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 100
                    elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    continue             

                diff_x = 0
                diff_y = 0
                devs = []
                for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                    local_diffs = []
                    deviation_local = True  
                    for k in range(0, len(transformed_plan_xs)):
                        diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        local_diffs.append(diff)                        
                    devs.append(min(local_diffs))

                if i == 0:
                    dev_original = max(devs)    

                self.local_plan_deviation.iloc[i, 0] = max(devs)

        elif mode == 'classification':                
            if deviation_type == 'stop':
                # fill in deviation dataframe
                for i in range(0, sampled_instance.shape[0]):
                    print('\ni = ', i)
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
                        
                    local_plan_point_in_obstacle = False
                    local_plan_found = True    
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(0, len(local_plan_xs)):
                            if sampled_instance[i][local_plan_ys[j], local_plan_xs[j], 0] == 99:
                                local_plan_point_in_obstacle = True
                                break

                        if local_plan_point_in_obstacle == True:
                            self.local_plan_deviation.iloc[i, 0] = 1.0
                        else:    
                            # test if there is local plan gap
                            diff = 0
                            local_plan_gap = False
                            local_plan_gaps = []
                            for j in range(0, len(local_plan_xs) - 1):
                                diff = math.sqrt( (local_plan_xs[j]-local_plan_xs[j+1])**2 + (local_plan_ys[j]-local_plan_ys[j+1])**2 )
                                local_plan_gaps.append(diff)
                            
                            max_local_plan_gap = max(local_plan_gaps)

                            if max_local_plan_gap > local_plan_gap_threshold:
                                local_plan_gap = True
                            else:
                                max_local_plan_gap_index = local_plan_gaps.index(max_local_plan_gap)
                                idx = max_local_plan_gap_index
                                d_x = local_plan_xs[idx + 1] - local_plan_xs[idx]
                                if d_x != 0:
                                    k = (local_plan_ys[idx + 1] - local_plan_ys[idx]) / (d_x)
                                    n = local_plan_ys[idx] - k * local_plan_xs[idx]

                                x_indices = []
                                y_indices = []

                                if abs(d_x) <= 5:
                                    if local_plan_ys[idx] <= local_plan_ys[idx + 1]:
                                        for y in range(local_plan_ys[idx], local_plan_ys[idx + 1] + 1):
                                            x_indices.append(local_plan_xs[idx])
                                            x_indices.append(local_plan_xs[idx+1])
                                            y_indices.append(y)
                                            y_indices.append(y)
                                    else:
                                        for y in range(local_plan_ys[idx+1], local_plan_ys[idx] + 1):
                                            x_indices.append(local_plan_xs[idx])
                                            x_indices.append(local_plan_xs[idx+1])
                                            y_indices.append(y)
                                            y_indices.append(y)        
                                else:            
                                    if local_plan_xs[idx] <= local_plan_xs[idx + 1]:
                                        for x in range(local_plan_xs[idx], local_plan_xs[idx + 1] + 1):
                                            y = int(k * x + n + 0.5)
                                            x_indices.append(x)
                                            y_indices.append(y)
                                    else:
                                        for x in range(local_plan_xs[idx + 1], local_plan_xs[idx] + 1):
                                            y = int(k * x + n + 0.5)
                                            x_indices.append(x)
                                            y_indices.append(y)

                                
                                print('idx = ', idx)
                                print('(local_plan_xs[idx], local_plan_ys[idx]): ', (local_plan_xs[idx], local_plan_ys[idx]))
                                print('(local_plan_xs[idx+1], local_plan_ys[idx+1]): ', (local_plan_xs[idx+1], local_plan_ys[idx+1]))
                                print('d_x: ', d_x)
                                if d_x != 0:
                                    print('k = ', k)
                                    print('n = ', n)
                                print('x_indices: ', x_indices)
                                print('y_indices: ', y_indices)
                                

                                if i == 33:
                                    pd.DataFrame(sampled_instance[i][:, :, 0]).to_csv('33.csv')

                                for j in range(0, len(x_indices)):
                                    for m in range(x_indices[j] - 0, x_indices[j] + 1):
                                        for q in range(y_indices[j] - 0, y_indices[j] + 1):
                                            if i == 33:
                                                print('(m, q, sampled_instance[i][q, m, 0]): ', (m, q, sampled_instance[i][q, m, 0]))
                                            if sampled_instance[i][q, m, 0] == 99:
                                                local_plan_gap = True
                                                break                 
                            
                            if local_plan_gap == True:
                                self.local_plan_deviation.iloc[i, 0] = 1.0
                            else:
                                self.local_plan_deviation.iloc[i, 0] = 0.0 
                    else:
                        self.local_plan_deviation.iloc[i, 0] = 1.0

                    if print_iterations == True:
                        print('\ni: ', i)
                        print('local plan found: ', local_plan_found)
                        print('local_plan_point_in_obstacle: ', local_plan_point_in_obstacle)
                        if local_plan_found == True and local_plan_point_in_obstacle == False:
                            print('local plan length: ', len(local_plan_xs))
                            print('local_plan_gap: ', local_plan_gap)
                            print('max(local_plan_gaps): ', max(local_plan_gaps))  
                        print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                        print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                
            elif deviation_type == 'big_deviation': 
                print('i = ', i)
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
                    
                local_plan_point_in_obstacle = False    
                if local_plan_found == True:
                    # test if any part of the local plan is in the obstacle region
                    for j in range(len(local_plan_xs) - 1, len(local_plan_xs)):
                        if sampled_instance[i][local_plan_ys[j], local_plan_xs[j], 0] == 99:
                            local_plan_point_in_obstacle = True
                            break

                    # if there is local plan in obstacle, it is stop - no deviation
                    if local_plan_point_in_obstacle == True:
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    else:                  
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
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        else:
                            diff_x = 0
                            diff_y = 0
                            real_deviation = False
                            devs = []
                            for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                                local_diffs = []
                                deviation_local = True  
                                for k in range(0, len(transformed_plan_xs)):
                                    diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                                    diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                                    diff = math.sqrt(diff_x + diff_y)
                                    local_diffs.append(diff)
                                    if diff <= big_deviation_threshold:
                                        deviation_local = False
                                        break # comment out to get the real biggest minimal difference between local and tranformed plan
                                devs.append(min(local_diffs))
                                if deviation_local == True:
                                    real_deviation = True
                                    break # comment out to get the real biggest minimal difference between local and tranformed plan
                            
                            if real_deviation == True:
                                self.local_plan_deviation.iloc[i, 0] = 1.0
                            else:    
                                self.local_plan_deviation.iloc[i, 0] = 0.0                
                else:
                    self.local_plan_deviation.iloc[i, 0] = 0.0
                
                if print_iterations == True:
                    print('\ni: ', i)
                    print('local plan found: ', local_plan_found)
                    print('local_plan_point_in_obstacle: ', local_plan_point_in_obstacle)
                    if local_plan_found == True and local_plan_point_in_obstacle == False:
                        print('local plan length: ', len(local_plan_xs))
                        print('local_plan_gap: ', local_plan_gap)
                        print('max(local_plan_gaps): ', max(local_plan_gaps))
                        if local_plan_gap == False:
                            print('deviation: ', real_deviation)
                            print('max dev: ', max(devs))
                    print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                    print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                
            elif deviation_type == 'small_deviation':
                # fill in deviation dataframe
                for i in range(0, sampled_instance.shape[0]):
                    print('i = ', i)
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

                    local_plan_point_in_obstacle = False 
                    # if there is local plan, test further   
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(0, len(local_plan_xs)):
                            if sampled_instance[i][local_plan_ys[j], local_plan_xs[j], 0] == 99:
                                local_plan_point_in_obstacle = True
                                break

                        # if any part of the local plan is in the obstacle, it is not small deviation
                        if local_plan_point_in_obstacle == True:
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        else:    
                            # test if there is local plan gap
                            diff = 0
                            local_plan_gap = False
                            local_plan_gaps = []
                            for j in range(0, len(local_plan_xs) - 1):
                                diff = math.sqrt( (local_plan_xs[j]-local_plan_xs[j+1])**2 + (local_plan_ys[j]-local_plan_ys[j+1])**2 )
                                local_plan_gaps.append(diff)
                            
                            if max(local_plan_gaps) > local_plan_gap_threshold:
                                local_plan_gap = True
                            
                            # if there is a big local plan gap, it is not small deviation
                            if local_plan_gap == True:
                                self.local_plan_deviation.iloc[i, 0] = 0.0
                            else:
                                diff_x = 0
                                diff_y = 0
                                small_deviation = False
                                big_dev = False
                                devs = []
                                # fix the local plan point and find the differences between the fixed local plan point and all transformed plan points
                                for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                                    diffs = []
                                    deviation_local = False  
                                    for k in range(0, len(transformed_plan_xs)):
                                        diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                                        diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                                        diff = math.sqrt(diff_x + diff_y)
                                        #print('diff: ', diff)
                                        diffs.append(diff)
                                    #print('j = ', j)
                                    #print('min(diffs): ', min(diffs))
                                    #print('diffs: ', diffs)

                                    min_temp = min(diffs)

                                    # test if minimal of these differences is bigger than big deviation threshold
                                    # If it is than it is big deviation
                                    if min(diffs) >= big_deviation_threshold:
                                        #print('BIG')
                                        big_dev = True
                                        break # comment out to get the real biggest minimal difference between local and tranformed plan
                                    # if minimal of these differences is bigger than small deviation threshold, it is small deviation than    
                                    if min(diffs) >= small_deviation_threshold:
                                        #print('SMALL')
                                        small_deviation = True
                                        break # comment out to get the real biggest minimal difference between local and tranformed plan

                                    devs.append(min(diffs))    

                                if big_dev == True:
                                    small_deviation = False
                                
                                if small_deviation == True:
                                    self.local_plan_deviation.iloc[i, 0] = 1.0
                                else:    
                                    self.local_plan_deviation.iloc[i, 0] = 0.0                
                    # if there is no local plan it is not a small deviation
                    else:
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    
                    if print_iterations == True:
                        print('\ni: ', i)
                        print('local plan found: ', local_plan_found)
                        print('local_plan_point_in_obstacle: ', local_plan_point_in_obstacle)
                        if local_plan_found == True and local_plan_point_in_obstacle == False:
                            print('local plan length: ', len(local_plan_xs))
                            print('local_plan_gap: ', local_plan_gap)
                            print('max(local_plan_gaps): ', max(local_plan_gaps))
                            if local_plan_gap == False:
                                print('deviation: ', small_deviation)
                                #print('minimal diff: ', min(diffs))
                                print('max(devs): ', max(devs))
                        print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                        print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                
            elif deviation_type == 'no_deviation':
                # fill in deviation dataframe
                for i in range(0, sampled_instance.shape[0]):
                    print('i = ', i)
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
                        
                    local_plan_point_in_obstacle = False
                    # if there is local plan test further    
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(0, len(local_plan_xs)):
                            if sampled_instance[i][local_plan_ys[j], local_plan_xs[j], 0] == 99:
                                local_plan_point_in_obstacle = True
                                break

                        # if there is a local plan point in the obstacle, it is not 'no deviation'
                        if local_plan_point_in_obstacle == True:
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        # if there is no local plan point in the obstacle, test further    
                        else:    
                            # test if there is local plan gap
                            diff = 0
                            local_plan_gap = False
                            local_plan_gaps = []
                            for j in range(0, len(local_plan_xs) - 1):
                                diff = math.sqrt( (local_plan_xs[j]-local_plan_xs[j+1])**2 + (local_plan_ys[j]-local_plan_ys[j+1])**2 )
                                local_plan_gaps.append(diff)
                            
                            if max(local_plan_gaps) > local_plan_gap_threshold:
                                local_plan_gap = True
                            
                            # if there is local plan gap, it is not "no deviation"
                            if local_plan_gap == True:
                                self.local_plan_deviation.iloc[i, 0] = 0.0
                            # if there is no local plan gap, test further    
                            else:
                                diff_x = 0
                                diff_y = 0
                                real_deviation = False
                                devs = []
                                for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                                    diffs = []
                                    deviation_local = True  
                                    for k in range(0, len(transformed_plan_xs)):
                                        diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                                        diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                                        diff = math.sqrt(diff_x + diff_y)
                                        diffs.append(diff)
                                        if diff < no_deviation_threshold:
                                            deviation_local = False
                                            #break # comment out to get the real biggest minimal difference between local and tranformed plan
                                    #print('j = ', j)
                                    #print('min(diffs): ', min(diffs))
                                    #print('diffs: ', diffs)
                                    devs.append(min(diffs))
                                    if deviation_local == True:
                                        real_deviation = True
                                        #break # comment out to get the real biggest minimal difference between local and tranformed plan
                                
                                if real_deviation == True:
                                    self.local_plan_deviation.iloc[i, 0] = 0.0
                                else:    
                                    self.local_plan_deviation.iloc[i, 0] = 1.0                
                    # if there is no local plan, it is not "no deviation"
                    else:
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    
                    if print_iterations == True:
                        print('\ni: ', i)
                        print('local plan found: ', local_plan_found)
                        print('local_plan_point_in_obstacle: ', local_plan_point_in_obstacle)
                        if local_plan_found == True and local_plan_point_in_obstacle == False:
                            print('local plan length: ', len(local_plan_xs))
                            print('local_plan_gap: ', local_plan_gap)
                            print('max(local_plan_gaps): ', max(local_plan_gaps))
                            if local_plan_gap == False:
                                print('deviation: ', real_deviation)
                                print('minimal diff: ', min(diffs))
                                print('max(devs): ', max(devs))
                        print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                        print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
    
        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        #self.cmd_vel_perturb['deviate'].to_csv('izlaz.csv')
        
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

        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        #print('\nclassifier_fn_image ended\n')

        return np.array(self.cmd_vel_perturb.iloc[:, 3:])

    # function for plotting lime image perturbations
    def classifier_fn_image_plot(self):
        #'''
        # indices of transformed plan's poses in local costmap
        self.transformed_plan_x_list = []
        self.transformed_plan_y_list = []
        for j in range(0, self.transformed_plan.shape[0]):
            index_x = int((self.transformed_plan.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            index_y = int((self.transformed_plan.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= index_x < 160 and 0 <= index_y < 160:
                self.transformed_plan_x_list.append(index_x)
                self.transformed_plan_y_list.append(index_y)
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
            local_plan_x_list = []
            local_plan_y_list = []
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    index_x = int((self.local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    index_y = int((self.local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= index_x < 160 and 0 <= index_y < 160:
                        local_plan_x_list.append(index_x)
                        local_plan_y_list.append(index_y)
                    '''
                    [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans.iloc[j, 2], self.local_plans.iloc[j, 3])
                    yaw_x = math.cos(yaw)
                    yaw_y = math.sin(yaw)
                    plt.quiver(index_x, index_y, yaw_x, yaw_y, color='white')
                    '''
            # print('i: ', i)
            # print('local_plan_x_list.size(): ', len(local_plan_x_list))
            # print('local_plan_y_list.size(): ', len(local_plan_y_list))

            # plot transformed plan
            plt.scatter(self.transformed_plan_x_list, self.transformed_plan_y_list, c='blue', marker='x')

            # plot footprint
            plt.scatter(self.footprint_x_list, self.footprint_y_list, c='green', marker='x')

            '''
            print(self.footprint_x_list)
            print(self.footprint_y_list)
            for j in range(0, len(self.footprint_x_list)):
                print("footprint_distance: ", np.sqrt((self.footprint_x_list[j] - self.x_odom_index[0])**2 + abs(self.footprint_y_list[j] - self.y_odom_index[0])**2))
            '''

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
            plt.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='x')

            # plot local plan last point
            #if len(local_plan_x_list) != 0:
            #    plt.scatter([local_plan_x_list[-1]], [local_plan_y_list[-1]], c='black', marker='x')

            # plot robot's location and orientation
            plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

            # plot command velocities as text
            plt.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_perturb.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_perturb.iloc[i, 2], 2)))

            # save figure
            plt.savefig('perturbation_' + str(i) + '.png')
            plt.clf()
       


    # plot explanation picture and segments
    def plotExplanation(self):
        path_core = os.getcwd()

        # import needed libraries
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # plot costmap
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #print('self.image.shape: ', self.image.shape)
        gray_shade = 180
        white_shade = 255
        image = gray2rgb(self.image)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = gray_shade
                elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = white_shade    
        #pd.DataFrame(image[:,:,0]).to_csv('R.csv')
        #pd.DataFrame(image[:,:,1]).to_csv('G.csv')
        #pd.DataFrame(image[:,:,2]).to_csv('B.csv')            
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/costmap.png')
        fig.clf()

        # plot costmap with plans
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        #ax.imshow(self.image, aspect='auto')
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/input.png')
        fig.clf()

        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')

        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        #marked_boundaries = mark_boundaries(self.temp_img, self.mask, color=(180, 180, 180)) #color=(255, 255, 0)
        #ax.imshow(marked_boundaries.astype(np.uint8), aspect='auto')  # , aspect='auto')
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')  # , aspect='auto')
        fig.savefig(path_core + '/explanation.png', transparent=False)
        fig.clf()
                        
        if self.semantic_seg == False:
            # plot segments with centroids and labels/weights
            fig = plt.figure(frameon=False)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.segments, aspect='auto')

            #'''
            self.segments += 1
            #print('self.exp: ', self.exp)
            #print('np.unique(self.segments): ', np.unique(self.segments))
            #'''
            regions = regionprops(self.segments.astype(int))
            #'''
            labels = []
            for props in regions:
                labels.append(props.label)
            #print('labels: ', labels)
            #'''    
            i = 0
            for props in regions:
                v = props.label  # value of label
                cx, cy = props.centroid  # centroid coordinates
                ax.scatter(cy, cx, c='white', marker='o')   
                # printing/plotting explanation weights
                for j in range(0, len(self.exp)):
                    if self.exp[j][0] == v - 1:
                        ax.text(cy, cx, str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                        break
                i = i + 1

            # Save segments with nice numbering as a picture
            fig.savefig(path_core + '/weighted_segments.png')
            fig.clf()                

        elif self.semantic_seg == True:
            semantic_map_1 = np.array(pd.read_csv('~/amar_ws/semantic_map_1.csv'))
            semantic_map_2 = np.array(pd.read_csv('~/amar_ws/semantic_map_2.csv'))
            semantic_map = np.array(pd.read_csv('~/amar_ws/semantic_map.csv'))
            semantic_tags = pd.read_csv('~/amar_ws/semantic_tags.csv')
            self.costmap_segmented_included = True
            if self.costmap_segmented_included == True:
                costmap_segmented = np.array(pd.read_csv('~/amar_ws/costmap_segmented.csv'))

            # plot segments with weights
            self.segments += 1
            regions = regionprops(self.segments.astype(int))
            labels = []
            for props in regions:
                labels.append(props.label)
            fig = plt.figure(frameon=False)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)    
            i = 0
            for props in regions:
                v = props.label  # value of label
                cx, cy = props.centroid  # centroid coordinates
                ax.scatter(cy, cx, c='black', marker='o')   
                # printing/plotting explanation weights
                for j in range(0, len(self.exp)):
                    if self.exp[j][0] == v - 1:
                        ax.text(cy, cx, str(round(self.exp[j][1], 4)) + ', ' + str(round(self.exp[j][0], 4)), c='white')  # str(round(self.exp[j][1],4)) #str(v))
                        break
                i = i + 1
            # Save segments with nice numbering as a picture
            ax.imshow(self.segments, aspect='auto')
            fig.savefig(path_core + '/weighted_segments.png')
            fig.clf()


            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([self.tf_odom_map.iloc[0, 3], self.tf_odom_map.iloc[0, 4], self.tf_odom_map.iloc[0, 5], self.tf_odom_map.iloc[0, 6]])
            r_array = np.asarray(r.as_matrix())
            t = np.array([self.tf_odom_map.iloc[0, 0], self.tf_odom_map.iloc[0, 1], self.tf_odom_map.iloc[0, 2]])

            map_pairs = []

            semantic_local_costmap = np.zeros(self.segments.shape, np.uint8) + 10

            for i in range(0, 160):
                for j in range(0, 160):            
                    x = j * self.localCostmapResolution + self.localCostmapOriginX
                    y = i * self.localCostmapResolution + self.localCostmapOriginY

                    p = np.array([x, y, 0])
                    # print('p: ', p)
                    pnew = p.dot(r_array) + t
                    # print('pnew: ', pnew)
                    x = pnew[0]
                    y = pnew[1]

                    j_map = int((x - self.mapOriginX) / self.mapResolution + 0.5) #- 1
                    i_map = int((y - self.mapOriginY) / self.mapResolution + 0.5) #- 1

                    if (i_map, j_map) not in map_pairs:
                        map_pairs.append((i_map, j_map))
                        #semantic_local_costmap[i, j] = semantic_map[i_map, j_map]
                        semantic_local_costmap[i, j] = semantic_map_1[i_map, j_map]
                    else:
                        #semantic_local_costmap[i, j] = semantic_map[i_map, j_map]
                        semantic_local_costmap[i, j] = semantic_map_2[i_map, j_map]

            #pd.DataFrame(semantic_local_costmap).to_csv('semantic_local_costmap.csv')               

        
            unknown_obstacles_vals = []
            self.segments -= 1
            for val in np.unique(self.segments):
                # static known obstacle
                if np.all(self.image[self.segments == val] == 99) == True:
                    
                    if self.costmap_segmented_included == False:
                        semantic_num = np.bincount(semantic_local_costmap[self.segments == val]).argmax()
                    else:
                        semantic_num = np.bincount(costmap_segmented[self.segments == val]).argmax()
                    
                    if semantic_num >= 90:
                        pass
                        #print('\nSTATIC KNOWN OBSTACLE!')
                        #print('val: ', val)
                        #print('bincount: ', np.bincount(semantic_local_costmap[self.segments == val]))
                        #print('semantic_num: ', semantic_num)
                    
                    else:
                        unknown_obstacles_vals.append(val)
                        #print('\nSTATIC UNKNOWN OBSTACLE!')
                        #print('val: ', val)
                        #print('bincount: ', np.bincount(semantic_local_costmap[self.segments == val]))
                        #print('semantic_num: ', semantic_num)

                # free space    
                else:
                    pass
                    #print('\nFREE SPACE')
                    #print('val: ', val)
                    #print('bincount: ', np.bincount(semantic_local_costmap[self.segments == val]))
                    #print('semantic_num: ', np.bincount(semantic_local_costmap[self.segments == val]).argmax())

            # plot semantic map
            fig = plt.figure(frameon=False)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            for i in range(0, len(unknown_obstacles_vals)):
                x_s = []
                y_s = []
                color = []
                for j in range(0, self.segments.shape[0]):
                    for k in range(0, self.segments.shape[1]):
                        if self.segments[j, k] == unknown_obstacles_vals[i]:
                            x_s.append(k)
                            y_s.append(j)
                            R = self.temp_img[x_s[0], y_s[0]][0] / 255
                            G = self.temp_img[x_s[0], y_s[0]][1] / 255
                            B = self.temp_img[x_s[0], y_s[0]][2] / 255
                            color.append(np.array([R, G, B]))

                x_center = sum(x_s) / len(x_s)
                y_center = sum(y_s) / len(y_s)
                deltas = []

                for j in range(0, len(x_s)):
                    delta = (x_s[j] - x_center)**2 + (y_s[j] - y_center)**2
                    delta = math.sqrt(delta)
                    deltas.append(delta)

                max_delta = max(deltas)

                points_to_plot_x = []
                points_to_plot_y = []    

                for j in range(0, len(x_s)):
                    if deltas[j] >= 0.8 * max_delta:
                        points_to_plot_x.append(x_s[j])
                        points_to_plot_y.append(y_s[j])

                ax.scatter(points_to_plot_x, points_to_plot_y, c='yellow', marker='.')
                for j in range(0, len(self.exp)):
                    if self.exp[j][0] == unknown_obstacles_vals[i]:
                        print('\nUnknown obstacle ' + str(i+1) + ' has a weight ' + str(round(self.exp[j][1], 4)))
                        ax.text(x_center, y_center - 10, 'Unknown obstacle ' + str(i+1), c='white')
                        break
                

            ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')        
            ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='purple', marker='o')
            ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')

            if self.costmap_segmented_included == True:
                regions = regionprops(np.array(costmap_segmented).astype(int))
            else:
                regions = regionprops(np.array(semantic_local_costmap).astype(int))
            
            #'''
            labels = []
            for props in regions:
                labels.append(props.label)
            #print('labels_semantic: ', labels)
            #'''    
            i = 0
            for props in regions:
                v = props.label  # value of label
                cx, cy = props.centroid  # centroid coordinates
                #ax.scatter(cy, cx, c='white', marker='o')   
                # printing/plotting explanation weights
                for j in range(0, len(semantic_tags)):
                    if semantic_tags.iloc[j, 0] == v:
                        ax.text(cy, cx, semantic_tags.iloc[j, 1], c='white')  # str(round(self.exp[j][1],4)) #str(v))
                        break
                i = i + 1

            if self.costmap_segmented_included == True:
                marked_boundaries = mark_boundaries(self.temp_img, np.array(costmap_segmented), color=(255, 255, 0))
            else:
                marked_boundaries = mark_boundaries(self.temp_img, np.array(semantic_local_costmap), color=(255, 255, 0))
    
            ax.imshow(marked_boundaries.astype(np.uint8), aspect='auto')  # , aspect='auto')
            fig.savefig(path_core + '/semantic_explanation.png', transparent=False)
            fig.clf()

            fig = plt.figure(frameon=False)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')
            fig.savefig(path_core + '/temp_img.png')
            fig.clf()

            fig = plt.figure(frameon=False)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(np.array(semantic_local_costmap).astype(np.uint8), aspect='auto')
            fig.savefig(path_core + '/semantic_local_costmap.png')
            fig.clf()

            #print('unknown_obstacles_vals: ', unknown_obstacles_vals)

            for i in range(0, semantic_tags.shape[0]):

                if self.costmap_segmented_included == True:
                    segments_label = np.unique(self.segments[costmap_segmented == semantic_tags.iloc[i, 0]])
                else:
                    segments_label = np.unique(self.segments[semantic_local_costmap == semantic_tags.iloc[i, 0]])
                
                '''
                print('\n')
                print('i = ', i)
                print('segments_label: ', segments_label)
                print('semantic_tags.iloc[i, 1]: ', semantic_tags.iloc[i, 1])
                print('semantic_tags.iloc[i, 0]: ', semantic_tags.iloc[i, 0])
                '''
                
                weights_unique = []
                for label in segments_label:
                    if label in unknown_obstacles_vals:
                        #print('label pass: ', label)
                        continue
                    for j in range(0, len(self.exp)):
                        if self.exp[j][0] == label:
                            rounded = str(round(self.exp[j][1], 4))
                            if rounded not in weights_unique:
                                '''
                                print('label:', label)
                                print('rounded: ', rounded)
                                print('weights_unique: ', weights_unique)
                                '''
                                weights_unique.append(rounded)        
                            break
                #print('weights_unique: ', weights_unique)        
                if len(weights_unique) == 1:        
                    print(semantic_tags.iloc[i, 1] + " has a weight " + weights_unique[0])
                else:
                    weights_string = ""
                    for j in range(0, len(weights_unique) - 1):
                        weights_string += weights_unique[j] + ", "
                    if len(weights_unique) > 0:    
                        weights_string += weights_unique[-1] + " "    
                        print(semantic_tags.iloc[i, 1] + " has weights " + weights_string)

            
    # flip matrix horizontally or vertically
    def matrixFlip(self, m, d):
        myl = np.array(m)
        if d == 'v':
            return np.flip(myl, axis=0)
        elif d == 'h':
            return np.flip(myl, axis=1)

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

    # convert euler angles to orientation quaternion
    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        #qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        #qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        
        return [qx, qy, qz, qw]

    # save data for local planner in lime image
    def limeImageSaveDataForLocalPlanner(self):
        if self.manual_instance_loading == False:
            # Take original command speed
            self.cmd_vel_original_tmp = self.cmd_vel_original.iloc[self.index, :]
            self.cmd_vel_original_tmp = pd.DataFrame(self.cmd_vel_original_tmp).transpose()
            self.cmd_vel_original_tmp = self.cmd_vel_original_tmp.iloc[:, 2:]

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

        elif self.manual_instance_loading == True:
            # Saving data to .csv files for C++ node - local navigation planner
            # Save footprint instance to a file
            self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/global_plan.csv', index=False,
                                        header=False)

            # Save costmap_info instance to file
            self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False,
                                        header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False,
                                        header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False,
                                        header=False)

            # Save odometry instance to file
            self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

    # Saving important data to class variables
    def saveImportantData2ClassVars(self):
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
        
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)

        # save indices of footprint's poses in local costmap to class variables
        self.footprint_x_list = []
        self.footprint_y_list = []
        for j in range(0, self.footprint_tmp.shape[0]):
            self.footprint_x_list.append(int((self.footprint_tmp.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.footprint_y_list.append(int((self.footprint_tmp.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

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

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)

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

        self.plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            self.plan_tmp_tmp.iloc[i, 0] = pnew[0]
            self.plan_tmp_tmp.iloc[i, 1] = pnew[1]
            self.plan_tmp_tmp.iloc[i, 2] = pnew[2]
    
        # Get coordinates of the global plan in the local costmap
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, self.plan_tmp_tmp.shape[0], 3):
            x_temp = int((self.plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((self.plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
                    

    # turn inflated costmap to static costmap
    def inflatedToFree(self):
        # Turn inflated area to free space and 100s to 99s
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                if 99 > self.image[i, j] > 0:
                    self.image[i, j] = 0
                elif self.image[i, j] == 100:
                    self.image[i, j] = 99


    def testSegmentation(self, expID):

        print('Test segmentation function beginning')

        if self.manual_instance_loading == False:
            # Get local costmap
            index = expID
            # Original costmap will be saved to self.local_costmap_original
            local_costmap_original = self.costmap_data.iloc[(index) * self.costmap_size:(index + 1) * self.costmap_size, :]

            # Make image a np.array deepcopy of local_costmap_original
            image = np.array(copy.deepcopy(local_costmap_original))

        else:
            image = self.image    

        # Turn inflated area to free space and 100s to 99s
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if 99 > image[i, j] > 0:
                    image[i, j] = 0
                elif image[i, j] == 100:
                    image[i, j] = 99

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        image = image * 1.0

        # Make image a np.array deepcopy of local_costmap_original
        img_ = copy.deepcopy(image)

        #'''
        # Save local costmap as gray image
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_, aspect='auto')
        fig.savefig('costmap.png')
        fig.clf()
        #'''

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
        #segments = felzenszwalb(rgb, scale=100, sigma=5, min_size=1500, multichannel=True)
        #segments = felzenszwalb(rgb, scale=1, sigma=0.8, min_size=20, multichannel=True)  # default

        # quickshift
        #segments = quickshift(rgb, ratio=1.0, kernel_size=8, max_dist=800, return_tree=False, sigma=0.0, convert2lab=True, random_seed=42)
        #segments = quickshift(rgb, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, random_seed=42) # default

        # slic
        segments = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False, start_label=None, mask=None)
        #segments = slic(rgb, n_segments=100, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=None, mask=None) # default

        # Turn segments gray image to rgb image
        #segments_rgb = gray2rgb(segments)

        # Save segments to .csv file
        #pd.DataFrame(segments).to_csv('~/amar_ws/segments_segmentation_test.csv', index=False, header=False)

        #'''
        # Save local costmap as rgb image
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments, aspect='auto')
        fig.savefig('segments.png')
        fig.clf()
        #'''

        print('Test segmentation function ending')    
            


    def explain_instance_dataset(self, expID, iteration_ID):
        print('explain_instance_dataset function starting\n')

        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.expID = expID
            self.index = expID

            self.manual_instance_loading = False
            self.manually_make_semantic_map = False
            self.test_segmentation = False 

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

            # Make image a np.array deepcopy of local_costmap_original
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            # Turn inflated area to free space and 100s to 99s
            self.inflatedToFree()

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0
            
            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveDataForLocalPlanner()

            # Saving important data to class variables
            self.saveImportantData2ClassVars()

            # Use new variable in the algorithm - possible time saving
            img = copy.deepcopy(self.image)

            segm_fn = 'custom_segmentation'

            #devDistance_x, sum_x, devDistance_y, sum_y, devDistance = self.findDevDistance()
            devDistance_x = 0
            sum_x = 0 
            devDistance_y = 0 
            sum_y = 0 
            devDistance = 0

            self.explanation, self.segments = self.explainer.explain_instance(img, self.classifier_fn_image, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                                                                self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                                                                                self.plan_x_list, self.plan_y_list,
                                                                                hide_color=perturb_hide_color_value, batch_size=2048, segmentation_fn=segm_fn, top_labels=10)
                    
            self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=False, negative_only=False, num_features=100,
                                                                        hide_rest=False, min_weight=0.0)            
            
            self.plotMinimalDataset(iteration_ID, self.segments)
            
    def plotMinimalDataset(self, iteration_ID, segments):
        path_core = os.getcwd()

        # import needed libraries
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # plot costmap small
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        gray_shade = 180
        white_shade = 255
        image = gray2rgb(self.image)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = gray_shade
                elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = white_shade    
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/small/' + str(iteration_ID) + '_costmap.png')
        fig.clf()

        # plot costmap big
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)    
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/big/' + str(iteration_ID) + '_costmap.png')
        fig.clf()

        # plot input big
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.imshow(self.image, aspect='auto')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/big/' + str(iteration_ID) + '_input.png')
        fig.clf()

        # plot input small
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.imshow(self.image, aspect='auto')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/small/' + str(iteration_ID) + '_input.png')
        fig.clf()

        # plot input segmented small
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.imshow(self.image, aspect='auto')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(self.segments.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/small/' + str(iteration_ID) + '_input_segmented.png')
        fig.clf()

        # plot input segmented big
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.imshow(self.image, aspect='auto')
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(self.segments.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/big/' + str(iteration_ID) + '_input_segmented.png')
        fig.clf()

        # plot explanation small
        fig = plt.figure(frameon=True)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)       
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto') 
        fig.savefig(path_core + '/small/' + str(iteration_ID) + '_output.png', transparent=False)
        fig.clf()

        # plot explanation big
        fig = plt.figure(frameon=True)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)        
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/big/' + str(iteration_ID) + '_output.png', transparent=False)
        fig.clf()

        # plot weighted segments small
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments, aspect='auto')
        self.segments += 1
        regions = regionprops(self.segments.astype(int))
        labels = []
        for props in regions:
            labels.append(props.label)
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            ax.scatter(cy, cx, c='white', marker='o')   
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == v - 1:
                    ax.text(cy, cx, str(round(self.exp[j][1], 4)))
                    break
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig(path_core + '/small/' + str(iteration_ID) + '_weighted_segments.png')
        fig.clf()

        # plot weighted segments big
        fig = plt.figure(frameon=False)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments, aspect='auto')
        self.segments += 1
        regions = regionprops(self.segments.astype(int))
        labels = []
        for props in regions:
            labels.append(props.label)
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            ax.scatter(cy, cx, c='white', marker='o')   
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == v - 1:
                    ax.text(cy, cx, str(round(self.exp[j][1], 4)))
                    break
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig(path_core + '/big/' + str(iteration_ID) + '_weighted_segments.png')
        fig.clf()
                        
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:        
                with open('local_plan_coordinates.csv', "a") as myfile:
                     myfile.write(str(iteration_ID) + ',' + str(self.local_plan_tmp.iloc[i, 0]) + ',' + str(self.local_plan_tmp.iloc[i, 1]) + '\n')


        for i in range(0, self.plan_tmp_tmp.shape[0], 3):
            x_temp = int((self.plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                with open('global_plan_coordinates.csv', "a") as myfile:
                    myfile.write(str(iteration_ID) + ',' + str(self.plan_tmp_tmp.iloc[i, 0]) + ',' + str(self.plan_tmp_tmp.iloc[i, 1]) + '\n')
       
        with open('costmap_data.csv', "a") as myfile:
                #myfile.write('picture_ID,width,heigth,origin_x,origin_y,resolution\n')
                myfile.write(str(iteration_ID) + ',' + str(self.localCostmapWidth) + ',' + str(self.localCostmapHeight) + ',' + str(self.localCostmapOriginX) + ',' + str(self.localCostmapOriginY) + ',' + str(self.localCostmapResolution) + '\n')

        with open('robot_coordinates.csv', "a") as myfile:
                #myfile.write('picture_ID,position_x,position_y\n')
                myfile.write(str(iteration_ID) + ',' + str(self.odom_x) + ',' + str(self.odom_y) + '\n')       
        


    def explain_instance_evaluation(self, expID, ID):
        print('explain_instance_evaluation function starting\n')

        self.expID = expID
            
        # if explanation_mode is 'image'
        if self.explanation_mode == 'image':
            self.index = self.expID

            self.manual_instance_loading = False
            self.manually_make_semantic_map = False
            self.test_segmentation = False 

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

            # Make image a np.array deepcopy of local_costmap_original
            self.image = np.array(copy.deepcopy(self.local_costmap_original))

            # Turn inflated area to free space and 100s to 99s
            self.inflatedToFree()

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
            self.image = self.image * 1.0
            
            # Saving data to .csv files for C++ node - local navigation planner
            self.limeImageSaveDataForLocalPlanner()

            # Saving important data to class variables
            self.saveImportantData2ClassVars()

            # Use new variable in the algorithm - possible time saving
            img = copy.deepcopy(self.image)

            segm_fn = 'custom_segmentation'

            #devDistance_x, sum_x, devDistance_y, sum_y, devDistance = self.findDevDistance()
            devDistance_x = 0
            sum_x = 0 
            devDistance_y = 0 
            sum_y = 0 
            devDistance = 0

            #samples_num, segments_num = self.calculateNumOfSamples(img)
            segments_num = 8
 
            import time

            #with open('explanations' + str(self.expID) + '.csv', "w") as myfile:
            with open('explanations' + str(ID) + '.csv', "w") as myfile:
                myfile.write('num_samples,segmentation_time,classifier_fn_time,planner_time,explanation_time,explanation_pics_time,plotting_time,weight_0,weight_1,weight_2,weight_3,weight_4,weight_5\n')
                
                for i in range(0, segments_num + 1):
                    num_of_iterations_for_one_num_of_segments = 50 #30 #50
                    
                    for j in range(0, num_of_iterations_for_one_num_of_segments): 
                        # measure explain_instance time
                        start = time.time()
                        self.explanation, self.segments, segmentation_time, classifier_fn_time, planner_time = self.explainer.explain_instance_evaluation(
                            img, self.classifier_fn_image_evaluation, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                            self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                            self.plan_x_list, self.plan_y_list,
                            hide_color=perturb_hide_color_value,
                            num_segments=segments_num,
                            num_segments_current=i,
                            batch_size=2048, segmentation_fn=segm_fn,
                            top_labels=10)
                        end = time.time()
                        explanation_time = round(end - start, 3)
                        # print('Explanation time: ', explanation_time)

                        # measure get explanation_picture time
                        start = time.time()
                        self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0,
                                                                                                positive_only=False,
                                                                                                negative_only=False,
                                                                                                num_features=100,
                                                                                                hide_rest=False,
                                                                                                min_weight=0.0)  # min_weight=0.1 - default
                        end = time.time()
                        explanation_pics_time = round(end - start, 3)
                        # print('Getting explanation pics time: ', explanation_pics_time)

                        # measure plotting time
                        self.segments += 1
                        start = time.time()
                        self.plotMinimalEvaluation(2 ** i)
                        end = time.time()
                        plotting_time = round(end - start, 3)
                        # print('Plotting time: ', round(plotting_time, 3))

                        # round measured times to 3 decimal places
                        segmentation_time = round(segmentation_time, 3)
                        classifier_fn_time = round(classifier_fn_time, 3)
                        planner_time = round(planner_time, 3)

                        # write measured times to .csv file
                        myfile.write(
                            str(2 ** i) + ',' + str(segmentation_time) + ',' + str(classifier_fn_time) + ',' + str(
                                planner_time) + ',' + str(explanation_time) + ',' + str(
                                explanation_pics_time) + ',' + str(plotting_time) + ',')
                        for k in range(0, len(self.exp)):
                            for l in range(0, len(self.exp)):
                                if k == self.exp[l][0]:
                                    if k != len(self.exp) - 1:
                                        myfile.write(str(round(self.exp[l][1], 4)) + ',')
                                    else:
                                        myfile.write(str(round(self.exp[l][1], 4)))
                                    break
                        myfile.write('\n')
                    myfile.write('\n')
            
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

        #print('classifier_fn_image started')

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

        #print('starting C++ node')

        start = time.time()

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        end = time.time()
        planner_time = end - start

        #rospy.sleep(1)

        #print('C++ node ended')

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

        # fill the list of the original local plan coordinates
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
        #closest_to_robot_index = -100
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
                    #closest_to_robot_index = len(transformed_plan_xs) - 1
        transformed_plan_xs = np.array(transformed_plan_xs)
        transformed_plan_ys = np.array(transformed_plan_ys)

        # DETERMINE THE DEVIATION TYPE
        # thresholds
        local_plan_gap_threshold = 48 #60 #48 #32
        small_deviation_threshold = 7.0 #5 #7
        big_deviation_threshold = 14
        no_deviation_threshold = 3.0

        # test for the original local plan gap
        local_plan_original_gap = False
        local_plan_gaps = []
        diff = 0
        for j in range(0, len(local_plan_xs_orig) - 1):
            diff = math.sqrt( (local_plan_xs_orig[j]-local_plan_xs_orig[j+1])**2 + (local_plan_ys_orig[j]-local_plan_ys_orig[j+1])**2 )
            local_plan_gaps.append(diff)
        if max(local_plan_gaps) > local_plan_gap_threshold:
            local_plan_original_gap = True

        # local gap too big - stop
        if local_plan_original_gap == True or len(local_plan_xs_orig) == 0:
            deviation_type = 'stop'
            local_plan_gap_threshold = 55
        # no local gap - test further    
        else:        
            diff_x = 0
            diff_y = 0
            
            big_deviation = False
            devs = []
            for j in range( 0, len(local_plan_xs_orig)):
                diffs = []
                deviation_local = True  
                for k in range(0, len(transformed_plan_xs)):
                    diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    diffs.append(diff)
                    if diff <= big_deviation_threshold:
                        deviation_local = False
                        break # commented (comment out) because of big_deviation_threshold = max(devs) * 0.8
                devs.append(min(diffs))    
                if deviation_local == True:
                    big_deviation = True
                    break # commented (comment out) because of big_deviation_threshold = max(devs) * 0.8
            
            if big_deviation == True:
                deviation_type = 'big_deviation'
                local_plan_gap_threshold = 48
                #print('max_dev: ', max(devs))
                big_deviation_threshold = max(devs) * 0.8
                #print('big_deviation_threshold: ', big_deviation_threshold)
            else:
                diff_x = 0
                diff_y = 0
            
                small_deviation = False
                for j in range( 0, len(local_plan_xs_orig)):
                    deviation_local = True  
                    for k in range(0, len(transformed_plan_xs)):
                        diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        if diff <= small_deviation_threshold:
                            deviation_local = False
                            break
                    if deviation_local == True:
                        small_deviation = True
                        break
                if small_deviation == True:
                    deviation_type = 'small_deviation'
                    #print('max_dev: ', max(devs))
                else:
                    deviation_type = 'no_deviation'
                    #print('max_dev: ', max(devs))

        print('deviation_type: ', deviation_type)            

        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        # MAIN PART
        # fill in deviation dataframe
        dev_original = 0
        # find if there is local plan
        for i in range(0, sampled_instance.shape[0]):
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

            if local_plan_found == False:
                if deviation_type == 'stop':
                    self.local_plan_deviation.iloc[i, 0] = dev_original
                elif deviation_type == 'no_deviation':
                    self.local_plan_deviation.iloc[i, 0] = 100
                elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                    self.local_plan_deviation.iloc[i, 0] = 0.0
                continue             

            diff_x = 0
            diff_y = 0
            devs = []
            for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                local_diffs = []
                deviation_local = True  
                for k in range(0, len(transformed_plan_xs)):
                    diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))

            if i == 0:
                dev_original = max(devs)    

            self.local_plan_deviation.iloc[i, 0] = max(devs)

        
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

        return np.array(self.cmd_vel_perturb.iloc[:, 3:]), planner_time

    def plotMinimalEvaluation(self, num_samples):
        path_core = os.getcwd()

        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')

        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/explanation_' + str(num_samples) + '.png', transparent=False)
        fig.clf()
        #fig.close()

        '''
        # plot segments
        fig = plt.figure(frameon=True)
        w = 1.6*3
        h = 1.6*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments.astype(np.uint8), aspect='auto')  # , aspect='auto')
        fig.savefig(path_core + '/segments_' + str(num_samples) + '.png', transparent=False)
        fig.clf()
        '''


    def getSegmentsForGanLimeEval(self, image):

        print('Test segmentation function beginning')

        # Make image a np.array deepcopy of local_costmap_original
        img_ = copy.deepcopy(image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # Generate segments - superpixels with my slic function
        segments = self.mySlicGanLimeEval(rgb)

        print('Test segmentation function ending')

        return segments

    def mySlicGanLimeEval(self, img_rgb):

        print('mySlic for evaluation starts')

        img = img_rgb[:, :, 0]

        # import needed libraries
        from skimage.segmentation import slic

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


    def classifier_fn_tabular(self, sampled_instance):

        print('classifier_fn_tabular started')

        # Save sampled_instance to self.sampled_instance
        self.sampled_instance = sampled_instance

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

        # Save costmap_data to file
        self.costmap_tmp = self.costmap_data.iloc[(self.index) * 160:(self.index + 1) * self.costmap_size,:]
        self.costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)

        # Save costmap_info instance to file
        self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
        self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
        self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
        self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose instance to file
        self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
        self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
        self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
        self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map instance to file
        self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
        self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
        self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom instance to file
        self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
        self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
        self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save sampled odom to file
        self.odom_tmp = pd.concat([self.odom.iloc[self.index, :], self.odom.iloc[self.index, :]], join='outer', axis=1, sort=False)
        for i in range(0, self.num_samples - 2):
            self.odom_tmp = pd.concat([self.odom_tmp, self.odom.iloc[self.index, :]], join='outer', axis=1, sort=False)
        self.odom_tmp = self.odom_tmp.transpose()
        self.odom_tmp = self.odom_tmp.iloc[:, 2:]
        for i in range(0, self.num_samples):
            self.odom_tmp.iloc[i, -2] = sampled_instance[i, 0]
            self.odom_tmp.iloc[i, -1] = sampled_instance[i, 1]

        self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        print('\nstarting C++ node')

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_tabular'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_tabular/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_tabular'))

        #rospy.sleep(1)

        print('\nC++ node ended')

        cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        print('cmd_vel: ', cmd_vel)

        print('classifier_fn_tabular ended')

        return np.array(cmd_vel.iloc[:, 2])  # srediti index ovdje

    def classifier_fn_tabular_costmap(self, sampled_instance):

        print('classifier_fn_tabular_costmap started')
        print("Broj perturbacija: ", sampled_instance.shape[0])

        # Save sampled_instance to self.sampled_instance
        self.sampled_instance = sampled_instance

        # Save footprint instance to a file
        self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
        self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
        self.footprint_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

        # Save local plan instance to a file
        self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
        self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
        self.local_plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

        # Save plan instance to a file
        self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
        self.plan_tmp = self.plan_tmp.iloc[:, 1:]
        self.plan_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

        # Save costmap_info to file
        self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
        self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
        self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
        self.costmap_info_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_info.csv', index=False, header=False)

        # Save amcl_pose to file
        self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
        self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
        self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
        self.amcl_pose_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map to file
        self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
        self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
        self.tf_odom_map_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False, header=False)

        # Save tf_map_odom to file
        self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
        self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
        self.tf_map_odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False, header=False)

        # Save sampled odom to file
        self.odom_tmp = self.odom.iloc[self.index, :]
        self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
        self.odom_tmp = self.odom_tmp.iloc[:, 2:]
        self.odom_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        # Save costmap_data to file
        self.costmap_tmp = pd.DataFrame(sampled_instance[0][0:self.costmap_size]).transpose()
        for j in range(1, self.costmap_size):
            self.costmap_tmp = pd.concat([self.costmap_tmp, pd.DataFrame(sampled_instance[0][self.costmap_size * j: self.costmap_size * (j + 1)]).transpose()], join='outer', axis=0, sort=False)
        for i in range(1, self.num_samples):
            if i % 100 == 0:
                print('Current sample: ', i)
            img = pd.DataFrame(sampled_instance[i][0:160]).transpose()
            for j in range(1, 160):
                img = pd.concat([img, pd.DataFrame(sampled_instance[i][self.costmap_size * j: self.costmap_size * (j + 1)]).transpose()], join='outer', axis=0, sort=False)
            self.costmap_tmp = pd.concat([self.costmap_tmp, img], join='outer', axis=0, sort=False)
        print('self.costmap_tmp.shape: ', self.costmap_tmp.shape)
        self.costmap_tmp.to_csv('~/amar_ws/src/teb_local_planner/src/Data/costmap_data.csv', index=False, header=False)

        print('\nstarting C++ node')

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_tabular_costmap'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_tabular_costmap/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_tabular_costmap'))

        #rospy.sleep(1)

        print('\nC++ node ended')

        cmd_vel = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        print('cmd_vel: ', cmd_vel)

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
'''
'''
process = launch.launch(node)
print(process.is_alive())
# Ovdje uvesti ros services
rospy.sleep(20)
process.stop()
# stop ROS node
node_process.terminate()
# finish killing node proces
node_process.terminate()
'''
#node_process = Popen(shlex.split('rosnode kill /perturb_node_image'))
       