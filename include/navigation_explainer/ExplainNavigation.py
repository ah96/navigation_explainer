#!/usr/bin/env python3

# import time tabular
#import lime
#import lime.lime_tabular

#import shap

import time

import os

# lime image - my implementation
from navigation_explainer import lime_image

# anchor_image - my implementation
from navigation_explainer import anchor_image

# for managing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculations
import math
import copy

# for running ROS node
import shlex
from psutil import Popen
import rospy

# for managing image
from skimage import *
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries, felzenszwalb, slic, quickshift
from skimage.measure import regionprops

from navigation_explainer import traj_dist

# important global variables
perturb_hide_color_value = 0 #0 #50

class ExplainRobotNavigation:
    # constructor
    def __init__(self, cmd_vel, odom, plan, global_plan, local_plan, current_goal, local_costmap_data,
                 local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, underlying_model_mode, explanation_mode, explanation_alg,
                 num_of_first_rows_to_delete, footprints, output_class_name, X_train, X_test, y_train, y_test, num_samples, plot=True):
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
        self.tabular_mode = underlying_model_mode
        self.underlying_model_mode = underlying_model_mode
        self.explanation_mode = explanation_mode
        self.explanation_algorithm = explanation_alg
        self.offset = num_of_first_rows_to_delete
        self.footprints = footprints
        self.costmap_size = local_costmap_info.iloc[0, 2]
        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test
        self.num_samples = num_samples
        self.output_class_name = output_class_name
        self.plot = plot

        self.dirCurr = os.getcwd()

        if self.explanation_algorithm == 'LIME':
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

        elif self.explanation_algorithm == 'Anchors':
            # manually modified Anchors Image
            if self.explanation_mode == 'image':
                self.explainer = anchor_image.AnchorImage( distribution_path=None, transform_img_fn=None, 
                                                           n=1000, dummys=None, white=None, segmentation_fn=None)

            elif self.explanation_mode == 'tabular':
                pass
              
            elif self.explanation_mode == 'tabular_costmap':
                pass

        elif self.explanation_algorithm == 'SHAP':
            if self.explanation_mode == 'image':
                pass

            elif self.explanation_mode == 'tabular':
                pass
              
            elif self.explanation_mode == 'tabular_costmap':
                pass

        
        print('\nConstructor ending')

    # SINGLE EXPLANATION
    # explain instance
    def explain_instance(self, expID):
        print('\nexplain_instance starting')

        self.expID = expID
        self.index = self.expID

        self.manual_instance_loading = False        

        if self.explanation_algorithm == 'LIME':    
            # if explanation_mode is 'image'
            if self.explanation_mode == 'image':
                import time
                #before_explain_instance_start = time.time()

                if self.manual_instance_loading == False:
                    # Get local costmap
                    # Original costmap will be saved to self.local_costmap_original
                    self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

                    # Make image a np.array deepcopy of local_costmap_original
                    self.image = np.array(copy.deepcopy(self.local_costmap_original))

                    #inflated_to_static_start = time.time()
                    # Turn inflated area to free space and 100s to 99s
                    self.inflatedToStatic()
                    #inflated_to_static_end = time.time()
                    #print('\ninflated_to_static_time: ', inflated_to_static_end - inflated_to_static_start)

                    # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
                    self.image = self.image * 1.0
                    #print('self.image.size = ', self.image.size)

                elif self.manual_instance_loading == True:
                    manual_case_names = ['big_deviation','big_deviation_without_wall','no_deviation','rotate_in_place','small_deviation','stop']
                    idx = 3
                    manual_case_name = manual_case_names[idx]
                    path_to_files = self.dirCurr + '/manual_instances/' + manual_case_name + '/'

                    # Load costmap
                    self.image = np.array(pd.read_csv(path_to_files + 'costmap_new.csv')) * 1.0

                    # Load footprint
                    self.footprint_tmp = pd.read_csv(path_to_files + 'footprint_new.csv')
                    #print(self.footprint_tmp)
                    
                    # Load local plan
                    self.local_plan_tmp = pd.read_csv(path_to_files + 'local_plan_new.csv')

                    # Load plan (from global planner)
                    self.plan_tmp = pd.read_csv(path_to_files + 'global_plan_new.csv')

                    # Load global plan
                    self.global_plan_tmp = pd.read_csv(path_to_files + 'global_plan_new.csv')

                    # Load costmap_info
                    self.costmap_info_tmp = pd.read_csv(path_to_files + 'costmap_info_new.csv')

                    # Load amcl_pose
                    self.amcl_pose_tmp = pd.read_csv(path_to_files + 'amcl_pose_new.csv')

                    # Load tf_odom_map
                    self.tf_odom_map_tmp = pd.read_csv(path_to_files + 'tf_odom_map_new.csv')

                    # Load tf_map_odom
                    self.tf_map_odom_tmp = pd.read_csv(path_to_files + 'tf_map_odom_new.csv')

                    # Load odometry
                    self.odom_tmp = pd.read_csv(path_to_files + 'odom_new.csv')
                
                #save_data_for_local_planner_start = time.time()
                # Saving data to .csv files for C++ node - local navigation planner
                self.SaveImageDataForLocalPlanner()
                #save_data_for_local_planner_end = time.time()
                #save_data_for_local_planner_time = save_data_for_local_planner_end - save_data_for_local_planner_start
                
                # Saving important data to class variables
                self.saveImportantData2ClassVars()

                # Choose semantic or standard segmentation
                segm_fn = 'custom_segmentation'

                #before_explain_instance_end = time.time()
                #before_explain_instance_time = before_explain_instance_end - before_explain_instance_start
                #print('\nsave_data_for_local_planner_time / before_explain_instance_time (%) = ', 100 * save_data_for_local_planner_time / before_explain_instance_time)

                # get data needed for a special segmentation method
                devDistance_x = 0 
                sum_x = 0
                devDistance_y = 0
                sum_y = 0
                devDistance = 0

                # explain with LIME    
                import time
                #real_explanation_start = time.time()
                self.explanation, self.segments = self.explainer.explain_instance(self.image, self.classifier_fn_image_lime, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                                                            self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                                                                            self.plan_x_list, self.plan_y_list,
                                                                            hide_color=perturb_hide_color_value, batch_size=2048, segmentation_fn=segm_fn, top_labels=10)
                #real_explanation_end = time.time()
                #real_explanation_time = real_explanation_end - real_explanation_start
                #print('\nReal (pure) explanation time = ', real_explanation_time)

                # get explanation image
                self.temp_img, self.exp = self.explanation.get_image_and_mask(label=0)
                #print('self.exp = ', self.exp)

                #self.plot = True
                if self.plot == True:
                    plotting_time_start = time.time()
                    self.plotExplanation()
                    plotting_time_end = time.time()
                    plotting_time_start = plotting_time_end - plotting_time_start
                    print('\nPlotting time = ', plotting_time_start)

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
        
        elif self.explanation_algorithm == 'Anchors':
            # manually modified Anchors Image
            if self.explanation_mode == 'image':

                if self.manual_instance_loading == False:
                    # Get local costmap
                    # Original costmap will be saved to self.local_costmap_original
                    self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

                    # Make image a np.array deepcopy of local_costmap_original
                    self.image = np.array(copy.deepcopy(self.local_costmap_original))

                    # Turn inflated area to free space and 100s to 99s
                    self.inflatedToStatic()

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
                    self.plan_tmp = pd.read_csv('global_plan_new.csv')

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
                self.SaveImageDataForLocalPlanner()
                
                # Saving important data to class variables
                self.saveImportantData2ClassVars()

                # Use new variable in the algorithm - possible time saving
                #img = copy.deepcopy(self.image)

                if self.semantic_seg == True:
                    segm_fn = 'semantic_segmentation'
                elif self.semantic_seg == False:
                    segm_fn = 'custom_segmentation'

                img = gray2rgb(self.image)

                # get data needed for a particular segmentation method
                devDistance_x = 0 
                sum_x = 0 
                devDistance_y = 0 
                sum_y = 0
                devDistance = 0
                
                self.segments, self.exp, self.best_tuples = self.explainer.explain_instance(img, self.classifier_fn_image_anchors, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                                                                    self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, 
                                                                                    devDistance_y, sum_y, devDistance, self.plan_x_list, self.plan_y_list,
                                                                                    threshold=0.95, delta=0.1, tau=0.15, batch_size=256)

                print('\nself.explanation = ', self.exp)
                print('\nbest_tuples = ', self.best_tuples)

                self.plotExplanationAnchors()
                
            elif self.explanation_mode == 'tabular':
                pass
              
            elif self.explanation_mode == 'tabular_costmap':
                pass
            
        elif self.explanation_algorithm == 'SHAP':
            if self.explanation_mode == 'image':
                pass

            elif self.explanation_mode == 'tabular':
                pass
              
            elif self.explanation_mode == 'tabular_costmap':
                pass        

        print('\nexplain_instance ending')

    # save data for local planner in explanation with image
    def SaveImageDataForLocalPlanner(self):
        if self.manual_instance_loading == False:
            # Take original command speed
            self.cmd_vel_original_tmp = self.cmd_vel_original.iloc[self.index, :]
            self.cmd_vel_original_tmp = pd.DataFrame(self.cmd_vel_original_tmp).transpose()
            self.cmd_vel_original_tmp = self.cmd_vel_original_tmp.iloc[:, 2:]

            # Saving data to .csv files for C++ node - local navigation planner
            # Save footprint instance to a file
            self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
            self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
            self.footprint_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
            self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
            self.local_plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
            self.plan_tmp = self.plan_tmp.iloc[:, 1:]
            self.plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp = self.global_plan.loc[self.global_plan['ID'] == self.index + self.offset]
            self.global_plan_tmp = self.global_plan_tmp.iloc[:, 1:]
            self.global_plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/global_plan.csv', index=False,
                                        header=False)

            # Save costmap_info instance to file
            self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
            self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
            self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
            self.costmap_info_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_info.csv', index=False,
                                        header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
            self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
            self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
            self.amcl_pose_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
            self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
            self.tf_odom_map_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False,
                                        header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
            self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
            self.tf_map_odom_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False,
                                        header=False)

            # Save odometry instance to file
            self.odom_tmp = self.odom.iloc[self.index, :]
            self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
            self.odom_tmp = self.odom_tmp.iloc[:, 2:]
            self.odom_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

        elif self.manual_instance_loading == True:
            # Saving data to .csv files for C++ node - local navigation planner
            # Save footprint instance to a file
            self.footprint_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/footprint.csv', index=False, header=False)

            # Save local plan instance to a file
            self.local_plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plan.csv', index=False, header=False)

            # Save plan (from global planner) instance to a file
            self.plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/plan.csv', index=False, header=False)

            # Save global plan instance to a file
            self.global_plan_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/global_plan.csv', index=False,
                                        header=False)

            # Save costmap_info instance to file
            self.costmap_info_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_info.csv', index=False,
                                        header=False)

            # Save amcl_pose instance to file
            self.amcl_pose_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/amcl_pose.csv', index=False, header=False)

            # Save tf_odom_map instance to file
            self.tf_odom_map_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/tf_odom_map.csv', index=False,
                                        header=False)

            # Save tf_map_odom instance to file
            self.tf_map_odom_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/tf_map_odom.csv', index=False,
                                        header=False)

            # Save odometry instance to file
            self.odom_tmp.to_csv(self.dirCurr + '/src/teb_local_planner/src/Data/odom.csv', index=False, header=False)

    # saving important data to class variables
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

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.local_plan_x_list.append(x_temp)
                self.local_plan_y_list.append(y_temp)

        #'''
        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2]
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        #'''

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

        #self.plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        self.plan_tmp_tmp = pd.DataFrame(0.0, index=np.arange(self.global_plan_tmp.shape[0]), columns=self.global_plan_tmp.columns)
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
            y_temp = int((self.plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(y_temp)
                    
    # turn inflated costmap to static costmap
    def inflatedToStatic(self):
        #self.image[self.image < 99] = 0
        #self.image[self.image > 0] = 99
        self.image[self.image == 100] = 99
        self.image[self.image <= 98] = 0

    # plot explanation picture and segments
    def plotExplanation(self):
        try:
            if self.transformed_plan_xs == [] or self.transformed_plan_ys == []:
                # fill the list of transformed plan coordinates
                self.transformed_plan_xs = []
                self.transformed_plan_ys = []
                for i in range(0, self.transformed_plan.shape[0]):
                    x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        self.transformed_plan_xs.append(x_temp)
                        self.transformed_plan_ys.append(y_temp)
        except:
            # if transformed_plan variables do not exist
            # fill the list of transformed plan coordinates
                self.transformed_plan_xs = []
                self.transformed_plan_ys = []
                for i in range(0, self.transformed_plan.shape[0]):
                    x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        self.transformed_plan_xs.append(x_temp)
                        self.transformed_plan_ys.append(y_temp)

        dirName = 'explanation_results'
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass

        self.eps = False

        big_plot_size = True
        w = h = 1.6
        if big_plot_size == True:
            w *= 3
            h *= 3    

        # import needed libraries
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # plot costmap
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        free_space_shade = 180
        obstacle_shade = 0
        image = gray2rgb(self.image)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = free_space_shade
                elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = obstacle_shade    
        #pd.DataFrame(image[:,:,0]).to_csv('R.csv')
        #pd.DataFrame(image[:,:,1]).to_csv('G.csv')
        #pd.DataFrame(image[:,:,2]).to_csv('B.csv')            
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/costmap.png')
        fig.clf()

        # plot costmap with plans
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')
        #ax.imshow(self.image, aspect='auto')
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/input.png')
        if self.eps == True:
            fig.savefig(self.dirCurr + '/' + dirName + '/input.eps')
        fig.clf()

        # plot explanation
        fig = plt.figure(frameon=True)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        #ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        #print('len(self.transformed_plan_x_list): ', len(self.transformed_plan_xs))
        ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')

        #marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        #marked_boundaries = mark_boundaries(self.temp_img, self.mask, color=(180, 180, 180)) #color=(255, 255, 0)
        #ax.imshow(marked_boundaries.astype(np.uint8), aspect='auto')  # , aspect='auto')
        for i in range(0, self.temp_img.shape[0]):
            for j in range(0, self.temp_img.shape[1]):
                if self.image[i, j] == 99 and self.temp_img[i, j, 0] == self.temp_img[i, j, 1] == self.temp_img[i, j, 2] == 180:
                    self.temp_img[i, j, 0] = 0
                    self.temp_img[i, j, 1] = 0
                    self.temp_img[i, j, 2] = 0
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')  # , aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/explanation.png', transparent=False)
        if self.eps == True:
            fig.savefig(self.dirCurr + '/' + dirName + '/explanation.eps', transparent=False)
        fig.clf()
                        
        # plot segments with weights
        self.segments += 1
        regions = regionprops(self.segments.astype(int))
        labels = []
        for props in regions:
            labels.append(props.label)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)    
        for props in regions:
            v = props.label  # value of label
            if v > 1:
                cx, cy = props.centroid  # centroid coordinates
                ax.scatter(cy, cx, c='white', marker='o')   
                # printing/plotting explanation weights
                for j in range(0, len(self.exp)):
                    if self.exp[j][0] == v-1:
                        ax.text(cy, cx, str(round(self.exp[j][1], 4)) + ', ' + str(round(self.exp[j][0], 4)), c='black')  # str(round(self.exp[j][1],4)) #str(v))
                        break
        # Save segments with nice numbering as a picture
        ax.imshow(self.segments, aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/weighted_segments.png')
        fig.clf()
        self.segments -= 1                

        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/temp_img.png')
        fig.clf()

    # classifier function for lime image
    def classifier_fn_image_lime(self, sampled_instance):

        print('\nclassifier_fn_image_lime started')

        print('\nsampled_instance.shape = ', sampled_instance.shape)

        self.sampled_instance_shape_len = len(sampled_instance.shape)
        self.sample_size = 1 if self.sampled_instance_shape_len == 2 else sampled_instance.shape[0]

        # Save perturbed costmap_data to file for C++ node
        #costmap_save_start = time.time()

        if self.sampled_instance_shape_len > 3:
            temp = np.delete(sampled_instance,2,3)
            #print(temp.shape)
            temp = np.delete(temp,1,3)
            #print(temp.shape)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 2:
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")

        #costmap_save_end = time.time()
        #costmap_save_time = costmap_save_end - costmap_save_start
        #print('\nsave perturbed costmap_data runtime: ', costmap_save_time)

        # calling ROS C++ node
        #print('\nstarting C++ node')

        #planner_calculation_start = time.time()

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        #planner_calculation_end = time.time()
        #planner_calculation_time = planner_calculation_end - planner_calculation_start
        #print('\nplanner calculation runtime = ', planner_calculation_time)

        #rospy.sleep(1)

        #print('\nC++ node ended')

        # GETTING OUTPUT
        #output_start = time.time()
        # load command velocities - output from local planner
        self.cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('self.cmd_vel: ', self.cmd_vel_perturb)
        #print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)
        #self.cmd_vel_perturb.to_csv('cmd_vel.csv')

        # load local plans - output from local planner
        self.local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('self.local_plans: ', self.local_plans)
        #print('self.local_plans.shape: ', self.local_plans.shape)
        #self.local_plans.to_csv('local_plans.csv')

        # load transformed global plan to /odom frame
        self.transformed_plan = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('self.transformed_plan: ', self.transformed_plan)
        #print('self.transformed_plan.shape: ', self.transformed_plan.shape)
        #self.transformed_plan.to_csv('transformed_plan.csv')
        #output_end = time.time()
        #output_time = output_end - output_start
        #print('\noutput time: ', output_time)
  
        # calculate original deviation - sum of minimal point-to-point distances
        calculate_original_deviation = False
        if calculate_original_deviation == True:
            original_deviation = -1.0
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, len(self.local_plan_x_list)):
                local_diffs = []
                for k in range(0, len(self.transformed_plan_xs)):
                    diff_x = (self.local_plan_x_list[j] - self.transformed_plan_xs[k]) ** 2
                    diff_y = (self.local_plan_y_list[j] - self.transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))   
            original_deviation = sum(devs)
            # original_deviation for big_deviation = 745.5051688094327
            # original_deviation for big_deviation without wall = 336.53749938826286
            # original_deviation for no_deviation = 56.05455197218764
            # original_deviation for small_deviation = 69.0
            # original_deviation for rotate_in_place = 307.4962940090125
            print('\noriginal_deviation = ', original_deviation)

        plot_perturbations = False
        if plot_perturbations == True:
            # only needed for classifier_fn_image_plot() function
            self.sampled_instance = sampled_instance
            # plot perturbation of local costmap
            self.classifier_fn_image_lime_plot()


        # DETERMINE THE DEVIATION TYPE
        determine_dev_type = False
        if determine_dev_type == True:
            #start_determine_dev = time.time()
            
            # thresholds
            local_plan_gap_threshold = 48.0
            big_deviation_threshold = 85.0
            small_deviation_threshold = 32.0 #30
            no_deviation_threshold = 0.0

            # test for the original local plan gap
            local_plan_original_gap = False
            local_plan_gaps = []
            diff = 0
            for j in range(0, len(self.local_plan_x_list) - 1):
                diff = math.sqrt( (self.local_plan_x_list[j]-self.local_plan_x_list[j+1])**2 + (self.local_plan_y_list[j]-self.local_plan_y_list[j+1])**2 )
                local_plan_gaps.append(diff)
            if max(local_plan_gaps) > local_plan_gap_threshold:
                local_plan_original_gap = True

            # local gap too big - stop (rotate_in_place)
            if local_plan_original_gap == True:
                deviation_type = 'stop'
                local_plan_gap_threshold = 55.0
            
            # no local gap - test further    
            elif original_deviation >= big_deviation_threshold:
                deviation_type = 'big_deviation'
            elif original_deviation >= small_deviation_threshold:
                deviation_type = 'small_deviation'
            else:
                deviation_type = 'no_deviation'    

            #end_determine_dev = time.time()
            #determine_dev_time = end_determine_dev - start_determine_dev
            #print('\ndetermine deviation type runtime = ', determine_dev_time)

            # PRINTING RESULTS                                       
            print('\ndeviation_type: ', deviation_type)
  
        mode = self.underlying_model_mode # 'regression' or 'classification' or 'regression_normalized_around_deviation' or 'regression_normalized'
        #print('\nmode = ', mode)

        # TARGET CALCULATION
        target_calculation_start = time.time()
        my_distance_fun = True
        if my_distance_fun == True:
            # deviation of local plan from global plan dataframe
            self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])
            #print('self.local_plan_deviation: ', self.local_plan_deviation)

            #### MAIN TARGET CALCULATION PART ####

            print_iterations = False
            
            #start_main = time.time()

            if mode == 'regression':
                local_deviation_frame = 'costmap'
                local_deviation_metric = 'L1'
                
                if local_deviation_frame == 'pixels' and local_deviation_metric == 'L2':
                    # transform transformed_plan to pixel locations
                    #start_transformed = time.time()
                    self.transformed_plan_xs = []
                    self.transformed_plan_ys = []
                    #print('len(TRANSFORMED_PLAN_BEFORE) = ', self.transformed_plan.shape[0])
                    for i in range(0, self.transformed_plan.shape[0]):
                        x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        #if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        self.transformed_plan_xs.append(x_temp)
                        self.transformed_plan_ys.append(y_temp)
                    #end_transformed = time.time()
                    #transformed_time = end_transformed - start_transformed
                    #print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)
                    #print('len(TRANSFORMED_PLAN_AFTER) = ', len(self.transformed_plan_xs))                    

                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        local_plan_xs = []
                        local_plan_ys = []
                        
                        # transform local_plan to pixel locations
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        #print('len(LOCAL_PLAN_BEFORE) = ', self.local_plans_local.shape[0])
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                #if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                        #print('len(LOCAL_PLAN_AFTER) = ', len(local_plan_xs))

                        # find deviation as a sum of minimal point-to-point differences
                        diff_x = 0
                        diff_y = 0
                        devs = []
                        for j in range(0, len(local_plan_xs)):
                            local_diffs = []
                            for k in range(0, len(self.transformed_plan_xs)):
                                diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                diff = math.sqrt(diff_x + diff_y)
                                local_diffs.append(diff)                        
                            devs.append(min(local_diffs))   

                        self.local_plan_deviation.iloc[i, 0] = sum(devs)
                
                elif local_deviation_frame == 'costmap' and local_deviation_metric == 'L2':
                    # fill in deviation dataframe
                    # transform transformed_plan to list
                    start_transformed = time.time()
                    transformed_plan_xs = []
                    transformed_plan_ys = []
                    for i in range(0, self.transformed_plan.shape[0]):
                        transformed_plan_xs.append(self.transformed_plan.iloc[i, 0])
                        transformed_plan_ys.append(self.transformed_plan.iloc[i, 1])
                    end_transformed = time.time()
                    transformed_time = end_transformed - start_transformed
                    print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)
                        
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        #print('len(LOCAL_PLAN) = ', self.local_plans_local.shape[0])

                        local_plan_xs = []
                        local_plan_ys = []
                        
                        # transform local_plan to list
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                local_plan_xs.append(self.local_plans_local.iloc[j, 0])
                                local_plan_ys.append(self.local_plans_local.iloc[j, 1])
                        
                        # find deviation as a sum of minimal point-to-point differences
                        diff_x = 0
                        diff_y = 0
                        devs = []
                        for j in range(0, self.local_plans_local.shape[0]):
                            local_diffs = []
                            for k in range(0, len(self.transformed_plan)):
                                #diff_x = (self.local_plans_local.iloc[j, 0] - self.transformed_plan.iloc[k, 0]) ** 2
                                #diff_y = (self.local_plans_local.iloc[j, 1] - self.transformed_plan.iloc[k, 1]) ** 2
                                diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                                diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                                diff = math.sqrt(diff_x + diff_y)
                                local_diffs.append(diff)                        
                            devs.append(min(local_diffs))   

                        self.local_plan_deviation.iloc[i, 0] = sum(devs)

                elif local_deviation_frame == 'pixels' and local_deviation_metric == 'L1':
                    # transform transformed_plan to pixel locations
                    #start_transformed = time.time()
                    self.transformed_plan_xs = []
                    self.transformed_plan_ys = []
                    #print('len(TRANSFORMED_PLAN_BEFORE) = ', self.transformed_plan.shape[0])
                    for i in range(0, self.transformed_plan.shape[0]):
                        x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        #if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        self.transformed_plan_xs.append(x_temp)
                        self.transformed_plan_ys.append(y_temp)
                    #end_transformed = time.time()
                    #transformed_time = end_transformed - start_transformed
                    #print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)
                    #print('len(TRANSFORMED_PLAN_AFTER) = ', len(self.transformed_plan_xs))                    

                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        local_plan_xs = []
                        local_plan_ys = []
                        
                        # transform local_plan to pixel locations
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        #print('len(LOCAL_PLAN_BEFORE) = ', self.local_plans_local.shape[0])
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                #if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                        #print('len(LOCAL_PLAN_AFTER) = ', len(local_plan_xs))

                        # find deviation as a sum of minimal point-to-point differences
                        diff_x = 0
                        diff_y = 0
                        devs = []
                        for j in range(0, len(local_plan_xs)):
                            local_diffs = []
                            for k in range(0, len(self.transformed_plan_xs)):
                                diff_x = abs(local_plan_xs[j] - self.transformed_plan_xs[k])
                                diff_y = abs(local_plan_ys[j] - self.transformed_plan_ys[k])
                                diff = diff_x + diff_y
                                local_diffs.append(diff)                        
                            devs.append(min(local_diffs))   

                        self.local_plan_deviation.iloc[i, 0] = sum(devs)
                
                elif local_deviation_frame == 'costmap' and local_deviation_metric == 'L1':
                    # fill in deviation dataframe
                    # transform transformed_plan to list
                    start_transformed = time.time()
                    transformed_plan_xs = []
                    transformed_plan_ys = []
                    for i in range(0, self.transformed_plan.shape[0]):
                        transformed_plan_xs.append(self.transformed_plan.iloc[i, 0])
                        transformed_plan_ys.append(self.transformed_plan.iloc[i, 1])
                    end_transformed = time.time()
                    transformed_time = end_transformed - start_transformed
                    print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)
                        
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        #print('len(LOCAL_PLAN) = ', self.local_plans_local.shape[0])

                        local_plan_xs = []
                        local_plan_ys = []
                        
                        # transform local_plan to list
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                local_plan_xs.append(self.local_plans_local.iloc[j, 0])
                                local_plan_ys.append(self.local_plans_local.iloc[j, 1])
                        
                        # find deviation as a sum of minimal point-to-point differences
                        diff_x = 0
                        diff_y = 0
                        devs = []
                        for j in range(0, self.local_plans_local.shape[0]):
                            local_diffs = []
                            for k in range(0, len(self.transformed_plan)):
                                #diff_x = abs(self.local_plans_local.iloc[j, 0] - self.transformed_plan.iloc[k, 0]
                                #diff_y = abs(self.local_plans_local.iloc[j, 1] - self.transformed_plan.iloc[k, 1]
                                diff_x = abs(local_plan_xs[j] - transformed_plan_xs[k])
                                diff_y = abs(local_plan_ys[j] - transformed_plan_ys[k])
                                diff = diff_x + diff_y
                                local_diffs.append(diff)                        
                            devs.append(min(local_diffs))   

                        self.local_plan_deviation.iloc[i, 0] = sum(devs)

            elif mode == 'classification':
                if deviation_type == 'stop':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        # test if there is local plan
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(0, len(local_plan_xs)):
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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

                                    for j in range(0, len(x_indices)):
                                        for m in range(x_indices[j] - 0, x_indices[j] + 1):
                                            for q in range(y_indices[j] - 0, y_indices[j] + 1):
                                                if self.sampled_instance_shape_len == 3:
                                                    if sampled_instance[i, q, m] == 99:
                                                        local_plan_gap = True
                                                        break
                                                else:
                                                    if sampled_instance[q, m] == 99:
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
                    for i in range(0, self.sample_size): 
                        #print('i = ', i)
                        # test if there is local plan
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False    
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(len(local_plan_xs) - 1, len(local_plan_xs)):
                                if self.sample_size > 1:
                                    if sampled_instance[i][local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if self.sampled_instance_shape_len == 3:
                                        if sampled_instance[0, local_plan_ys[j], local_plan_xs[j]] == 99:
                                            local_plan_point_in_obstacle = True
                                            break
                                    else:
                                        if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)    

                                    if deviation >= big_deviation_threshold:
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
                                    #print('deviation: ', real_deviation)
                                    print('max dev: ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                    
                elif deviation_type == 'small_deviation':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('i = ', i)
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
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)
                                    
                                    if small_deviation_threshold <= deviation < big_deviation_threshold:
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
                                    #print('deviation: ', small_deviation)
                                    #print('minimal diff: ', min(diffs))
                                    print('max(devs): ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                    
                elif deviation_type == 'no_deviation':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('i = ', i)
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False
                        # if there is local plan test further    
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(0, len(local_plan_xs)):
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)
                                    
                                    if no_deviation_threshold <= deviation < small_deviation_threshold:
                                        self.local_plan_deviation.iloc[i, 0] = 1.0
                                    else:    
                                        self.local_plan_deviation.iloc[i, 0] = 0.0                
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
                                    #print('deviation: ', real_deviation)
                                    #print('minimal diff: ', min(diffs))
                                    print('max(devs): ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])

            elif mode == 'regression_normalized_around_deviation':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)    

                    deviation = sum(devs)
                    deviation = deviation / original_deviation if deviation <= original_deviation else (2*original_deviation - deviation) / original_deviation
                    if deviation < 0:
                        deviation = 0.0
                    self.local_plan_deviation.iloc[i, 0] = deviation

            elif mode == 'regression_normalized':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)    

                    self.local_plan_deviation.iloc[i, 0] = sum(devs) / original_deviation

        elif my_distance_fun == False:
            self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])

            if mode == 'regression':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    '''
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)
                    '''      

                    #t1 = np.column_stack((local_plan_xs, local_plan_ys))
                    #t1 = np.vstack((local_plan_xs, local_plan_ys))
                    #print('\nt1 = ', t1)
                    #print('\nt1.shape = ', t1.shape)
                    #t1 = np.fliplr(t1)
                    #print('\nt1.shape = ', t1.shape)
                    #t2 = np.column_stack((self.transformed_plan_xs, self.transformed_plan_ys))
                    #t2 = np.vstack((self.transformed_plan_xs, self.transformed_plan_ys))
                    #print('\nt2 = ', t2)
                    #print('\nt2.shape = ', t2.shape)
                    #min_len = min(t1.shape[1], t2.shape[1])
                    #dist = traj_dist.eucl_dist_traj(np.array(local_plan_xs), np.array(self.transformed_plan_xs)) + traj_dist.eucl_dist_traj(np.array(local_plan_ys), np.array(self.transformed_plan_ys))
                    
                    #dist = traj_dist.eucl_dist_traj(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.eucl_dist_traj(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D
                    #dist = traj_dist.e_hausdorff(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_hausdorff(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D
                    #dist = traj_dist.e_sspd(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_sspd(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D

                    #dist = traj_dist.discret_frechet(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.discret_frechet(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    #dist = traj_dist.e_dtw(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_dtw(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    #dist = traj_dist.e_edr(np.array(local_plan_xs), np.array(self.transformed_plan_xs), 0.01)**2 + traj_dist.e_edr(np.array(local_plan_ys), np.array(self.transformed_plan_ys), 0.01)**2
                    #dist = traj_dist.e_lcss(np.array(local_plan_xs), np.array(self.transformed_plan_xs), 0.01)**2 + traj_dist.e_lcss(np.array(local_plan_ys), np.array(self.transformed_plan_ys), 0.01)**2
                    dist = traj_dist.owd_grid_brut(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.owd_grid_brut(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    
                    dist = math.sqrt(dist)
                    self.local_plan_deviation.iloc[i, 0] = dist
                    
                    #dist = traj_dist.discret_frechet(t1[:,0:min_len], t2[:,0:min_len])
                    #print('\ndist_raw = ', self.local_plan_deviation.iloc[i, 0])
                    #self.local_plan_deviation.iloc[i, 0] = math.sqrt(dist[0,0]**2+dist[1,1]**2)
                    #print('\ndist = ', self.local_plan_deviation.iloc[i, 0])

        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        #print('self.local_plan_deviation = ', self.local_plan_deviation)
        #self.cmd_vel_perturb['deviate'].to_csv('deviations.csv')

        target_calculation_end = time.time()
        target_calculation_time = target_calculation_end - target_calculation_start
        print('\ntarget calculation runtime = ', target_calculation_time)

        # if more outputs wanted
        more_outputs = False
        if more_outputs == True:
            # classification
            print('\n(lin_x, ang_z) = ', (self.cmd_vel_perturb.iloc[0, 0], self.cmd_vel_perturb.iloc[0, 2]))
            
            # stop
            if abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                stop_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        stop_list.append(1.0)
                    else:
                        stop_list.append(0.0)
                self.cmd_vel_perturb['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['stop'])
            
            # ahead_straight
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                ahead_straight_list = [] = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        ahead_straight_list = [].append(1.0)
                    else:
                        ahead_straight_list = [].append(0.0)
                self.cmd_vel_perturb['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_straight'])

            # back_straight
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                back_straight_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        back_straight_list = [].append(1.0)
                    else:
                        back_straight_list = [].append(0.0)
                self.cmd_vel_perturb['back_straight'] = pd.DataFrame(np.array(back_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['back_straight'])
            
            # rotate_right_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_in_place_list.append(1.0)
                    else:
                        rotate_right_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_in_place'] = pd.DataFrame(np.array(rotate_right_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_in_place'])

            # rotate_left_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_in_place_list.append(1.0)
                    else:
                        rotate_left_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_in_place'] = pd.DataFrame(np.array(rotate_left_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_in_place'])

            # rotate_right_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_ahead_list.append(1.0)
                    else:
                        rotate_right_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_ahead'] = pd.DataFrame(np.array(rotate_right_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_ahead'])

            # rotate_left_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_ahead_list.append(1.0)
                    else:
                        rotate_left_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_ahead'] = pd.DataFrame(np.array(rotate_left_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_ahead'])

            # rotate_right_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_back_list.append(1.0)
                    else:
                        rotate_right_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_back'] = pd.DataFrame(np.array(rotate_right_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_back'])

            # rotate_left_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_back_list.append(1.0)
                    else:
                        rotate_left_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_back'] = pd.DataFrame(np.array(rotate_left_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_back'])

            print('\nrobot_decision = ', self.cmd_vel_perturb.columns.values[-1])

        print('\nclassifier_fn_image_lime ended\n')

        return np.array(self.cmd_vel_perturb.iloc[:, 3:])

    # function for plotting lime image perturbations
    def classifier_fn_image_lime_plot(self):

        dirName = 'perturbations'
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass

        #print('self.sample_size = ', self.sample_size)

        # plot every perturbation
        for ctr in range(0, self.sample_size):

            # save current perturbation as .csv file
            #pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('perturbation_' + str(i) + '.csv', index=False, header=False)

            free_space_shade = 180
            obstacle_shade = 0
            image = copy.deepcopy(self.sampled_instance[ctr][:, :])
            image = gray2rgb(image)
            #pd.DataFrame(image[:,:,0]).to_csv('IMAGE' + str(ctr) + '.csv')
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                        image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = free_space_shade
                    elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                        image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = obstacle_shade

            # plot perturbed local costmap
            #plt.imshow(self.sampled_instance[i][:, :])
            fig = plt.figure(frameon=True)
            w = 1.6*3
            h = 1.6*3
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image.astype(np.uint8))

            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []

            # find if there is local plan
            self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == ctr]
            for j in range(0, self.local_plans_local.shape[0]):
                    x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)
                        
                        '''
                        [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans_local.iloc[j, 2], self.local_plans_local.iloc[j, 3])
                        yaw_x = math.cos(yaw)
                        yaw_y = math.sin(yaw)
                        plt.quiver(x_temp, y_temp, yaw_y, yaw_x, color='white')
                        '''

            # plot transformed plan
            # fill the list of transformed plan coordinates
            #start_transformed = time.time()
            transformed_plan_xs = []
            transformed_plan_ys = []
            for i in range(0, self.transformed_plan.shape[0]):
                x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                    transformed_plan_xs.append(x_temp)
                    transformed_plan_ys.append(y_temp)
            #end_transformed = time.time()
            #transformed_time = end_transformed - start_transformed
            #print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)    
            plt.scatter(transformed_plan_xs, transformed_plan_ys, c='blue', marker='x')

            #'''
            # plot footprints for first five points of local plan
            # indices of local plan's poses in local costmap
            self.footprint_local_plan_x_list = []
            self.footprint_local_plan_y_list = []
            self.footprint_local_plan_x_list_angle = []
            self.footprint_local_plan_y_list_angle = []
            for j in range(0, self.local_plans_local.shape[0]):
                for k in range(6, 7):
                    [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans_local.iloc[j + k, 2], self.local_plans_local.iloc[j + k, 3])
                    sin_th = math.sin(yaw)
                    cos_th = math.cos(yaw)

                    for l in range(0, self.footprint_tmp.shape[0]):
                        x_new = self.footprint_tmp.iloc[l, 0] + (self.local_plans_local.iloc[j + k, 0] - self.odom_x)
                        y_new = self.footprint_tmp.iloc[l, 1] + (self.local_plans_local.iloc[j + k, 1] - self.odom_y)
                        self.footprint_local_plan_x_list.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                        self.footprint_local_plan_y_list.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))

                        x_new = self.local_plans_local.iloc[j + k, 0] + (self.footprint_tmp.iloc[l, 0] - self.odom_x) * sin_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * cos_th
                        y_new = self.local_plans_local.iloc[j + k, 1] - (self.footprint_tmp.iloc[l, 0] - self.odom_x) * cos_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * sin_th
                        self.footprint_local_plan_x_list_angle.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                        self.footprint_local_plan_y_list_angle.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))
                break
            # plot footprints
            print('self.footprint_local_plan_x_list = ', self.footprint_local_plan_x_list)
            plt.scatter(self.footprint_local_plan_x_list, self.footprint_local_plan_y_list, c='green', marker='x')
            plt.scatter(self.footprint_local_plan_x_list_angle, self.footprint_local_plan_y_list_angle, c='white', marker='x')
            #'''

            # plot local plan
            plt.scatter(local_plan_x_list, local_plan_y_list, c='yellow', marker='x')

            # plot robot's location and orientation
            #plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            #plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')

            # plot command velocities as text
            #plt.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_perturb.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_perturb.iloc[i, 2], 2)))

            #print('\nPERTURBATION_' + str(i))
            
            # save figure
            #print('i = ', ctr)
            fig.savefig(self.dirCurr + '/' + dirName + '/perturbation_' + str(ctr) + '.png')
            fig.clf()

            


    # EVALUATION
    def explain_instance_evaluation(self, expID, ID):
        print('explain_instance_evaluation function starting\n')

        self.expID = expID
        self.index = self.expID

        self.manual_instance_loading = False

        if self.explanation_algorithm == 'LIME':        
            # if explanation_mode is 'image'
            if self.explanation_mode == 'image':
                import time
                before_explain_instance_start = time.time()

                # Get local costmap
                # Original costmap will be saved to self.local_costmap_original
                self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

                # Make image a np.array deepcopy of local_costmap_original
                self.image = np.array(copy.deepcopy(self.local_costmap_original))

                # Turn inflated area to free space and 100s to 99s
                self.inflatedToStatic()

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
                self.image = self.image * 1.0
                
                # Saving data to .csv files for C++ node - local navigation planner
                self.SaveImageDataForLocalPlanner()

                # Saving important data to class variables
                self.saveImportantData2ClassVars()

                segm_fn = 'custom_segmentation'

                devDistance_x = 0
                sum_x = 0 
                devDistance_y = 0 
                sum_y = 0 
                devDistance = 0

                before_explain_instance_end = time.time()
                before_explain_instance_time = before_explain_instance_end - before_explain_instance_start

                segments_num = 8
    
                import time

                nums_of_samples = [2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256]

                dirName = 'evaluation_results'
                try:
                    os.mkdir(dirName)
                except FileExistsError:
                    pass

                with open(self.dirCurr + '/' + dirName + '/explanations' + str(ID) + '.csv', "a") as myfile:
                    myfile.write('num_samples,before_explain_instance_time,segmentation_time,classifier_fn_time,planner_time,target_calculation_time,costmap_save_time,distances_time,regressor_time,explain_instance_time,explanation_pics_time,plotting_time,weight_0,weight_1,weight_2,weight_3,weight_4,weight_5,weight_6,weight_7,weight_8\n')                    
                    
                    for i in nums_of_samples:
                        #print('\i = ', i)
                        num_of_iterations_for_one_num_of_segments = 10 #30 #50

                        if i == 256:
                            num_of_iterations_for_one_num_of_segments = 1
                        
                        for j in range(0, num_of_iterations_for_one_num_of_segments): 
                            # measure explain_instance time
                            start = time.time()
                            self.explanation, self.segments, segmentation_time, classifier_fn_time, planner_time, target_calculation_time,costmap_save_time, distances_time, regressor_time = self.explainer.explain_instance_evaluation(
                                self.image, self.classifier_fn_image_lime_evaluation, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                                self.plan_x_list, self.plan_y_list,
                                hide_color=perturb_hide_color_value,
                                num_segments=segments_num,
                                num_samples_current=i,
                                batch_size=2048, segmentation_fn=segm_fn,
                                top_labels=10)
                            end = time.time()
                            explain_instance_time = round(end - start, 3)

                            # measure get explanation_picture time
                            start = time.time()
                            self.temp_img, self.exp = self.explanation.get_image_and_mask_evaluation(label=0)  # min_weight=0.1 - default
                            end = time.time()
                            explanation_pics_time = round(end - start, 3)

                            # measure plotting time
                            self.segments += 1
                            start = time.time()
                            self.plotMinimalEvaluation(i)
                            end = time.time()
                            plotting_time = round(end - start, 3)

                            # round measured times to 3 decimal places
                            before_explain_instance_time = round(before_explain_instance_time, 3)
                            segmentation_time = round(segmentation_time, 3)
                            classifier_fn_time = round(classifier_fn_time, 3)
                            planner_time = round(planner_time, 3)
                            target_calculation_time = round(target_calculation_time,3)
                            costmap_save_time = round(costmap_save_time,3)
                            distances_time = round(distances_time,3)
                            regressor_time = round(regressor_time,3)

                            # write measured times to .csv file
                            myfile.write(str(i) + ',' + str(before_explain_instance_time) + ',' + str(segmentation_time) + ',' + str(classifier_fn_time) + ',' + str(planner_time) + ',' + str(target_calculation_time) + ',' + str(costmap_save_time) + ',' + str(distances_time) + ',' + str(regressor_time) + ',' + str(explain_instance_time) + ',' + str(explanation_pics_time) + ',' + str(plotting_time) + ',')
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
            
        elif self.explanation_algorithm == 'Anchors':
            pass

        elif self.explanation_algorithm == 'SHAP':
            pass

    def plotMinimalEvaluation(self, num_samples):
        dirName = 'evaluation_results'
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass

        big_plot_size = True
        w = h = 1.6
        if big_plot_size == True:
            w *= 3
            h *= 3

        '''
        # plot segments
        fig = plt.figure(frameon=True)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.segments.astype(np.uint8), aspect='auto')  # , aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/segments_' + str(num_samples) + '.png', transparent=False)
        fig.clf()
        '''

        # plot explanation
        fig = plt.figure(frameon=True)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)        
        ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        ax.imshow(self.temp_img.astype(np.uint8), aspect='auto')
        fig.savefig(self.dirCurr + '/' + dirName + '/explanation_' + str(num_samples) + '.png', transparent=False)
        fig.clf()

    def classifier_fn_image_lime_evaluation(self, sampled_instance):

        # Save perturbed costmap_data to file for C++ node
        costmap_start = time.time()

        self.sampled_instance_shape_len = len(sampled_instance.shape)
        self.sample_size = 1 if self.sampled_instance_shape_len == 2 else sampled_instance.shape[0]

        if self.sampled_instance_shape_len > 3:
            temp = np.delete(sampled_instance,2,3)
            #print(temp.shape)
            temp = np.delete(temp,1,3)
            #print(temp.shape)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 2:
            np.savetxt(self.dirCurr + '/src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")
        costmap_end = time.time()
        costmap_save_time = costmap_end - costmap_start
        print('Save perturbed costmap_data runtime: ', costmap_save_time)

        #print('starting C++ node')

        planner_start = time.time()

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        planner_end = time.time()
        planner_time = planner_end - planner_start

        #rospy.sleep(1)

        #print('C++ node ended')

        # load command velocities
        self.cmd_vel_perturb = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('self.cmd_vel: ', self.cmd_vel_perturb)
        #print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)

        # load local plans
        self.local_plans = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/local_plans.csv')
        #print('self.local_plans: ', self.local_plans)
        #print('self.local_plans.shape: ', self.local_plans.shape)

        # load transformed plan
        self.transformed_plan = pd.read_csv(self.dirCurr + '/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('self.transformed_plan: ', self.transformed_plan)
        #print('self.transformed_plan.shape: ', self.transformed_plan.shape)

        # fill the list of transformed plan coordinates
        self.transformed_plan_xs = []
        self.transformed_plan_ys = []
        for i in range(0, self.transformed_plan.shape[0]):
            x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)

        # DETERMINE THE DEVIATION TYPE
        # calculate original deviation - sum of minimal point-to-point distances
        calculate_original_deviation = False
        if calculate_original_deviation == True:
            original_deviation = -1.0
            diff_x = 0
            diff_y = 0
            devs = []
            for j in range(0, len(self.local_plan_x_list)):
                local_diffs = []
                for k in range(0, len(self.transformed_plan_xs)):
                    diff_x = (self.local_plan_x_list[j] - self.transformed_plan_xs[k]) ** 2
                    diff_y = (self.local_plan_y_list[j] - self.transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    local_diffs.append(diff)                        
                devs.append(min(local_diffs))   
            original_deviation = sum(devs)
            # original_deviation for big_deviation = 745.5051688094327
            # original_deviation for big_deviation without wall = 336.53749938826286
            # original_deviation for no_deviation = 56.05455197218764
            # original_deviation for small_deviation = 69.0
            # original_deviation for rotate_in_place = 307.4962940090125
            print('\noriginal_deviation = ', original_deviation)

        plot_perturbations = False
        if plot_perturbations == True:
            # only needed for classifier_fn_image_plot() function
            self.sampled_instance = sampled_instance
            # plot perturbation of local costmap
            self.classifier_fn_image_lime_plot()


        # DETERMINE THE DEVIATION TYPE
        determine_dev_type = False
        if determine_dev_type == True:
            #start_determine_dev = time.time()
            
            # thresholds
            local_plan_gap_threshold = 48.0
            big_deviation_threshold = 85.0
            small_deviation_threshold = 32.0 #30
            no_deviation_threshold = 0.0

            # test for the original local plan gap
            local_plan_original_gap = False
            local_plan_gaps = []
            diff = 0
            for j in range(0, len(self.local_plan_x_list) - 1):
                diff = math.sqrt( (self.local_plan_x_list[j]-self.local_plan_x_list[j+1])**2 + (self.local_plan_y_list[j]-self.local_plan_y_list[j+1])**2 )
                local_plan_gaps.append(diff)
            if max(local_plan_gaps) > local_plan_gap_threshold:
                local_plan_original_gap = True

            # local gap too big - stop (rotate_in_place)
            if local_plan_original_gap == True:
                deviation_type = 'stop'
                local_plan_gap_threshold = 55.0
            
            # no local gap - test further    
            elif original_deviation >= big_deviation_threshold:
                deviation_type = 'big_deviation'
            elif original_deviation >= small_deviation_threshold:
                deviation_type = 'small_deviation'
            else:
                deviation_type = 'no_deviation'    

            #end_determine_dev = time.time()
            #determine_dev_time = end_determine_dev - start_determine_dev
            #print('\ndetermine deviation type runtime = ', determine_dev_time)

            # PRINTING RESULTS                                       
            print('\ndeviation_type: ', deviation_type)
  
        mode = self.underlying_model_mode # 'regression' or 'classification' or 'regression_normalized_around_deviation' or 'regression_normalized'
        print('\nmode = ', mode)

        
        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        # TARGET CALCULATION PART
        target_start = time.time()

        # TARGET CALCULATION
        my_distance_fun = True
        if my_distance_fun == True:
            # deviation of local plan from global plan dataframe
            self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])
            #print('self.local_plan_deviation: ', self.local_plan_deviation)

            #### MAIN TARGET CALCULATION PART ####

            print_iterations = False
            
            #start_main = time.time()

            if mode == 'regression':
                # fill in deviation dataframe
                dev_original = 0
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                    
                    # find deviation as a sum of minimal point-to-point differences
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    self.local_plan_deviation.iloc[i, 0] = sum(devs)

            elif mode == 'classification':
                if deviation_type == 'stop':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('\ni = ', i)
                        # test if there is local plan
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(0, len(local_plan_xs)):
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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

                                    for j in range(0, len(x_indices)):
                                        for m in range(x_indices[j] - 0, x_indices[j] + 1):
                                            for q in range(y_indices[j] - 0, y_indices[j] + 1):
                                                if self.sampled_instance_shape_len == 3:
                                                    if sampled_instance[i, q, m] == 99:
                                                        local_plan_gap = True
                                                        break
                                                else:
                                                    if sampled_instance[q, m] == 99:
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
                    for i in range(0, self.sample_size): 
                        #print('i = ', i)
                        # test if there is local plan
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False    
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(len(local_plan_xs) - 1, len(local_plan_xs)):
                                if self.sample_size > 1:
                                    if sampled_instance[i][local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if self.sampled_instance_shape_len == 3:
                                        if sampled_instance[0, local_plan_ys[j], local_plan_xs[j]] == 99:
                                            local_plan_point_in_obstacle = True
                                            break
                                    else:
                                        if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)    

                                    if deviation >= big_deviation_threshold:
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
                                    #print('deviation: ', real_deviation)
                                    print('max dev: ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                    
                elif deviation_type == 'small_deviation':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('i = ', i)
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
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)
                                    
                                    if small_deviation_threshold <= deviation < big_deviation_threshold:
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
                                    #print('deviation: ', small_deviation)
                                    #print('minimal diff: ', min(diffs))
                                    print('max(devs): ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                    
                elif deviation_type == 'no_deviation':
                    # fill in deviation dataframe
                    for i in range(0, self.sample_size):
                        #print('i = ', i)
                        local_plan_xs = []
                        local_plan_ys = []
                        local_plan_found = False
                        
                        # find if there is local plan
                        self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                        for j in range(0, self.local_plans_local.shape[0]):
                                x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                                y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                                if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                    local_plan_xs.append(x_temp)
                                    local_plan_ys.append(y_temp)
                                    local_plan_found = True
                            
                        local_plan_point_in_obstacle = False
                        # if there is local plan test further    
                        if local_plan_found == True:
                            # test if any part of the local plan is in the obstacle region
                            for j in range(0, len(local_plan_xs)):
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                    # find deviation as a sum of minimal point-to-point differences
                                    diff_x = 0
                                    diff_y = 0
                                    devs = []
                                    for j in range(0, len(local_plan_xs)):
                                        local_diffs = []
                                        for k in range(0, len(self.transformed_plan_xs)):
                                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                            diff = math.sqrt(diff_x + diff_y)
                                            local_diffs.append(diff)                        
                                        devs.append(min(local_diffs))   

                                    deviation = sum(devs)
                                    
                                    if no_deviation_threshold <= deviation < small_deviation_threshold:
                                        self.local_plan_deviation.iloc[i, 0] = 1.0
                                    else:    
                                        self.local_plan_deviation.iloc[i, 0] = 0.0                
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
                                    #print('deviation: ', real_deviation)
                                    #print('minimal diff: ', min(diffs))
                                    print('max(devs): ', max(devs))
                            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                            print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])

            elif mode == 'regression_normalized_around_deviation':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)    

                    deviation = sum(devs)
                    deviation = deviation / original_deviation if deviation <= original_deviation else (2*original_deviation - deviation) / original_deviation
                    if deviation < 0:
                        deviation = 0.0
                    self.local_plan_deviation.iloc[i, 0] = deviation

            elif mode == 'regression_normalized':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)    

                    self.local_plan_deviation.iloc[i, 0] = sum(devs) / original_deviation

        elif my_distance_fun == False:
            self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])

            if mode == 'regression':
                # fill in deviation dataframe
                dev_original = 0
                #for i in range(0, sampled_instance.shape[0]):
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                    
                    # this happens almost never when only obstacles are segments, but let it stay for now
                    if local_plan_found == False:
                        if deviation_type == 'stop':
                            self.local_plan_deviation.iloc[i, 0] = dev_original
                        elif deviation_type == 'no_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                        elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        continue             

                    # find deviation as a sum of minimal point-to-point differences
                    '''
                    diff_x = 0
                    diff_y = 0
                    devs = []
                    for j in range(0, len(local_plan_xs)):
                        local_diffs = []
                        for k in range(0, len(self.transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            local_diffs.append(diff)                        
                        devs.append(min(local_diffs))   

                    if i == 0:
                        dev_original = sum(devs)
                    '''      

                    #t1 = np.column_stack((local_plan_xs, local_plan_ys))
                    #t1 = np.vstack((local_plan_xs, local_plan_ys))
                    #print('\nt1 = ', t1)
                    #print('\nt1.shape = ', t1.shape)
                    #t1 = np.fliplr(t1)
                    #print('\nt1.shape = ', t1.shape)
                    #t2 = np.column_stack((self.transformed_plan_xs, self.transformed_plan_ys))
                    #t2 = np.vstack((self.transformed_plan_xs, self.transformed_plan_ys))
                    #print('\nt2 = ', t2)
                    #print('\nt2.shape = ', t2.shape)
                    #min_len = min(t1.shape[1], t2.shape[1])
                    #dist = traj_dist.eucl_dist_traj(np.array(local_plan_xs), np.array(self.transformed_plan_xs)) + traj_dist.eucl_dist_traj(np.array(local_plan_ys), np.array(self.transformed_plan_ys))
                    
                    #dist = traj_dist.eucl_dist_traj(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.eucl_dist_traj(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D
                    #dist = traj_dist.e_hausdorff(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_hausdorff(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D
                    #dist = traj_dist.e_sspd(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_sspd(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2 #2D

                    #dist = traj_dist.discret_frechet(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.discret_frechet(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    #dist = traj_dist.e_dtw(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.e_dtw(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    #dist = traj_dist.e_edr(np.array(local_plan_xs), np.array(self.transformed_plan_xs), 0.01)**2 + traj_dist.e_edr(np.array(local_plan_ys), np.array(self.transformed_plan_ys), 0.01)**2
                    #dist = traj_dist.e_lcss(np.array(local_plan_xs), np.array(self.transformed_plan_xs), 0.01)**2 + traj_dist.e_lcss(np.array(local_plan_ys), np.array(self.transformed_plan_ys), 0.01)**2
                    dist = traj_dist.owd_grid_brut(np.array(local_plan_xs), np.array(self.transformed_plan_xs))**2 + traj_dist.owd_grid_brut(np.array(local_plan_ys), np.array(self.transformed_plan_ys))**2
                    
                    dist = math.sqrt(dist)
                    self.local_plan_deviation.iloc[i, 0] = dist
                    
                    #dist = traj_dist.discret_frechet(t1[:,0:min_len], t2[:,0:min_len])
                    #print('\ndist_raw = ', self.local_plan_deviation.iloc[i, 0])
                    #self.local_plan_deviation.iloc[i, 0] = math.sqrt(dist[0,0]**2+dist[1,1]**2)
                    #print('\ndist = ', self.local_plan_deviation.iloc[i, 0])

        #print(self.local_plan_deviation)

        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        #self.cmd_vel_perturb['deviate'].to_csv('deviations.csv')

        #end_main = time.time()
        #main_time = end_main - start_main
        #print('\ntarget calculation runtime = ', main_time)

        # if more outputs wanted
        more_outputs = False
        if more_outputs == True:
            # classification
            print('\n(lin_x, ang_z) = ', (self.cmd_vel_perturb.iloc[0, 0], self.cmd_vel_perturb.iloc[0, 2]))
            
            # stop
            if abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                stop_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        stop_list.append(1.0)
                    else:
                        stop_list.append(0.0)
                self.cmd_vel_perturb['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['stop'])
            
            # ahead_straight
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                ahead_straight_list = [] = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        ahead_straight_list = [].append(1.0)
                    else:
                        ahead_straight_list = [].append(0.0)
                self.cmd_vel_perturb['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_straight'])

            # back_straight
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                back_straight_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        back_straight_list = [].append(1.0)
                    else:
                        back_straight_list = [].append(0.0)
                self.cmd_vel_perturb['back_straight'] = pd.DataFrame(np.array(back_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['back_straight'])
            
            # rotate_right_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_in_place_list.append(1.0)
                    else:
                        rotate_right_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_in_place'] = pd.DataFrame(np.array(rotate_right_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_in_place'])

            # rotate_left_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_in_place_list.append(1.0)
                    else:
                        rotate_left_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_in_place'] = pd.DataFrame(np.array(rotate_left_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_in_place'])

            # rotate_right_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_ahead_list.append(1.0)
                    else:
                        rotate_right_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_ahead'] = pd.DataFrame(np.array(rotate_right_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_ahead'])

            # rotate_left_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_ahead_list.append(1.0)
                    else:
                        rotate_left_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_ahead'] = pd.DataFrame(np.array(rotate_left_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_ahead'])

            # rotate_right_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_back_list.append(1.0)
                    else:
                        rotate_right_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_back'] = pd.DataFrame(np.array(rotate_right_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_back'])

            # rotate_left_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_back_list.append(1.0)
                    else:
                        rotate_left_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_back'] = pd.DataFrame(np.array(rotate_left_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_back'])

            print('\nrobot_decision = ', self.cmd_vel_perturb.columns.values[-1])
        
        target_end = time.time()
        target_calculation_time = target_end - target_start            
           
        #self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        
        print('classifier_fn_image ended')

        return np.array(self.local_plan_deviation), planner_time, target_calculation_time, costmap_save_time


    # TABULAR
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


    # ANCHORS
    # classifier function for anchors image
    def classifier_fn_image_anchors(self, sampled_instance):

        print('\nclassifier_fn_image_anchors started')

        print('\nsampled_instance.shape = ', sampled_instance.shape)

        self.sampled_instance_shape_len = len(sampled_instance.shape)
        self.sample_size = 1 if self.sampled_instance_shape_len == 2 else sampled_instance.shape[0]

        # Save perturbed costmap_data to file for C++ node
        #costmap_save_start = time.time()

        if self.sampled_instance_shape_len > 3:
            temp = np.delete(sampled_instance,2,3)
            #print(temp.shape)
            temp = np.delete(temp,1,3)
            #print(temp.shape)
            temp = temp.reshape(temp.shape[0]*160,160)
            np.savetxt('./src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 3:
            temp = sampled_instance.reshape(sampled_instance.shape[0]*160,160)
            np.savetxt('./src/teb_local_planner/src/Data/costmap_data.csv', temp, delimiter=",")
        elif self.sampled_instance_shape_len == 2:
            np.savetxt('./src/teb_local_planner/src/Data/costmap_data.csv', sampled_instance, delimiter=",")

        #costmap_save_end = time.time()
        #costmap_save_time = costmap_save_end - costmap_save_start
        #print('\nsave perturbed costmap_data runtime: ', costmap_save_time)

        # calling ROS C++ node
        #print('\nstarting C++ node')

        #planner_calculation_start = time.time()

        # start perturbed_node_image ROS C++ node
        Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/perturb_node_image/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /perturb_node_image'))

        #planner_calculation_end = time.time()
        #planner_calculation_time = planner_calculation_end - planner_calculation_start
        #print('\nplanner calculation runtime = ', planner_calculation_time)

        rospy.sleep(1)

        #print('\nC++ node ended')

        #output_start = time.time()
        # load command velocities - output from local planner
        self.cmd_vel_perturb = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/cmd_vel.csv')
        #print('self.cmd_vel: ', self.cmd_vel_perturb)
        #print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)
        #self.cmd_vel_perturb.to_csv('cmd_vel.csv')

        # load local plans - output from local planner
        self.local_plans = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/local_plans.csv')
        #print('self.local_plans: ', self.local_plans)
        #print('self.local_plans.shape: ', self.local_plans.shape)
        #self.local_plans.to_csv('local_plans.csv')

        # load transformed global plan to /odom frame
        self.transformed_plan = pd.read_csv('~/amar_ws/src/teb_local_planner/src/Data/transformed_plan.csv')
        #print('self.transformed_plan: ', self.transformed_plan)
        #print('self.transformed_plan.shape: ', self.transformed_plan.shape)
        #self.transformed_plan.to_csv('transformed_plan.csv')
        #output_end = time.time()
        #output_time = output_end - output_start
        #print('\noutput time: ', output_time)

        # fill the list of transformed plan coordinates
        #start_transformed = time.time()
        self.transformed_plan_xs = []
        self.transformed_plan_ys = []
        for i in range(0, self.transformed_plan.shape[0]):
            x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                self.transformed_plan_xs.append(x_temp)
                self.transformed_plan_ys.append(y_temp)
        #end_transformed = time.time()
        #transformed_time = end_transformed - start_transformed
        #print('\nfill the list of transformed plan coordinates runtime = ', transformed_time)

        # calculate original deviation - sum of minimal point-to-point distances
        original_deviation = -1.0
        diff_x = 0
        diff_y = 0
        devs = []
        for j in range(0, len(self.local_plan_x_list)):
            local_diffs = []
            deviation_local = True  
            for k in range(0, len(self.transformed_plan_xs)):
                diff_x = (self.local_plan_x_list[j] - self.transformed_plan_xs[k]) ** 2
                diff_y = (self.local_plan_y_list[j] - self.transformed_plan_ys[k]) ** 2
                diff = math.sqrt(diff_x + diff_y)
                local_diffs.append(diff)                        
            devs.append(min(local_diffs))   
        original_deviation = sum(devs)
        print('\noriginal_deviation = ', original_deviation)
        # original_deviation for big_deviation = 745.5051688094327
        # original_deviation for big_deviation without wall = 336.53749938826286
        # original_deviation for no_deviation = 56.05455197218764
        # original_deviation for small_deviation = 69.0
        # original_deviation for rotate_in_place = 307.4962940090125

        plot_perturbations = False
        if plot_perturbations == True:
            # only needed for classifier_fn_image_plot() function
            self.sampled_instance = sampled_instance
            # plot perturbation of local costmap
            self.classifier_fn_image_anchors_plot()

        # DETERMINE THE DEVIATION TYPE
        determine_dev_type = True
        if determine_dev_type == True:
            #start_determine_dev = time.time()
            
            # thresholds
            local_plan_gap_threshold = 48.0
            big_deviation_threshold = 85.0
            small_deviation_threshold = 32.0 #30
            no_deviation_threshold = 0.0

            # test for the original local plan gap
            local_plan_original_gap = False
            local_plan_gaps = []
            diff = 0
            for j in range(0, len(self.local_plan_x_list) - 1):
                diff = math.sqrt( (self.local_plan_x_list[j]-self.local_plan_x_list[j+1])**2 + (self.local_plan_y_list[j]-self.local_plan_y_list[j+1])**2 )
                local_plan_gaps.append(diff)
            if max(local_plan_gaps) > local_plan_gap_threshold:
                local_plan_original_gap = True

            # local gap too big - stop (rotate_in_place)
            if local_plan_original_gap == True:
                deviation_type = 'stop'
                local_plan_gap_threshold = 55.0
            
            # no local gap - test further    
            elif original_deviation >= big_deviation_threshold:
                deviation_type = 'big_deviation'
            elif original_deviation >= small_deviation_threshold:
                deviation_type = 'small_deviation'
            else:
                deviation_type = 'no_deviation'    

            #end_determine_dev = time.time()
            #determine_dev_time = end_determine_dev - start_determine_dev
            #print('\ndetermine deviation type runtime = ', determine_dev_time)

            # PRINTING RESULTS                                       
            print('\ndeviation_type: ', deviation_type)
       
        # deviation of local plan from global plan dataframe
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(self.sample_size), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)

        #### MAIN TARGET CALCULATION PART ####

        print_iterations = False
        
        mode = self.underlying_model_mode
        #mode = 'regression' # 'regression' or 'classification' or 'regression_normalized_around_deviation' or 'regression_normalized'
        #print('\nmode = ', mode)

        #start_main = time.time()

        if mode == 'regression':
            # fill in deviation dataframe
            dev_original = 0
            #for i in range(0, sampled_instance.shape[0]):
            for i in range(0, self.sample_size):
                #print('\ni = ', i)
                local_plan_xs = []
                local_plan_ys = []
                local_plan_found = False
                
                # find if there is local plan
                self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                for j in range(0, self.local_plans_local.shape[0]):
                        x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                            local_plan_xs.append(x_temp)
                            local_plan_ys.append(y_temp)
                            local_plan_found = True
                
                # this happens almost never when only obstacles are segments, but let it stay for now
                if local_plan_found == False:
                    if deviation_type == 'stop':
                        self.local_plan_deviation.iloc[i, 0] = dev_original
                    elif deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                    elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    continue             

                # find deviation as a sum of minimal point-to-point differences
                diff_x = 0
                diff_y = 0
                devs = []
                for j in range(0, len(local_plan_xs)):
                    local_diffs = []
                    for k in range(0, len(self.transformed_plan_xs)):
                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        local_diffs.append(diff)                        
                    devs.append(min(local_diffs))   

                if i == 0:
                    dev_original = sum(devs)    

                self.local_plan_deviation.iloc[i, 0] = sum(devs) / original_deviation

        elif mode == 'classification':
            if deviation_type == 'stop':
                # fill in deviation dataframe
                for i in range(0, self.sample_size):
                    #print('\ni = ', i)
                    # test if there is local plan
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                        
                    local_plan_point_in_obstacle = False
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(0, len(local_plan_xs)):
                            if self.sampled_instance_shape_len == 3:
                                if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                    local_plan_point_in_obstacle = True
                                    break
                            else:
                                if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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

                                for j in range(0, len(x_indices)):
                                    for m in range(x_indices[j] - 0, x_indices[j] + 1):
                                        for q in range(y_indices[j] - 0, y_indices[j] + 1):
                                            if self.sampled_instance_shape_len == 3:
                                                if sampled_instance[i, q, m] == 99:
                                                    local_plan_gap = True
                                                    break
                                            else:
                                                if sampled_instance[q, m] == 99:
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
                for i in range(0, self.sample_size): 
                    #print('i = ', i)
                    # test if there is local plan
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                        
                    local_plan_point_in_obstacle = False    
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(len(local_plan_xs) - 1, len(local_plan_xs)):
                            if self.sample_size > 1:
                                if sampled_instance[i][local_plan_ys[j], local_plan_xs[j]] == 99:
                                    local_plan_point_in_obstacle = True
                                    break
                            else:
                                if self.sampled_instance_shape_len == 3:
                                    if sampled_instance[0, local_plan_ys[j], local_plan_xs[j]] == 99:
                                        local_plan_point_in_obstacle = True
                                        break
                                else:
                                    if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                # find deviation as a sum of minimal point-to-point differences
                                diff_x = 0
                                diff_y = 0
                                devs = []
                                for j in range(0, len(local_plan_xs)):
                                    local_diffs = []
                                    for k in range(0, len(self.transformed_plan_xs)):
                                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                        diff = math.sqrt(diff_x + diff_y)
                                        local_diffs.append(diff)                        
                                    devs.append(min(local_diffs))   

                                deviation = sum(devs)    

                                if deviation >= big_deviation_threshold:
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
                for i in range(0, self.sample_size):
                    #print('i = ', i)
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
                            if self.sampled_instance_shape_len == 3:
                                if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                    local_plan_point_in_obstacle = True
                                    break
                            else:
                                if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                # find deviation as a sum of minimal point-to-point differences
                                diff_x = 0
                                diff_y = 0
                                devs = []
                                for j in range(0, len(local_plan_xs)):
                                    local_diffs = []
                                    for k in range(0, len(self.transformed_plan_xs)):
                                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                        diff = math.sqrt(diff_x + diff_y)
                                        local_diffs.append(diff)                        
                                    devs.append(min(local_diffs))   

                                deviation = sum(devs)
                                
                                if small_deviation_threshold <= deviation < big_deviation_threshold:
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
                                #print('deviation: ', small_deviation)
                                #print('minimal diff: ', min(diffs))
                                print('max(devs): ', max(devs))
                        print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                        print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])
                
            elif deviation_type == 'no_deviation':
                # fill in deviation dataframe
                for i in range(0, self.sample_size):
                    #print('i = ', i)
                    local_plan_xs = []
                    local_plan_ys = []
                    local_plan_found = False
                    
                    # find if there is local plan
                    self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                    for j in range(0, self.local_plans_local.shape[0]):
                            x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                            y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                                local_plan_xs.append(x_temp)
                                local_plan_ys.append(y_temp)
                                local_plan_found = True
                        
                    local_plan_point_in_obstacle = False
                    # if there is local plan test further    
                    if local_plan_found == True:
                        # test if any part of the local plan is in the obstacle region
                        for j in range(0, len(local_plan_xs)):
                            if self.sampled_instance_shape_len == 3:
                                if sampled_instance[i, local_plan_ys[j], local_plan_xs[j]] == 99:
                                    local_plan_point_in_obstacle = True
                                    break
                            else:
                                if sampled_instance[local_plan_ys[j], local_plan_xs[j]] == 99:
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
                                # find deviation as a sum of minimal point-to-point differences
                                diff_x = 0
                                diff_y = 0
                                devs = []
                                for j in range(0, len(local_plan_xs)):
                                    local_diffs = []
                                    for k in range(0, len(self.transformed_plan_xs)):
                                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                                        diff = math.sqrt(diff_x + diff_y)
                                        local_diffs.append(diff)                        
                                    devs.append(min(local_diffs))   

                                deviation = sum(devs)
                                
                                if no_deviation_threshold <= deviation < small_deviation_threshold:
                                    self.local_plan_deviation.iloc[i, 0] = 1.0
                                else:    
                                    self.local_plan_deviation.iloc[i, 0] = 0.0                
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
                                #print('deviation: ', real_deviation)
                                #print('minimal diff: ', min(diffs))
                                print('max(devs): ', max(devs))
                        print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
                        print('self.local_plan_deviation.iloc[i, 0]: ', self.local_plan_deviation.iloc[i, 0])

        elif mode == 'regression_normalized_around_deviation':
            # fill in deviation dataframe
            dev_original = 0
            #for i in range(0, sampled_instance.shape[0]):
            for i in range(0, self.sample_size):
                #print('\ni = ', i)
                local_plan_xs = []
                local_plan_ys = []
                local_plan_found = False
                
                # find if there is local plan
                self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                for j in range(0, self.local_plans_local.shape[0]):
                        x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                            local_plan_xs.append(x_temp)
                            local_plan_ys.append(y_temp)
                            local_plan_found = True
                
                # this happens almost never when only obstacles are segments, but let it stay for now
                if local_plan_found == False:
                    if deviation_type == 'stop':
                        self.local_plan_deviation.iloc[i, 0] = dev_original
                    elif deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                    elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    continue             

                # find deviation as a sum of minimal point-to-point differences
                diff_x = 0
                diff_y = 0
                devs = []
                for j in range(0, len(local_plan_xs)):
                    local_diffs = []
                    for k in range(0, len(self.transformed_plan_xs)):
                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        local_diffs.append(diff)                        
                    devs.append(min(local_diffs))   

                if i == 0:
                    dev_original = sum(devs)    

                deviation = sum(devs)
                deviation = deviation / original_deviation if deviation <= original_deviation else (2*original_deviation - deviation) / original_deviation
                if deviation < 0:
                    deviation = 0.0
                self.local_plan_deviation.iloc[i, 0] = deviation

        elif mode == 'regression_normalized':
            # fill in deviation dataframe
            dev_original = 0
            #for i in range(0, sampled_instance.shape[0]):
            for i in range(0, self.sample_size):
                #print('\ni = ', i)
                local_plan_xs = []
                local_plan_ys = []
                local_plan_found = False
                
                # find if there is local plan
                self.local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
                for j in range(0, self.local_plans_local.shape[0]):
                        x_temp = int((self.local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                        y_temp = int((self.local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                        if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                            local_plan_xs.append(x_temp)
                            local_plan_ys.append(y_temp)
                            local_plan_found = True
                
                # this happens almost never when only obstacles are segments, but let it stay for now
                if local_plan_found == False:
                    if deviation_type == 'stop':
                        self.local_plan_deviation.iloc[i, 0] = dev_original
                    elif deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 745.5051688094327 #1000
                    elif deviation_type == 'big_deviation' or deviation_type == 'small_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    continue             

                # find deviation as a sum of minimal point-to-point differences
                diff_x = 0
                diff_y = 0
                devs = []
                for j in range(0, len(local_plan_xs)):
                    local_diffs = []
                    for k in range(0, len(self.transformed_plan_xs)):
                        diff_x = (local_plan_xs[j] - self.transformed_plan_xs[k]) ** 2
                        diff_y = (local_plan_ys[j] - self.transformed_plan_ys[k]) ** 2
                        diff = math.sqrt(diff_x + diff_y)
                        local_diffs.append(diff)                        
                    devs.append(min(local_diffs))   

                if i == 0:
                    dev_original = sum(devs)    

                self.local_plan_deviation.iloc[i, 0] = sum(devs) / original_deviation

        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        #self.cmd_vel_perturb['deviate'].to_csv('deviations.csv')

        #end_main = time.time()
        #main_time = end_main - start_main
        #print('\ntarget calculation runtime = ', main_time)

        # if more outputs wanted
        more_outputs = True
        if more_outputs == True:
            # classification
            print('\n(lin_x, ang_z) = ', (self.cmd_vel_perturb.iloc[0, 0], self.cmd_vel_perturb.iloc[0, 2]))
            
            # stop
            if abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                stop_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        stop_list.append(1.0)
                    else:
                        stop_list.append(0.0)
                self.cmd_vel_perturb['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['stop'])
            
            # ahead_straight
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                ahead_straight_list = [] = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        ahead_straight_list = [].append(1.0)
                    else:
                        ahead_straight_list = [].append(0.0)
                self.cmd_vel_perturb['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_straight'])

            # back_straight
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[0, 2]) <= 0.01:
                back_straight_list = [] = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) <= 0.01:
                        back_straight_list = [].append(1.0)
                    else:
                        back_straight_list = [].append(0.0)
                self.cmd_vel_perturb['back_straight'] = pd.DataFrame(np.array(back_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['back_straight'])
            
            # rotate_right_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_in_place_list.append(1.0)
                    else:
                        rotate_right_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_in_place'] = pd.DataFrame(np.array(rotate_right_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_in_place'])

            # rotate_left_in_place
            elif abs(self.cmd_vel_perturb.iloc[0, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_in_place_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if abs(self.cmd_vel_perturb.iloc[i, 0]) <= 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_in_place_list.append(1.0)
                    else:
                        rotate_left_in_place_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_in_place'] = pd.DataFrame(np.array(rotate_left_in_place_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_in_place'])

            # rotate_right_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_ahead_list.append(1.0)
                    else:
                        rotate_right_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_ahead'] = pd.DataFrame(np.array(rotate_right_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_ahead'])

            # rotate_left_ahead
            elif self.cmd_vel_perturb.iloc[0, 0] > 0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_ahead_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_ahead_list.append(1.0)
                    else:
                        rotate_left_ahead_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_ahead'] = pd.DataFrame(np.array(rotate_left_ahead_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_ahead'])

            # rotate_right_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] > 0.01:
                rotate_right_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.01:
                        rotate_right_back_list.append(1.0)
                    else:
                        rotate_right_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_right_back'] = pd.DataFrame(np.array(rotate_right_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right_back'])

            # rotate_left_back
            elif self.cmd_vel_perturb.iloc[0, 0] < -0.01 and self.cmd_vel_perturb.iloc[0, 2] < -0.01:
                rotate_left_back_list = []
                for i in range(0, self.cmd_vel_perturb.shape[0]):
                    if self.cmd_vel_perturb.iloc[i, 0] < -0.01 and self.cmd_vel_perturb.iloc[i, 2] < -0.01:
                        rotate_left_back_list.append(1.0)
                    else:
                        rotate_left_back_list.append(0.0)
                self.cmd_vel_perturb['rotate_left_back'] = pd.DataFrame(np.array(rotate_left_back_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left_back'])

            print('\nrobot_decision = ', self.cmd_vel_perturb.columns.values[-1])

        print('\nclassifier_fn_image_anchors ended\n')

        return np.array(self.cmd_vel_perturb.iloc[:, 4:])

    # function for plotting anchors image perturbations
    def classifier_fn_image_anchors_plot(self):

        if self.sampled_instance_shape_len == 2 or self.sample_size == 1:
            return 0

        # plot every perturbation
        for i in range(0, self.sample_size):

            # save current perturbation as .csv file
            #pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('perturbation_' + str(i) + '.csv', index=False, header=False)

            # plot perturbed local costmap
            plt.imshow(self.sampled_instance[i, :, :])

            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []

            # find if there is local plan
            local_plans_local = self.local_plans.loc[self.local_plans['ID'] == i]
            for j in range(0, local_plans_local.shape[0]):
                    x_temp = int((local_plans_local.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((local_plans_local.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)
                        
                        #'''
                        #[yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans_local.iloc[j, 2], self.local_plans_local.iloc[j, 3])
                        #yaw_x = math.cos(yaw)
                        #yaw_y = math.sin(yaw)
                        #plt.quiver(x_temp, y_temp, yaw_y, yaw_x, color='white')
                        #'''

            # plot transformed plan
            plt.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='x')

            '''
            # plot footprints for first five points of local plan
            # indices of local plan's poses in local costmap
            self.footprint_local_plan_x_list = []
            self.footprint_local_plan_y_list = []
            self.footprint_local_plan_x_list_angle = []
            self.footprint_local_plan_y_list_angle = []
            for j in range(0, self.local_plans_local.shape[0]):
                for k in range(6, 7):
                    [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans_local.iloc[j + k, 2], self.local_plans_local.iloc[j + k, 3])
                    sin_th = math.sin(yaw)
                    cos_th = math.cos(yaw)

                    for l in range(0, self.footprint_tmp.shape[0]):
                        x_new = self.footprint_tmp.iloc[l, 0] + (self.local_plans_local.iloc[j + k, 0] - self.odom_x)
                        y_new = self.footprint_tmp.iloc[l, 1] + (self.local_plans_local.iloc[j + k, 1] - self.odom_y)
                        self.footprint_local_plan_x_list.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                        self.footprint_local_plan_y_list.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))

                        x_new = self.local_plans_local.iloc[j + k, 0] + (self.footprint_tmp.iloc[l, 0] - self.odom_x) * sin_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * cos_th
                        y_new = self.local_plans_local.iloc[j + k, 1] - (self.footprint_tmp.iloc[l, 0] - self.odom_x) * cos_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * sin_th
                        self.footprint_local_plan_x_list_angle.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                        self.footprint_local_plan_y_list_angle.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))
                break
            # plot footprints
            plt.scatter(self.footprint_local_plan_x_list, self.footprint_local_plan_y_list, c='green', marker='x')
            plt.scatter(self.footprint_local_plan_x_list_angle, self.footprint_local_plan_y_list_angle, c='white', marker='x')
            '''

            # plot local plan
            plt.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='x')

            # plot robot's location and orientation
            plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            #plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')

            # plot command velocities as text
            plt.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_perturb.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_perturb.iloc[i, 2], 2)))

            # save figure
            plt.savefig('perturbation_' + str(i) + '.png')
            plt.clf()

    # plot explanation picture and segments for anchors
    def plotExplanationAnchors(self):
        path_core = os.getcwd()

        # import needed libraries
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # plot costmap
        fig = plt.figure(frameon=False)
        w = 1.6#*3
        h = 1.6#*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #print('self.image.shape: ', self.image.shape)
        gray_shade = 180
        white_shade = 0
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
        w = 1.6#*3
        h = 1.6#*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
        #ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_y, self.yaw_odom_x, color='white')
        #ax.imshow(self.image, aspect='auto')
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(path_core + '/input.png')
        if self.eps == True:
            fig.savefig(path_core + '/input.eps')
        fig.clf()


        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6#*3
        h = 1.6#*3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        #ax.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        #print('len(self.transformed_plan_x_list): ', len(self.transformed_plan_xs))
        ax.scatter(self.transformed_plan_xs, self.transformed_plan_ys, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='yellow', marker='o')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')

        exp_img = copy.deepcopy(image)
        include_features_from_exp = True
        if include_features_from_exp == True:
            for e in self.exp:
                exp_img[:,:,0][self.segments == e[0]+1] = 255
                exp_img[:,:,1][self.segments == e[0]+1] = 255
                exp_img[:,:,2][self.segments == e[0]+1] = 255
        else:
            print('\nlen(self.best_tuples) = ', len(self.best_tuples))
            for t in self.best_tuples[2:3]:
                for t_ in t:
                    exp_img[:,:,0][self.segments == t_+1] = 255
                    exp_img[:,:,1][self.segments == t_+1] = 255
                    exp_img[:,:,2][self.segments == t_+1] = 255        
        ax.imshow(exp_img.astype(np.uint8), aspect='auto') 
        fig.savefig(path_core + '/explanation.png', transparent=False)
        if self.eps == True:
            fig.savefig(path_core + '/explanation.eps', transparent=False)
        fig.clf()


    # DATASET
    def explain_instance_dataset(self, expID, iteration_ID):
        print('explain_instance_dataset function starting\n')

        self.expID = expID
        self.index = expID

        self.manual_instance_loading = False

        if self.explanation_algorithm == 'LIME': 

            # if explanation_mode is 'image'
            if self.explanation_mode == 'image':

                # Get local costmap
                # Original costmap will be saved to self.local_costmap_original
                self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

                # Make image a np.array deepcopy of local_costmap_original
                self.image = np.array(copy.deepcopy(self.local_costmap_original))

                # Turn inflated area to free space and 100s to 99s
                self.inflatedToStatic()

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
                self.image = self.image * 1.0
                
                # Saving data to .csv files for C++ node - local navigation planner
                self.SaveImageDataForLocalPlanner()

                # Saving important data to class variables
                self.saveImportantData2ClassVars()

                # Use new variable in the algorithm - possible time saving
                img = copy.deepcopy(self.image)

                segm_fn = 'custom_segmentation'
                #print('segm_fn = ', segm_fn)

                devDistance_x = 0
                sum_x = 0 
                devDistance_y = 0 
                sum_y = 0 
                devDistance = 0

                self.explanation, self.segments = self.explainer.explain_instance(img, self.classifier_fn_image_lime, self.costmap_info_tmp, self.map_info, self.tf_odom_map,
                                                                                    self.localCostmapIndex_x_odom, self.localCostmapIndex_y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance,
                                                                                    self.plan_x_list, self.plan_y_list,
                                                                                    hide_color=perturb_hide_color_value, batch_size=2048, segmentation_fn=segm_fn, top_labels=10)
                        
                self.temp_img, self.exp = self.explanation.get_image_and_mask(label=0)            
                
                self.plotMinimalDataset(iteration_ID)

        elif self.explanation_algorithm == 'Anchors':
            pass

    def plotMinimalDataset(self, iteration_ID):
        dirName = 'dataset_creation_results'
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass

        dir_small = dirName + '/small'
        try:
            os.mkdir(dir_small)
        except FileExistsError:
            pass
        
        dir_big = dirName + '/big'
        try:
            os.mkdir(dir_big)
        except FileExistsError:
            pass    

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
        white_shade = 0
        image = gray2rgb(self.image)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = gray_shade
                elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                    image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = white_shade    
        ax.imshow(image.astype(np.uint8), aspect='auto')
        fig.savefig(self.dirCurr + '/' + dir_small + '/' + str(iteration_ID) + '_costmap.png')
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
        fig.savefig(self.dirCurr + '/' + dir_big + '/' + str(iteration_ID) + '_costmap.png')
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
        fig.savefig(self.dirCurr + '/' + dir_big + '/' + str(iteration_ID) + '_input.png')
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
        fig.savefig(self.dirCurr + '/' + dir_small + '/' + str(iteration_ID) + '_input.png')
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
        fig.savefig(self.dirCurr + '/' + dir_small + '/' + str(iteration_ID) + '_input_segmented.png')
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
        fig.savefig(self.dirCurr + '/' + dir_big + '/' + str(iteration_ID) + '_input_segmented.png')
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
        fig.savefig(self.dirCurr + '/' + dir_small + '/' + str(iteration_ID) + '_output.png', transparent=False)
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
        fig.savefig(self.dirCurr + '/' + dir_big + '/' + str(iteration_ID) + '_output.png', transparent=False)
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
        fig.savefig(self.dirCurr + '/' + dir_small + '/' + str(iteration_ID) + '_weighted_segments.png')
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
        fig.savefig(self.dirCurr + '/' + dir_big + '/' + str(iteration_ID) + '_weighted_segments.png')
        fig.clf()
                        
        for i in range(1, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:        
                with open(self.dirCurr + '/' + dirName + '/local_plan_coordinates.csv', "a") as myfile:
                     myfile.write(str(iteration_ID) + ',' + str(self.local_plan_tmp.iloc[i, 0]) + ',' + str(self.local_plan_tmp.iloc[i, 1]) + '\n')


        for i in range(0, self.plan_tmp_tmp.shape[0], 3):
            x_temp = int((self.plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
            if 0 <= x_temp < self.costmap_size and 0 <= y_temp < self.costmap_size:
                with open(self.dirCurr + '/' + dirName + '/global_plan_coordinates.csv', "a") as myfile:
                    myfile.write(str(iteration_ID) + ',' + str(self.plan_tmp_tmp.iloc[i, 0]) + ',' + str(self.plan_tmp_tmp.iloc[i, 1]) + '\n')
       
        with open(self.dirCurr + '/' + dirName + '/costmap_data.csv', "a") as myfile:
                #myfile.write('picture_ID,width,heigth,origin_x,origin_y,resolution\n')
                myfile.write(str(iteration_ID) + ',' + str(self.localCostmapWidth) + ',' + str(self.localCostmapHeight) + ',' + str(self.localCostmapOriginX) + ',' + str(self.localCostmapOriginY) + ',' + str(self.localCostmapResolution) + '\n')

        with open(self.dirCurr + '/' + dirName + '/robot_coordinates.csv', "a") as myfile:
                #myfile.write('picture_ID,position_x,position_y\n')
                myfile.write(str(iteration_ID) + ',' + str(self.odom_x) + ',' + str(self.odom_y) + '\n')       
        

    # HELPER CALCULATION FUNCTIONS     
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


'''
# Other code for dealing with calling other ROS node different from this one
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
       
