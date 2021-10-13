#!/usr/bin/env python3

# Defining parameters - global variables

# test type: 'single', 'dataset_creation', 'evaluation', 'GAN', 'LIMEvsGAN'
test_type = 'single'

# possible explanation algorithms: 'lime', 'shap', 'anchors'
explanation_alg = 'lime'

# possible explanation modes: 'tabular', 'image', 'tabular_costmap', 'text'
explanation_mode = 'image'

# tabular explanation modes: 'regression', 'classification'
tabular_mode = 'regression'

# one hot encoding: 'True' or 'False' - for tabular classification
one_hot_encoding = True

# header of the output class/column
#output_class_name = 'beginning'

# set number of samples (does not define/affect the number of samples in LIME image)
num_samples = 256

# size of the one dimension of a local costmap
costmap_size = 160

import math

def quaternion_to_euler(x, y, z, w):
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

def preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints):
    offsets = []
    # Detect number of entries with 'None' frame based on local_costmap_info
    offsets.append(len(local_costmap_info[local_costmap_info['frame'] == 'None']))

    # detect offsets in plans
    offsets.append(int(plan.iloc[0, 5]))
    offsets.append(int(teb_global_plan.iloc[0, 5]))
    offsets.append(int(teb_local_plan.iloc[0, 5]))
    offsets.append(int(footprints.iloc[0, 4]))
    #print('offsets: ', offsets)
    
    num_of_first_rows_to_delete = max(offsets)
    '''
    print('num_of_first_rows_to_delete: ', num_of_first_rows_to_delete)
    print('\n')
    '''

    # Delete entries with 'None' frame from local_costmap_info
    local_costmap_info.drop(index=local_costmap_info.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('local_costmap_info after deleting entries with None frame or plans offset or footprint offset: ')
    print(local_costmap_info)
    print('\n')
    '''

    '''
    print('local_costmap_info.shape after deleting entries with None frame or plans offset or footprint offset: ', local_costmap_info.shape)
    print('\n')
    '''


    # Delete entries with 'None' frame from odom
    odom.drop(index=odom.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('odom after deleting entries with None frame or plans offset or footprint offset:')
    print(odom)
    print('\n')
    '''

    '''
    print('odom.shape after deleting entries with None frame or plans offset or footprint offset: ', odom.shape)
    print('\n')
    '''


    # Delete entries with 'None' frame from amcl_pose
    amcl_pose.drop(index=amcl_pose.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('amcl_pose after deleting entries with None frame or plans offset or footprint offset:')
    print(amcl_pose)
    print('\n')
    '''

    '''
    print('amcl_pose.shape after deleting entries with None frame or plans offset or footprint offset: ', amcl_pose.shape)
    print('\n')
    '''


    # Delete entries with 'None' frame from cmd_vel
    cmd_vel.drop(index=cmd_vel.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('cmd_vel after deleting entries with None frame or plans offset or footprint offset:')
    print(cmd_vel)
    print('\n')
    '''

    '''
    print('cmd_vel.shape after deleting entries with None frame or plans offset or footprint offset: ', cmd_vel.shape)
    print('\n')
    '''


    # Delete entries with 'None' frame from tf_odom_map
    tf_odom_map.drop(index=tf_odom_map.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('tf_odom_map after deleting entries with None frame or plans offset or footprint offset:')
    print(tf_odom_map)
    print('\n')
    '''

    '''
    print('tf_odom_map.shape after deleting entries with None frame or plans offset or footprint offset: ', tf_odom_map.shape)
    print('\n')
    '''


    # Delete entries with 'None' frame from tf_map_odom
    tf_map_odom.drop(index=tf_map_odom.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('tf_map_odom after deleting entries with None frame or plans offset or footprint offset:')
    print(tf_map_odom)
    print('\n')
    '''

    '''
    print('tf_map_odom.shape after deleting entries with None frame or plans offset or footprint offset: ', tf_map_odom.shape)
    print('\n')
    '''

    # Deletion of entries with 'None' frame from plans and footprints has not yet been implemented,
    # because after deleting rows from dataframes, indexes retain their values,
    # so that further plans' and footprints' instances can be indexed on the same way.

    return num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom




if explanation_alg == 'lime':

    # Data loading
    from lime_explainer import DataLoader
    
    # load input data
    odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data()
    '''
    print("---input loaded---")
    print('\n')
    '''

    # load output data
    cmd_vel = DataLoader.load_output_data()
    '''
    print("---output loaded---")
    print('\n')
    '''

    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints)

    costmap_size = local_costmap_info.iloc[0, 2]
    #print('costmap_size: ', costmap_size)
    
    # Dataset creation
    X_train = []
    X_test = []

    # if LIME image
    if explanation_mode == 'image':
        # output_class_name - not important for LIME image
        output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

        # Explanation
        from lime_explainer import ExplainNavigation

        exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                          current_goal, local_costmap_data, local_costmap_info,
                                                          amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                          X_train, X_test, tabular_mode, explanation_mode, num_samples,
                                                          output_class_name, num_of_first_rows_to_delete, footprints, test_type, costmap_size)

        # Representative situations/costmaps
        # New datasets:
        # Dataset1: #60, #165
        # Dataset2: #15 # 78 #163, #599
        # Old datasets:
        # Dataset1: #163
        # Dataset2:
        # Dataset3:
        # Dataset4: #100 #190
        # Dataset HARL Workshop 2021 paper: #71

        if test_type == 'single':
            # optional instance selection - deterministic
            #expID = 160 #27 #49 #68 #69 #77 #94 #97 #117 #131 #160 #184 #185 #227

            # random instance selection
            import random
            expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

            exp_nav.explain_instance(expID)
            #exp_nav.testSegmentation(expID)

        elif test_type == 'dataset_creation':
            #dataset_size = 1000
            for i in range(60, 61):
                # optional instance selection - deterministic
                #expID = 60

                # random instance selection
                #import random
                #expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

                exp_nav.explain_instance_dataset(i, i)

        elif test_type == 'evaluation':
            import time
            evaluation_sample_size = 1
            
            with open("explanations.txt", "w") as myfile:
                        myfile.write('explain_instance_time\n')
            
            for i in range(0, evaluation_sample_size):
                # optional instance selection - deterministic
                expID = 60

                # random instance selection
                # import random
                # expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

                start = time.time()
                exp_nav.explain_instance_evaluation(expID, i)
                end = time.time()
                with open("explanations.txt", "a") as myfile:
                    myfile.write(str(round(end - start, 4)) + '\n')

        elif test_type == 'GAN':
            # optional instance selection - deterministic
            #expID = 15

            # rando1m instance selection
            import random
            expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

            index = expID
            offset = num_of_first_rows_to_delete

            # Get local costmap
            # Original costmap will be saved to self.local_costmap_original
            local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
            
            # Make image a np.array deepcopy of local_costmap_original
            import numpy as np
            import copy
            image = np.array(copy.deepcopy(local_costmap_original))

            # '''
            # Turn inflated area to free space and 100s to 99s
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if 99 > image[i, j] > 0:
                        image[i, j] = 0
                    elif image[i, j] == 100:
                        image[i, j] = 99
            # '''

            # Turn every local costmap entry from int to float, so the segmentation algorithm works okay - here probably not needed
            image = image * 1.0

            flipped = True
            if flipped == True:
                image_flipped = np.flip(image, axis=1)

                import matplotlib.pyplot as plt
                fig = plt.figure(frameon=False)
                w = 1.6 #* 3
                h = 1.6 #* 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image_flipped.astype('float64'), aspect='auto')

                import pandas as pd

                costmap_info_tmp = local_costmap_info.iloc[index, :]
                costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
                costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

                # save costmap info to class variables
                localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
                localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
                localCostmapResolution = costmap_info_tmp.iloc[0, 0]
                localCostmapHeight = costmap_info_tmp.iloc[0, 2]
                localCostmapWidth = costmap_info_tmp.iloc[0, 1]

                odom_tmp = odom.iloc[index, :]
                odom_tmp = pd.DataFrame(odom_tmp).transpose()
                odom_tmp = odom_tmp.iloc[:, 2:]
                # save robot odometry location to class variables
                odom_x = odom_tmp.iloc[0, 0]
                odom_y = odom_tmp.iloc[0, 1]

                # save indices of robot's odometry location in local costmap to class variables
                localCostmapIndex_x_odom = 160 - int((odom_x - localCostmapOriginX) / localCostmapResolution)
                localCostmapIndex_y_odom = int((odom_y - localCostmapOriginY) / localCostmapResolution)

                # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
                x_odom_index = [localCostmapIndex_x_odom]
                y_odom_index = [localCostmapIndex_y_odom]

                # save robot odometry orientation to class variables
                # save robot odometry orientation to class variables
                odom_z = odom_tmp.iloc[0, 2] # minus je upitan
                znak = 1
                if odom_z < 0:
                    znak = -1
                odom_z = znak * (1 - abs(odom_z))
                odom_w = odom_tmp.iloc[0, 3]
                # calculate Euler angles based on orientation quaternion
                [yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
                # find yaw angles projections on x and y axes and save them to class variables
                yaw_odom_x = math.cos(yaw_odom)
                yaw_odom_y = math.sin(yaw_odom)

                local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
                local_plan_tmp = local_plan_tmp.iloc[:, 1:]
                # indices of local plan's poses in local costmap
                local_plan_x_list = []
                local_plan_y_list = []
                for i in range(1, local_plan_tmp.shape[0]):
                    x_temp = 160 - int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)

                tf_map_odom_tmp = tf_map_odom.iloc[index, :]
                tf_map_odom_tmp = pd.DataFrame(tf_map_odom_tmp).transpose()

                # transform global plan from /map to /odom frame
                # rotation matrix
                from scipy.spatial.transform import Rotation as R

                r = R.from_quat(
                    [tf_map_odom_tmp.iloc[0, 3], tf_map_odom_tmp.iloc[0, 4], tf_map_odom_tmp.iloc[0, 5],
                    tf_map_odom_tmp.iloc[0, 6]])
                # print('r: ', r.as_matrix())
                r_array = np.asarray(r.as_matrix())
                # print('r_array: ', r_array)
                # print('r_array.shape: ', r_array.shape)
                # translation vector
                t = np.array(
                    [tf_map_odom_tmp.iloc[0, 0], tf_map_odom_tmp.iloc[0, 1], tf_map_odom_tmp.iloc[0, 2]])
                # print('t: ', t)
                global_plan_tmp = teb_global_plan.loc[teb_global_plan['ID'] == index + offset]
                global_plan_tmp = global_plan_tmp.iloc[:, 1:]
                plan_tmp_tmp = copy.deepcopy(global_plan_tmp)
                for i in range(0, global_plan_tmp.shape[0]):
                    p = np.array(
                        [global_plan_tmp.iloc[i, 0], global_plan_tmp.iloc[i, 1], global_plan_tmp.iloc[i, 2]])
                    # print('p: ', p)
                    pnew = p.dot(r_array) + t
                    # print('pnew: ', pnew)
                    plan_tmp_tmp.iloc[i, 0] = pnew[0]
                    plan_tmp_tmp.iloc[i, 1] = pnew[1]
                    plan_tmp_tmp.iloc[i, 2] = pnew[2]

                # Get coordinates of the global plan in the local costmap
                # '''
                plan_x_list = []
                plan_y_list = []
                for i in range(0, plan_tmp_tmp.shape[0], 3):
                    x_temp = 160 - int(
                        (plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        #print('x_temp: ', x_temp)
                        #print('y_temp: ', y_temp)
                        #print('\n')
                        plan_x_list.append(x_temp)
                        plan_y_list.append(y_temp)
                ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
                # '''

                ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')
                
                # plot robots' location, orientation and local plan
                ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
                ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

                fig.savefig('input.png', transparent=False)
                fig.clf()

            else:
                import matplotlib.pyplot as plt
                fig = plt.figure(frameon=False)
                w = 1.6 #* 3
                h = 1.6 #* 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image.astype('float64'), aspect='auto')

                import pandas as pd

                costmap_info_tmp = local_costmap_info.iloc[index, :]
                costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
                costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

                # save costmap info to class variables
                localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
                localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
                localCostmapResolution = costmap_info_tmp.iloc[0, 0]
                localCostmapHeight = costmap_info_tmp.iloc[0, 2]
                localCostmapWidth = costmap_info_tmp.iloc[0, 1]

                odom_tmp = odom.iloc[index, :]
                odom_tmp = pd.DataFrame(odom_tmp).transpose()
                odom_tmp = odom_tmp.iloc[:, 2:]
                # save robot odometry location to class variables
                odom_x = odom_tmp.iloc[0, 0]
                odom_y = odom_tmp.iloc[0, 1]

                # save indices of robot's odometry location in local costmap to class variables
                localCostmapIndex_x_odom = int((odom_x - localCostmapOriginX) / localCostmapResolution)
                localCostmapIndex_y_odom = int((odom_y - localCostmapOriginY) / localCostmapResolution)

                # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
                x_odom_index = [localCostmapIndex_x_odom]
                y_odom_index = [localCostmapIndex_y_odom]

                # save robot odometry orientation to class variables
                odom_z = odom_tmp.iloc[0, 2]
                odom_w = odom_tmp.iloc[0, 3]
                # calculate Euler angles based on orientation quaternion
                [yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
                # find yaw angles projections on x and y axes and save them to class variables
                yaw_odom_x = math.cos(yaw_odom)
                yaw_odom_y = math.sin(yaw_odom)

                local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
                local_plan_tmp = local_plan_tmp.iloc[:, 1:]
                # indices of local plan's poses in local costmap
                local_plan_x_list = []
                local_plan_y_list = []
                for i in range(1, local_plan_tmp.shape[0]):
                    x_temp = int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)

                tf_map_odom_tmp = tf_map_odom.iloc[index, :]
                tf_map_odom_tmp = pd.DataFrame(tf_map_odom_tmp).transpose()

                # transform global plan from /map to /odom frame
                # rotation matrix
                from scipy.spatial.transform import Rotation as R

                r = R.from_quat(
                    [tf_map_odom_tmp.iloc[0, 3], tf_map_odom_tmp.iloc[0, 4], tf_map_odom_tmp.iloc[0, 5],
                    tf_map_odom_tmp.iloc[0, 6]])
                # print('r: ', r.as_matrix())
                r_array = np.asarray(r.as_matrix())
                # print('r_array: ', r_array)
                # print('r_array.shape: ', r_array.shape)
                # translation vector
                t = np.array(
                    [tf_map_odom_tmp.iloc[0, 0], tf_map_odom_tmp.iloc[0, 1], tf_map_odom_tmp.iloc[0, 2]])
                # print('t: ', t)
                global_plan_tmp = teb_global_plan.loc[teb_global_plan['ID'] == index + offset]
                global_plan_tmp = global_plan_tmp.iloc[:, 1:]
                plan_tmp_tmp = copy.deepcopy(global_plan_tmp)
                for i in range(0, global_plan_tmp.shape[0]):
                    p = np.array(
                        [global_plan_tmp.iloc[i, 0], global_plan_tmp.iloc[i, 1], global_plan_tmp.iloc[i, 2]])
                    # print('p: ', p)
                    pnew = p.dot(r_array) + t
                    # print('pnew: ', pnew)
                    plan_tmp_tmp.iloc[i, 0] = pnew[0]
                    plan_tmp_tmp.iloc[i, 1] = pnew[1]
                    plan_tmp_tmp.iloc[i, 2] = pnew[2]

                # Get coordinates of the global plan in the local costmap
                # '''
                plan_x_list = []
                plan_y_list = []
                for i in range(0, plan_tmp_tmp.shape[0], 3):
                    x_temp = int(
                        (plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        plan_x_list.append(x_temp)
                        plan_y_list.append(y_temp)
                ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')

                ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')

                # plot robots' location, orientation and local plan
                ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
                ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

                # '''
                fig.savefig('input.png', transparent=False)
                fig.clf() 


            from GAN import gan            
            gan.predict()       

        elif test_type == 'LIMEvsGAN':
            num_iter = 50
            lime_time_avg = 0
            gan_time_avg = 0

            COUNTER = [0.0] * 10
            R_PERC = [0.0] * 10
            G_PERC = [0.0] * 10
            B_PERC = [0.0] * 10
            PERC_FROM_CHANNELS = [0.0] * 10
            PERC_NOT_FROM_CHANNELS = [0.0] * 10

            from test_color import *
            create_dict_my()

            import pandas as pd
            import time
            import numpy as np
            import copy
            import matplotlib.pyplot as plt

            with open("times.csv", "w") as myfile:
                    myfile.write("lime,gan\n")

            with open("percentages.csv", "w") as myfile:
                    myfile.write("color,R,G,B,RGB_after,RGB_from_iter\n")

            with open("local_plan.csv", "w") as myfile:
                    myfile.write("color,RGB_after,RGB_from_iter\n")

            with open("global_plan.csv", "w") as myfile:
                    myfile.write("color,RGB_after,RGB_from_iter\n")

            with open("obstacles.csv", "w") as myfile:
                    myfile.write("color,RGB_after,RGB_from_iter\n")

            with open("free_space.csv", "w") as myfile:
                    myfile.write("color,RGB_after,RGB_from_iter\n")

            #with open("robot_position.csv", "w") as myfile:
            #        myfile.write("color,RGB_after,RGB_from_iter\n")
                
            for num in range(0, num_iter):
                print('NUM:  ', num)

                lime_time_avg = 0
                gan_time_avg = 0
                
                # optional instance selection - deterministic
                #expID = 160 #254 #160 #135 #35 #2 #0 

                # random instance selection
                import random
                expID = random.randint(5, local_costmap_info.shape[0] - 5) # expID se trazi iz local_costmap_info

                # call LIME    
                time_before = time.time()
                exp_nav.explain_instance(expID)
                time_after = time.time()
                lime_time_avg += time_after - time_before
                #print('LIME exp time: ', time_after - time_before)           


                # call GAN
                # quality evaluation will be done on flipped images
                # Prepare data for GAN
                time_before = time.time()
                index = expID
                offset = num_of_first_rows_to_delete

                # Get local costmap
                local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
                
                # Make image a np.array deepcopy of local_costmap_original
                image = np.array(copy.deepcopy(local_costmap_original))

                # '''
                # Turn inflated area to free space and 100s to 99s
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        if 99 > image[i, j] > 0:
                            image[i, j] = 0
                        elif image[i, j] == 100:
                            image[i, j] = 99
                # '''

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
                image = image * 1.0

                # Get flipped input image    
                image_flipped = np.flip(image, axis=1)

                # plot input image
                fig = plt.figure(frameon=False)
                w = 1.6 #* 3
                h = 1.6 #* 3
                fig.set_size_inches(w, h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image_flipped.astype('float64'), aspect='auto')

                # get costmap info
                costmap_info_tmp = local_costmap_info.iloc[index, :]
                costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
                costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

                # save costmap info to class variables
                localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
                localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
                localCostmapResolution = costmap_info_tmp.iloc[0, 0]
                localCostmapHeight = costmap_info_tmp.iloc[0, 2]
                localCostmapWidth = costmap_info_tmp.iloc[0, 1]

                # get odometry info
                odom_tmp = odom.iloc[index, :]
                odom_tmp = pd.DataFrame(odom_tmp).transpose()
                odom_tmp = odom_tmp.iloc[:, 2:]
                # save robot odometry location to class variables
                odom_x = odom_tmp.iloc[0, 0]
                odom_y = odom_tmp.iloc[0, 1]

                # save indices of robot's odometry location in local costmap to class variables
                localCostmapIndex_x_odom = 160 - int((odom_x - localCostmapOriginX) / localCostmapResolution)
                localCostmapIndex_y_odom = int((odom_y - localCostmapOriginY) / localCostmapResolution)

                # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
                x_odom_index = [localCostmapIndex_x_odom]
                y_odom_index = [localCostmapIndex_y_odom]

                # save robot odometry orientation to class variables
                # save robot odometry orientation to class variables
                odom_z = odom_tmp.iloc[0, 2] # minus je upitan
                znak = 1
                if odom_z < 0:
                    znak = -1
                odom_z = znak * (1 - abs(odom_z))
                odom_w = odom_tmp.iloc[0, 3]
                # calculate Euler angles based on orientation quaternion
                [yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
                # find yaw angles projections on x and y axes and save them to class variables
                yaw_odom_x = math.cos(yaw_odom)
                yaw_odom_y = math.sin(yaw_odom)

                # get local plan
                local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
                local_plan_tmp = local_plan_tmp.iloc[:, 1:]
                # indices of local plan's poses in local costmap
                local_plan_x_list = []
                local_plan_y_list = []
                for i in range(1, local_plan_tmp.shape[0]):
                    x_temp = 160 - int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        local_plan_x_list.append(x_temp)
                        local_plan_y_list.append(y_temp)

                # get tranformation info
                tf_map_odom_tmp = tf_map_odom.iloc[index, :]
                tf_map_odom_tmp = pd.DataFrame(tf_map_odom_tmp).transpose()

                # transform global plan from /map to /odom frame
                # rotation matrix
                from scipy.spatial.transform import Rotation as R

                r = R.from_quat(
                    [tf_map_odom_tmp.iloc[0, 3], tf_map_odom_tmp.iloc[0, 4], tf_map_odom_tmp.iloc[0, 5],
                    tf_map_odom_tmp.iloc[0, 6]])
                # print('r: ', r.as_matrix())
                r_array = np.asarray(r.as_matrix())
                # print('r_array: ', r_array)
                # print('r_array.shape: ', r_array.shape)
                # translation vector
                t = np.array(
                    [tf_map_odom_tmp.iloc[0, 0], tf_map_odom_tmp.iloc[0, 1], tf_map_odom_tmp.iloc[0, 2]])
                # print('t: ', t)
                global_plan_tmp = teb_global_plan.loc[teb_global_plan['ID'] == index + offset]
                global_plan_tmp = global_plan_tmp.iloc[:, 1:]
                plan_tmp_tmp = copy.deepcopy(global_plan_tmp)
                for i in range(0, global_plan_tmp.shape[0]):
                    p = np.array(
                        [global_plan_tmp.iloc[i, 0], global_plan_tmp.iloc[i, 1], global_plan_tmp.iloc[i, 2]])
                    # print('p: ', p)
                    pnew = p.dot(r_array) + t
                    # print('pnew: ', pnew)
                    plan_tmp_tmp.iloc[i, 0] = pnew[0]
                    plan_tmp_tmp.iloc[i, 1] = pnew[1]
                    plan_tmp_tmp.iloc[i, 2] = pnew[2]

                # Get coordinates of the global plan in the local costmap
                # '''
                plan_x_list = []
                plan_y_list = []
                for i in range(0, plan_tmp_tmp.shape[0], 3):
                    x_temp = 160 - int(
                        (plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        #print('x_temp: ', x_temp)
                        #print('y_temp: ', y_temp)
                        #print('\n')
                        plan_x_list.append(x_temp)
                        plan_y_list.append(y_temp)
                # plot global plan        
                ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
                # '''

                # plot local plan
                ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')
                
                # plot robots' location and orientation
                ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
                ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

                fig.savefig('input.png', transparent=False)
                fig.clf()

                from GAN import gan            
                gan.predict()

                time_after = time.time()
                gan_time_avg += time_after - time_before

                print('LIME time: ', lime_time_avg / num_iter)
                print('GAN time: ', gan_time_avg / num_iter)

                with open("times.csv", "a") as myfile:
                    myfile.write(str(lime_time_avg) + "," + str(gan_time_avg) + "\n")

                segments = exp_nav.getSegmentsForEval(image)
                #print('exp_nav.exp: ', exp_nav.exp)
                #plt.imshow(segments)
                #plt.savefig('SEGMENTS.png')

                segments = np.flip(segments, axis=1)

                #pd.DataFrame(segments).to_csv('SEGMENTS.csv', index=False)


                # RGB evaluation
                import PIL.Image
                import os
                path1 = os.getcwd() + '/flipped_explanation.png'
                exp_lime_orig = PIL.Image.open(path1).convert('RGB')
                path1 = os.getcwd() + '/GAN.png'
                exp_gan_orig = PIL.Image.open(path1).convert('RGB')

                exp_lime = np.array(exp_lime_orig)
                #print('exp_lime.shape: ', exp_lime.shape)
                exp_gan = np.array(exp_gan_orig)
                #print('exp_gan.shape: ', exp_gan.shape)

                #pd.DataFrame(exp_lime[:,:,0]).to_csv("exp_lime_R.csv")
                #pd.DataFrame(exp_lime[:,:,1]).to_csv("exp_lime_G.csv")
                #pd.DataFrame(exp_lime[:,:,2]).to_csv("exp_lime_B.csv")

                #pd.DataFrame(exp_gan[:,:,0]).to_csv("exp_gan_R.csv")
                #pd.DataFrame(exp_gan[:,:,1]).to_csv("exp_gan_G.csv")
                #pd.DataFrame(exp_gan[:,:,2]).to_csv("exp_gan_B.csv")

                #seg_unique = np.unique(segments)

                avg_R_list = []
                avg_G_list = []
                avg_B_list = []

                diff_R_list = []
                diff_G_list = []
                diff_B_list = []

                diff_R_percent_list = []
                diff_G_percent_list = []
                diff_B_percent_list = []

                avg_list = []
                diff_list = []
                diff_percent_list = []

                avg_avg_list = []
                avg_diff_list = []
                avg_diff_percent_list = []

                color_coverage_percent = []

                weights = []

                for e in exp_nav.exp:
                    if abs(e[1]) >= 0.0:
                        count = 0
                        
                        avg_R = 0
                        avg_G = 0
                        avg_B = 0

                        diff_R = 0
                        diff_G = 0
                        diff_B = 0

                        avg_avg = 0
                        avg_diff = 0

                        same_color_count = 0

                        weights.append(abs(e[1]))

                        for row in range(0, segments.shape[0]):
                            for columns in range(0, segments.shape[1]):
                                if segments[row, columns] == e[0]:
                                    #print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                                    #print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                                    #print('\n')

                                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                                    if lime_color_name == gan_color_name:
                                        same_color_count += 1

                                    count += 1

                                    avg_R += int(exp_lime[row, columns, 0])

                                    avg_G += int(exp_lime[row, columns, 1])

                                    avg_B += int(exp_lime[row, columns, 2])

                                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                                    avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                                    avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
                                    

                        avg_R /= count
                        if avg_R == 0:
                            avg_R = 1
                        avg_G /= count
                        if avg_G == 0:
                            avg_G = 1
                        avg_B /= count
                        if avg_B == 0:
                            avg_B = 1

                        diff_R /= count
                        diff_G /= count
                        diff_B /= count

                        avg_R_list.append(avg_R)
                        avg_G_list.append(avg_G)
                        avg_B_list.append(avg_B)

                        diff_R_list.append(diff_R)
                        diff_G_list.append(diff_G)
                        diff_B_list.append(diff_B)

                        diff_R_percent_list.append(100 * diff_R / avg_R)
                        diff_G_percent_list.append(100 * diff_G / avg_G)
                        diff_B_percent_list.append(100 * diff_B / avg_B)

                        avg = (avg_R + avg_G + avg_B) / 3
                        avg_list.append(avg)
                        diff = (diff_R + diff_G + diff_B) / 3
                        diff_list.append(diff)
                        diff_percent_list.append(100 * diff / avg) 

                        avg_avg /= count
                        avg_diff /= count
                        avg_avg_list.append(avg_avg)
                        avg_diff_list.append(avg_diff)
                        avg_diff_percent_list.append(100 * avg_diff / avg_avg)

                        color_coverage_percent.append(100 * same_color_count / count)

                '''       
                print('\n')

                print('expID: ', expID)
                print('\n')

                print('LIME time: ', lime_time_avg / num_iter)
                print('\n')
                print('GAN time: ', gan_time_avg / num_iter)
                print('\n')

                print('exp_nav.exp: ', exp_nav.exp)
                print('\n')

                print('avg_R_list: ', avg_R_list)
                print('\n')
                print('avg_G_list: ', avg_G_list)
                print('\n')
                print('avg_B_list: ', avg_B_list)
                print('\n')

                print('diff_R_list: ', diff_R_list)
                print('\n')
                print('diff_G_list: ', diff_G_list)
                print('\n')
                print('diff_B_list: ', diff_B_list)
                print('\n')

                print('diff_R_percent_list (%): ', diff_R_percent_list)
                print('\n')
                print('diff_G_percent_list (%):  ', diff_G_percent_list)
                print('\n')
                print('diff_B_percent_list (%): ', diff_B_percent_list)
                print('\n')

                print('avg_list: ', avg_list)
                print('\n')
                print('diff_list: ', diff_list)
                print('\n')
                print('diff_percent_list (%): ', diff_percent_list)
                print('\n')

                print('avg_avg_list: ', avg_avg_list)
                print('\n')
                print('avg_diff_list: ', avg_diff_list)
                print('\n')
                print('avg_diff_percent_list (%): ', avg_diff_percent_list)
                print('\n')

                print('color_coverage_percent_list (%): ', color_coverage_percent)
                print('\n')
                '''


                weights_sum = sum(weights)
                
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                weights_percentage = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += color_coverage_percent[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(color_coverage_percent[i] * weights[i] / weights_sum)
                    weights_percentage.append(100 * weights[i] / weights_sum)
                '''    
                print('weights: ', weights)
                print('\n')
                print('weights_percentage_list (%): ', weights_percentage)
                print('\n')
                print('explanation_saved_percentage_list_color (%): ', explanation_saved_percentage_list)
                print('\n')
                print('explanation_saved_percentage_color (%): ', explanation_saved_percentage)
                print('\n')
                '''
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",")


                avg_similarity_percentage = []
                for i in range(0, len(diff_R_percent_list)):
                    avg_similarity_percentage.append(100.0 - diff_R_percent_list[i])
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",") 


                avg_similarity_percentage = []
                for i in range(0, len(diff_G_percent_list)):
                    avg_similarity_percentage.append(100.0 - diff_G_percent_list[i])
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",")


                avg_similarity_percentage = []
                for i in range(0, len(diff_B_percent_list)):
                    avg_similarity_percentage.append(100.0 - diff_B_percent_list[i])
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",")           


                avg_similarity_percentage = []
                for i in range(0, len(diff_percent_list)):
                    avg_similarity_percentage.append(100.0 - diff_percent_list[i])
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)
                '''
                print('weights: ', weights)
                print('\n')
                print('weights_percentage_list (%): ', weights_percentage)
                print('\n')
                print('explanation_saved_percentage_list_avg (%): ', explanation_saved_percentage_list)
                print('\n')
                print('explanation_saved_percentage_avg (%): ', explanation_saved_percentage)
                print('\n')
                '''
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",")

                avg_avg_similarity_percentage = []
                for i in range(0, len(avg_diff_percent_list)):
                    avg_avg_similarity_percentage.append(100.0 - avg_diff_percent_list[i])
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += avg_avg_similarity_percentage[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(avg_avg_similarity_percentage[i] * weights[i] / weights_sum)
                '''
                print('weights: ', weights)
                print('\n')
                print('weights_percentage_list (%): ', weights_percentage)
                print('\n')
                print('explanation_saved_percentage_list_avg_avg (%): ', explanation_saved_percentage_list)
                print('\n')
                print('explanation_saved_percentage_avg_avg (%): ', explanation_saved_percentage)
                print('\n')
                '''
                with open("percentages.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + "\n")



                # LOCAL PLAN eval 
                same_color_count = 0
                count = 0

                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0
                for i in range(0, len(local_plan_x_list)):
                    count += 1
                    row = local_plan_y_list[i]
                    columns = local_plan_x_list[i]
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    avg_R += int(exp_lime[row, columns, 0])

                    avg_G += int(exp_lime[row, columns, 1])

                    avg_B += int(exp_lime[row, columns, 2])

                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                    avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                    avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
    
                color_coverage_percent = 100 * same_color_count / count
                #print('LOCAL PLAN color_coverage_percent (%): ', color_coverage_percent)
                #print('\n')

                avg_R /= count
                avg_G /= count
                avg_B /= count
                
                if avg_R == 0:
                    avg_R = 1
                if avg_G == 0:
                    avg_G = 1
                if avg_B == 0:
                    avg_B = 1

                diff_R /= count
                diff_G /= count
                diff_B /= count

                diff_R_percent = 100 * diff_R / avg_R
                diff_G_percent = 100 * diff_G / avg_G
                diff_B_percent = 100 * diff_B / avg_B

                avg = (avg_R + avg_G + avg_B) / 3
                diff = (diff_R + diff_G + diff_B) / 3
                diff_percent = 100 * diff / avg 

                avg_avg /= count
                if avg_avg == 0:
                    avg_avg = 1
                avg_diff /= count
                avg_diff_percent = 100 * avg_diff / avg_avg

                '''
                print('LOCAL PLAN avg_R: ', avg_R)
                print('\n')
                print('LOCAL PLAN avg_G: ', avg_G)
                print('\n')
                print('LOCAL PLAN avg_B: ', avg_B)
                print('\n')

                print('LOCAL PLAN diff_R: ', diff_R)
                print('\n')
                print('LOCAL PLAN diff_G: ', diff_G)
                print('\n')
                print('LOCAL PLAN diff_B: ', diff_B)
                print('\n')

                print('LOCAL PLAN diff_R_percent (%): ', diff_R_percent)
                print('\n')
                print('LOCAL PLAN diff_G_percent (%):  ', diff_G_percent)
                print('\n')
                print('LOCAL PLAN diff_B_percent (%): ', diff_B_percent)
                print('\n')

                print('LOCAL PLAN avg: ', avg)
                print('\n')
                print('LOCAL PLAN diff: ', diff)
                print('\n')
                print('LOCAL PLAN diff_percent (%): ', diff_percent)
                print('\n')

                print('LOCAL PLAN avg_avg: ', avg_avg)
                print('\n')
                print('LOCAL PLAN avg_diff: ', avg_diff)
                print('\n')
                print('LOCAL PLAN avg_diff_percent (%): ', avg_diff_percent)
                print('\n')
                '''

                with open("local_plan.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + "," + str(100 - diff_percent) + "," + str(100 - avg_diff_percent) + "\n")


                # GLOBAL PLAN eval 
                same_color_count = 0
                count = 0

                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0
                for i in range(0, len(plan_x_list)):
                    count += 1
                    row = plan_y_list[i]
                    columns = plan_x_list[i]
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    avg_R += int(exp_lime[row, columns, 0])

                    avg_G += int(exp_lime[row, columns, 1])

                    avg_B += int(exp_lime[row, columns, 2])

                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                    avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                    avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
    
                color_coverage_percent = 100 * same_color_count / count
                #print('GLOBAL PLAN color_coverage_percent (%): ', color_coverage_percent)
                #print('\n')

                avg_R /= count
                avg_G /= count
                avg_B /= count

                if avg_R == 0:
                    avg_R = 1
                if avg_G == 0:
                    avg_G = 1
                if avg_B == 0:
                    avg_B = 1

                diff_R /= count
                diff_G /= count
                diff_B /= count

                diff_R_percent = 100 * diff_R / avg_R
                diff_G_percent = 100 * diff_G / avg_G
                diff_B_percent = 100 * diff_B / avg_B

                avg = (avg_R + avg_G + avg_B) / 3
                diff = (diff_R + diff_G + diff_B) / 3
                diff_percent = 100 * diff / avg 

                avg_avg /= count
                if avg_avg == 0:
                    avg_avg = 1
                avg_diff /= count
                avg_diff_percent = 100 * avg_diff / avg_avg
                '''
                print('GLOBAL PLAN avg_R: ', avg_R)
                print('\n')
                print('GLOBAL PLAN avg_G: ', avg_G)
                print('\n')
                print('GLOBAL PLAN avg_B: ', avg_B)
                print('\n')

                print('GLOBAL PLAN diff_R: ', diff_R)
                print('\n')
                print('GLOBAL PLAN diff_G: ', diff_G)
                print('\n')
                print('GLOBAL PLAN diff_B: ', diff_B)
                print('\n')

                print('GLOBAL PLAN diff_R_percent (%): ', diff_R_percent)
                print('\n')
                print('GLOBAL PLAN diff_G_percent (%):  ', diff_G_percent)
                print('\n')
                print('GLOBAL PLAN diff_B_percent (%): ', diff_B_percent)
                print('\n')

                print('GLOBAL PLAN avg: ', avg)
                print('\n')
                print('GLOBAL PLAN diff: ', diff)
                print('\n')
                print('GLOBAL PLAN diff_percent (%): ', diff_percent)
                print('\n')

                print('GLOBAL PLAN avg_avg: ', avg_avg)
                print('\n')
                print('GLOBAL PLAN avg_diff: ', avg_diff)
                print('\n')
                print('GLOBAL PLAN avg_diff_percent (%): ', avg_diff_percent)
                print('\n')
                '''

                with open("global_plan.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + "," + str(100 - diff_percent) + "," + str(100 - avg_diff_percent) + "\n")


                # OBSTACLES eval 
                same_color_count = 0
                count = 0

                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0
                for i in range(0, image_flipped.shape[0]):
                    for j in range(0, image_flipped.shape[1]):
                        if image_flipped[i, j] == 99:
                            count += 1
                            row = i
                            columns = j
                            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                            if lime_color_name == gan_color_name:
                                same_color_count += 1

                            avg_R += int(exp_lime[row, columns, 0])

                            avg_G += int(exp_lime[row, columns, 1])

                            avg_B += int(exp_lime[row, columns, 2])

                            diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                            diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                            diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                            avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                            avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                            - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
        
                color_coverage_percent = 100 * same_color_count / count
                #print('OBSTACLES color_coverage_percent (%): ', color_coverage_percent)
                #print('\n')

                avg_R /= count
                avg_G /= count
                avg_B /= count

                if avg_R == 0:
                    avg_R = 1
                if avg_G == 0:
                    avg_G = 1
                if avg_B == 0:
                    avg_B = 1

                diff_R /= count
                diff_G /= count
                diff_B /= count

                diff_R_percent = 100 * diff_R / avg_R
                diff_G_percent = 100 * diff_G / avg_G
                diff_B_percent = 100 * diff_B / avg_B

                avg = (avg_R + avg_G + avg_B) / 3
                diff = (diff_R + diff_G + diff_B) / 3
                diff_percent = 100 * diff / avg 

                avg_avg /= count
                avg_diff /= count
                if avg_avg == 0:
                    avg_avg = 1
                avg_diff_percent = 100 * avg_diff / avg_avg
                '''
                print('OBSTACLES avg_R: ', avg_R)
                print('\n')
                print('OBSTACLES avg_G: ', avg_G)
                print('\n')
                print('OBSTACLES avg_B: ', avg_B)
                print('\n')

                print('OBSTACLES diff_R: ', diff_R)
                print('\n')
                print('OBSTACLES diff_G: ', diff_G)
                print('\n')
                print('OBSTACLES diff_B: ', diff_B)
                print('\n')

                print('OBSTACLES diff_R_percent (%): ', diff_R_percent)
                print('\n')
                print('OBSTACLES diff_G_percent (%):  ', diff_G_percent)
                print('\n')
                print('OBSTACLES diff_B_percent (%): ', diff_B_percent)
                print('\n')

                print('OBSTACLES avg: ', avg)
                print('\n')
                print('OBSTACLES diff: ', diff)
                print('\n')
                print('OBSTACLES diff_percent (%): ', diff_percent)
                print('\n')

                print('OBSTACLES avg_avg: ', avg_avg)
                print('\n')
                print('OBSTACLES avg_diff: ', avg_diff)
                print('\n')
                print('OBSTACLES avg_diff_percent (%): ', avg_diff_percent)
                print('\n')
                '''

                with open("obstacles.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + "," + str(100 - diff_percent) + "," + str(100 - avg_diff_percent) + "\n")


                # FREE SPACE eval 
                same_color_count = 0
                count = 0

                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0
                for i in range(0, image_flipped.shape[0]):
                    for j in range(0, image_flipped.shape[1]):
                        if image_flipped[i, j] == 0:
                            count += 1
                            row = i
                            columns = j
                            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                            if lime_color_name == gan_color_name:
                                same_color_count += 1

                            avg_R += int(exp_lime[row, columns, 0])

                            avg_G += int(exp_lime[row, columns, 1])

                            avg_B += int(exp_lime[row, columns, 2])

                            diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                            diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                            diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                            avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                            avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                            - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
        
                color_coverage_percent = 100 * same_color_count / count
                #print('FREE SPACE color_coverage_percent (%): ', color_coverage_percent)
                #print('\n')

                avg_R /= count
                avg_G /= count
                avg_B /= count

                if avg_R == 0:
                    avg_R = 1
                if avg_G == 0:
                    avg_G = 1
                if avg_B == 0:
                    avg_B = 1

                diff_R /= count
                diff_G /= count
                diff_B /= count

                diff_R_percent = 100 * diff_R / avg_R
                diff_G_percent = 100 * diff_G / avg_G
                diff_B_percent = 100 * diff_B / avg_B

                avg = (avg_R + avg_G + avg_B) / 3
                diff = (diff_R + diff_G + diff_B) / 3
                diff_percent = 100 * diff / avg 

                avg_avg /= count
                if avg_avg == 0:
                    avg_avg = 1
                avg_diff /= count
                avg_diff_percent = 100 * avg_diff / avg_avg
                '''
                print('FREE SPACE avg_R: ', avg_R)
                print('\n')
                print('FREE SPACE avg_G: ', avg_G)
                print('\n')
                print('FREE SPACE avg_B: ', avg_B)
                print('\n')

                print('FREE SPACE diff_R: ', diff_R)
                print('\n')
                print('FREE SPACE diff_G: ', diff_G)
                print('\n')
                print('FREE SPACE diff_B: ', diff_B)
                print('\n')

                print('FREE SPACE diff_R_percent (%): ', diff_R_percent)
                print('\n')
                print('FREE SPACE diff_G_percent (%):  ', diff_G_percent)
                print('\n')
                print('FREE SPACE diff_B_percent (%): ', diff_B_percent)
                print('\n')

                print('FREE SPACE avg: ', avg)
                print('\n')
                print('FREE SPACE diff: ', diff)
                print('\n')
                print('FREE SPACE diff_percent (%): ', diff_percent)
                print('\n')

                print('FREE SPACE avg_avg: ', avg_avg)
                print('\n')
                print('FREE SPACE avg_diff: ', avg_diff)
                print('\n')
                print('FREE SPACE avg_diff_percent (%): ', avg_diff_percent)
                print('\n')
                '''

                with open("free_space.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + "," + str(100 - diff_percent) + "," + str(100 - avg_diff_percent) + "\n")

                '''
                # ROBOT POSITION eval 
                same_color_count = 0
                count = 0

                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0
                for i in range(0, len(x_odom_index)):
                    count += 1
                    row = y_odom_index[i]
                    columns = x_odom_index[i]
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    avg_R += int(exp_lime[row, columns, 0])

                    avg_G += int(exp_lime[row, columns, 1])

                    avg_B += int(exp_lime[row, columns, 2])

                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))

                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))

                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) 

                    avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                    avg_diff += abs((int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3)
    
                color_coverage_percent = 100 * same_color_count / count
                #print('ROBOT POSITION color_coverage_percent (%): ', color_coverage_percent)
                #print('\n')

                avg_R /= count
                avg_G /= count
                avg_B /= count

                if avg_R == 0:
                    avg_R = 1
                if avg_G == 0:
                    avg_G = 1
                if avg_B == 0:
                    avg_B = 1

                diff_R /= count
                diff_G /= count
                diff_B /= count

                diff_R_percent = 100 * diff_R / avg_R
                diff_G_percent = 100 * diff_G / avg_G
                diff_B_percent = 100 * diff_B / avg_B

                avg = (avg_R + avg_G + avg_B) / 3
                diff = (diff_R + diff_G + diff_B) / 3
                diff_percent = 100 * diff / avg 

                avg_avg /= count
                if avg_avg == 0:
                    avg_avg = 1
                avg_diff /= count
                avg_diff_percent = 100 * avg_diff / avg_avg
                '''

                '''
                print('ROBOT POSITION avg_R: ', avg_R)
                print('\n')
                print('ROBOT POSITION avg_G: ', avg_G)
                print('\n')
                print('ROBOT POSITION avg_B: ', avg_B)
                print('\n')

                print('ROBOT POSITION diff_R: ', diff_R)
                print('\n')
                print('ROBOT POSITION diff_G: ', diff_G)
                print('\n')
                print('ROBOT POSITION diff_B: ', diff_B)
                print('\n')

                print('ROBOT POSITION diff_R_percent (%): ', diff_R_percent)
                print('\n')
                print('ROBOT POSITION diff_G_percent (%):  ', diff_G_percent)
                print('\n')
                print('ROBOT POSITION diff_B_percent (%): ', diff_B_percent)
                print('\n')

                print('ROBOT POSITION avg: ', avg)
                print('\n')
                print('ROBOT POSITION diff: ', diff)
                print('\n')
                print('ROBOT POSITION diff_percent (%): ', diff_percent)
                print('\n')

                print('ROBOT POSITION avg_avg: ', avg_avg)
                print('\n')
                print('ROBOT POSITION avg_diff: ', avg_diff)
                print('\n')
                print('ROBOT POSITION avg_diff_percent (%): ', avg_diff_percent)
                print('\n')
                '''

                #with open("robot_position.csv", "a") as myfile:
                #    myfile.write(str(color_coverage_percent) + "," + str(100 - diff_percent) + "," + str(100 - avg_diff_percent) + "\n")    


                
                '''
                # The Intersection over Union (IoU) metric, also referred to as the Jaccard index
                intersection = np.logical_and(exp_lime, exp_gan)
                print('intersection.shape: ', intersection.shape)
                union = np.logical_or(exp_lime, exp_gan)
                print('union.shape: ', union.shape)
                iou_score = np.sum(intersection) / np.sum(union)
                print('np.sum(intersection): ', np.sum(intersection))
                print('np.sum(union): ', np.sum(union))
                print('iou_score: ', iou_score) 
                print('\n\n\n')
                '''

                '''
                for id in range(0, len(diff_percent_list)):
                    COUNTER[id] += 1
                    R_PERC[id] += diff_R_percent_list[id]
                    G_PERC[id] += diff_G_percent_list[id]
                    B_PERC[id] += diff_B_percent_list[id]
                    PERC_FROM_CHANNELS[id] += diff_percent_list[id]
                    PERC_NOT_FROM_CHANNELS[id] += avg_diff_percent_list[id]
                '''

                '''
                # Grayscale evaluation
                exp_lime_gs = exp_lime_orig.convert('L')
                exp_lime_gs.save('lime_gs.png')

                exp_gan_gs = exp_gan_orig.convert('L')
                exp_gan_gs.save('gan_gs.png')

                exp_lime = np.array(exp_lime_gs)
                print('exp_lime.shape: ', exp_lime.shape)
                exp_gan = np.array(exp_gan_gs)
                print('exp_gan.shape: ', exp_gan.shape)

                avg_list = []
                diff_list = []
                diff_percent_list = []

                same_pixels_count_list = []

                for e in exp_nav.exp:
                    if abs(e[1]) >= 0.1:
                        count = 0
                        same_pixels_count = 0

                        avg = 0
                        diff = 0
                        
                        for row in range(0, segments.shape[0]):
                            for columns in range(0, segments.shape[1]):
                                if segments[row, columns] == e[0]:
                                    count += 1

                                    avg += int(exp_lime[row, columns])

                                    diff += abs( int(exp_gan[row, columns]) - int(exp_lime[row, columns]) )
                                    
                                    if exp_lime[row, columns] == exp_gan[row, columns]:
                                        same_pixels_count += 1
                                        

                        avg_list.append(avg)
                        diff_list.append(diff)
                        diff_percent_list.append(100 * diff / avg) 

                        same_pixels_count_list.append(100 * same_pixels_count / count)

                        
                print('\n')

                print('expID: ', expID)
                print('\n')

                print('LIME time: ', lime_time_avg / num_iter)
                print('\n')
                print('GAN time: ', gan_time_avg / num_iter)
                print('\n')

                print('exp_nav.exp: ', exp_nav.exp)
                print('\n')

                print('avg_list: ', avg_list)
                print('\n')
                print('diff_list: ', diff_list)
                print('\n')
                print('diff_percent_list (%): ', diff_percent_list)
                print('\n')

                print('same_pixels_count_list (%): ', same_pixels_count_list)
                print('\n')

                # The Intersection over Union (IoU) metric, also referred to as the Jaccard inde
                intersection = np.logical_and(exp_lime, exp_gan)
                print('intersection.shape: ', intersection.shape)
                union = np.logical_or(exp_lime, exp_gan)
                print('union.shape: ', union.shape)
                iou_score = np.sum(intersection) / np.sum(union)
                print('np.sum(intersection): ', np.sum(intersection))
                print('np.sum(union): ', np.sum(union))
                print('iou_score: ', iou_score)

                print('iter: ', num)
                '''

            '''
            for id in range(0, 10):
                if COUNTER[id] > 0.0:
                    R_PERC[id] /= COUNTER[id]
                    G_PERC[id] /= COUNTER[id]
                    B_PERC[id] /= COUNTER[id]
                    PERC_FROM_CHANNELS[id] /= COUNTER[id]
                    PERC_NOT_FROM_CHANNELS[id] /= COUNTER[id]

            print('COUNTER: ', COUNTER) 
            print('R_PERC: ', R_PERC)
            print('G_PERC: ', G_PERC)
            print('B_PERC: ', B_PERC)
            print('PERC_FROM_CHANNELS: ', PERC_FROM_CHANNELS)
            print('PERC_NOT_FROM_CHANNELS: ', PERC_NOT_FROM_CHANNELS)
            '''

        


'''
# Dataset creation
X_train = []
X_test = []

# If the LIME tabular is to be used
# ' cmd_vel_ang_z' - sometime in the future to correct this gap - delete it
if explanation_mode == 'tabular':
    from lime_explainer import DatasetCreator
    
    # Select input for explanation algorithm
    X = odom.iloc[:, 6:8]  # input for explanation are odometry velocities
    # print(X)

    if mode == 'regression':
        import numpy as np

        # regression
        # Selecting the output for the explanation algorithm
        index_output_class = 1 # [0] - command linear velocity, [1] - command angular velocity
        y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
        #print(y)        
        output_class_name = y.columns.values[0]

    elif (mode == 'classification') & (one_hot_encoding == False):
        import numpy as np
        
        # classification        
        # left-right-straight logic
        conditions = [
            (cmd_vel[' cmd_vel_ang_z'] >= 0),
            (cmd_vel[' cmd_vel_ang_z'] < 0)
            ]

        values = ['left', 'right']

        cmd_vel['direction'] = np.select(conditions, values)
        
        # Selecting the output for the explanation algorithm
        index_output_class = 2 # [2] - direction
        y = cmd_vel.iloc[:,index_output_class:index_output_class+1] # the output for explanation is direction
        #print(y)
        output_class_name = y.columns.values[0]

    elif (mode == 'classification') & (one_hot_encoding == True):
        import numpy as np
        
        # random forest classification - one-hot encoding        
        # left-right-straight logic
        conditions = [
            (cmd_vel[' cmd_vel_ang_z'] >= 0),
            (cmd_vel[' cmd_vel_ang_z'] < 0)
            ]

        # one-hot left-right coding
        valuesLeft = [1.0, 0.0]

        cmd_vel['left'] = np.select(conditions, valuesLeft)

        valuesRight = [0.0, 1.0]

        cmd_vel['right'] = np.select(conditions, valuesRight)
        
        # Selecting the output for the explanation algorithm
        index_output_class = 2 # [2] - 'left', [3] - 'right'
        y = cmd_vel.iloc[:,index_output_class:index_output_class+2] # the explanation output is direction, i.e. left, right and straight one-hot encoded
        #print(y)
        output_class_name = y.columns.values[0] # 'left' - [0] or 'right' - [1]

    
    import random
    #randomNum  = random.randint(0, 100)
    randomNum = 42
    
    slice_ratio = 0.01 # very small slice_ratio ensures that almost all data is put in X_train
    
    X_train, X_test, y_train, y_test = DatasetCreator.split_test_train(X, y, slice_ratio, randomNum) # Row names (indexes) remain preserved even after mixing - very good


    # Ensuring that this data is Dataframes, but not necessary, because they already are. Let it stand here for printing while doing possible debugging process.
    import pandas as pd
    X_train = pd.DataFrame(X_train)
    print(X_train)
    y_train = pd.DataFrame(y_train)
    print(y_train)
    
    X_test = pd.DataFrame(X_test)
    print(X_test)
    y_test = pd.DataFrame(y_test)
    print(y_test)


# If a LIME image or LIME tabular with costmap as an input will be used
if (explanationMode == 'image') | (explanationMode == 'tabular_costmap'):
    import pandas as pd
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()
    

# Choosing expID
# choose/generate expID - ordinal number of the row in X_test or X_train from which the index is extracted
# if LIME tabular will be used
if explanationMode == 'tabular':
    # optional selection - deterministic
    expID = 86

    # Dataset1:
    # Dataset2:
    # Dataset3:
    
    # random selection
    #import random
    #expID = random.randint(0, X_train.shape[0]) # expID se trazi iz X_train
    
'''

