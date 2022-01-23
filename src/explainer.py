#!/usr/bin/env python3

# Global variables

# possible explanation algorithms: 'lime', 'anchors'
explanation_alg = ''

# possible explanation modes: 'tabular', 'image', 'costmap'
explanation_mode = ''

# tabular explanation modes: 'regression', 'classification'
tabular_mode = ''

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
    

    # Calculating offset
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
    
    '''
    # Deletion of entries with 'None' frame from plans and footprints has not yet been implemented,
    # because after deleting rows from dataframes, indexes retain their values,
    # so that further plans' and footprints' instances can be indexed on the same way.
    '''

    return num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom

def LimeSingle():
    import pandas as pd
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()

    output_class_name = ''
    num_samples = 100

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

    # preprocess data
    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints)
        
    if explanation_mode == 'tabular' or explanation_mode == 'tabular_costmap':
        from lime_explainer import DatasetCreator
    
        # Select input for explanation algorithm
        X = odom.iloc[:, 6:8]  # input for explanation are odometry velocities
        # print(X)

        one_hot_encoding = True

        if tabular_mode == 'regression':
            import numpy as np

            # regression
            # Selecting the output for the explanation algorithm
            index_output_class = 1 # [0] - command linear velocity, [1] - command angular velocity
            y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
            #print(y)        
            output_class_name = y.columns.values[0]

        elif (tabular_mode == 'classification') & (one_hot_encoding == False):
            import numpy as np
            
            # classification        
            # left-right-straight logic
            conditions = [
                (cmd_vel['cmd_vel_ang_z'] >= 0),
                (cmd_vel['cmd_vel_ang_z'] < 0)
                ]

            values = ['left', 'right']

            cmd_vel['direction'] = np.select(conditions, values)
            
            # Selecting the output for the explanation algorithm
            index_output_class = 2 # [2] - direction
            y = cmd_vel.iloc[:,index_output_class:index_output_class+1] # the output for explanation is direction
            #print(y)
            output_class_name = y.columns.values[0]

        elif (tabular_mode == 'classification') & (one_hot_encoding == True):
            import numpy as np
            
            # random forest classification - one-hot encoding        
            # left-right-straight logic
            conditions = [
                (cmd_vel['cmd_vel_ang_z'] >= 0),
                (cmd_vel['cmd_vel_ang_z'] < 0)
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
        X_train = pd.DataFrame(X_train)
        print('\nX_train: ')
        print(X_train)
        y_train = pd.DataFrame(y_train)
        print('\ny_train: ')
        print(y_train)
        
        X_test = pd.DataFrame(X_test)
        print('\nX_test: ')
        print(X_test)
        y_test = pd.DataFrame(y_test)
        print('\ny_test: ')
        print(y_test)

    # Explanation
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        tabular_mode, explanation_mode, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)
    
    choose_random_instance = False

    if choose_random_instance == True:
        # random instance selection
        print('\nexpID range: ', (0, local_costmap_info.shape[0] - num_of_first_rows_to_delete))
        import random
        expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)
        print('\nexpID: ', expID)
    else:     
        # optional instance selection - deterministic
        expID = 78 #DS1: #51 #78 #84 #144, #DS2: #260
        print('\nexpID: ', expID)

    exp_nav.explain_instance(expID)
    #exp_nav.testSegmentation(expID)

def CreateDataset():
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

    # output_class_name - not important for LIME image
    output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

    # Explanation
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        X_train, X_test, tabular_mode, explanation_mode, num_samples,
                                                        output_class_name, num_of_first_rows_to_delete, footprints, costmap_size)

    #'''
    with open('costmap_data.csv', "w") as myfile:
            myfile.write('picture_ID,width,height,origin_x,origin_y,resolution\n')

    with open('local_plan_coordinates.csv', "w") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open('global_plan_coordinates.csv', "w") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open('robot_coordinates.csv', "w") as myfile:
        myfile.write('picture_ID,position_x,position_y\n')
    #''' 

    dataset_size = local_costmap_info.shape[0]
    import random    
    for i in range(1, dataset_size):
        # optional instance selection - deterministic
        expID = i

        # random instance selection
        #expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete) 

        exp_nav.explain_instance_dataset(expID, i)
        #exp_nav.testSegmentation(expID)

def EvaluateLIME():
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

    # output_class_name - not important for LIME image
    output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

    # Explanation
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        X_train, X_test, tabular_mode, explanation_mode, num_samples,
                                                        output_class_name, num_of_first_rows_to_delete, footprints, costmap_size)

    import time
    evaluation_sample_size = 1
    
    with open("explanations.txt", "w") as myfile:
        myfile.write('explain_instance_time\n')
    
    for i in range(0, evaluation_sample_size):
        # optional instance selection - deterministic
        expID = 24

        # random instance selection
        #import random
        #expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete) 

        start = time.time()
        exp_nav.explain_instance_evaluation(expID, i)
        end = time.time()
        with open("explanations.txt", "a") as myfile:
            myfile.write(str(round(end - start, 2)) + '\n')

def RunGAN():
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

    # output_class_name - not important for LIME image
    output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

    # Explanation
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        X_train, X_test, tabular_mode, explanation_mode, num_samples,
                                                        output_class_name, num_of_first_rows_to_delete, footprints, costmap_size)

    # optional instance selection - deterministic
    #expID = 28

    # rando1m instance selection
    import random
    expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)

    print('expID: ', expID)
    print('\n')

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

    flipped = False

    if flipped == True:
        # za prvu verziju GAN-a
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
        #odom_z = odom_tmp.iloc[0, 2] # minus je upitan
        #odom_w = odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        #[yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
        #yaw_sign = math.copysign(1, self.yaw_odom)
        #self.yaw_odom = -1 * yaw_sign * (math.pi - abs(self.yaw_odom))
        # find yaw angles projections on x and y axes and save them to class variables
        #yaw_odom_x = math.cos(yaw_odom)
        #yaw_odom_y = math.sin(yaw_odom)

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
        # '''
        
        ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
        ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')
        # plot robots' location, orientation and local plan
        ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
        #ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

        fig.savefig('input.png', transparent=False)
        fig.clf()

    elif flipped == False:
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
        #odom_z = odom_tmp.iloc[0, 2]
        #odom_w = odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        #[yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
        # find yaw angles projections on x and y axes and save them to class variables
        #yaw_odom_x = math.cos(yaw_odom)
        #yaw_odom_y = math.sin(yaw_odom)

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
        t = np.array([tf_map_odom_tmp.iloc[0, 0], tf_map_odom_tmp.iloc[0, 1], tf_map_odom_tmp.iloc[0, 2]])
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
        # za novu verziju GAN-a isključiti crtanje orijentacije
        #ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')
        # '''
        
        fig.savefig('input.png', transparent=False)
        fig.clf() 


    from GAN import gan            
    gan.predict()

def EvaluateLIMEvsGAN():
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

    # output_class_name - not important for LIME image
    output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

    # Explanation
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        X_train, X_test, tabular_mode, explanation_mode, num_samples,
                                                        output_class_name, num_of_first_rows_to_delete, footprints, costmap_size)

    from test_color import create_dict_my
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
            myfile.write("color,RGB_after,RGB_from_iter,color_flip,color_turn,color_other\n")

    with open("free_space.csv", "w") as myfile:
            myfile.write("color,RGB_after,RGB_from_iter,color_flip,color_turn,color_other\n")

    with open("obstacles_weighted.csv", "w") as myfile:
            myfile.write("color,R,G,B,RGB_after,RGB_from_iter,color_flip,color_turn,color_other\n")

    with open("free_space_weighted.csv", "w") as myfile:
            myfile.write("color,R,G,B,RGB_after,RGB_from_iter,color_flip,color_turn,color_other\n")

    #with open("robot_position.csv", "w") as myfile:
    #        myfile.write("color,RGB_after,RGB_from_iter\n")

    num_iter = 1
    
    lime_time_avg = 0
    gan_time_avg = 0

    R_PERC = [0.0] * 10
    G_PERC = [0.0] * 10
    B_PERC = [0.0] * 10

    flipped = False

    exp_IDs_list_test_ds1 = [5, 23, 44, 52, 75, 88, 94, 104, 118, 128, 136, 150, 151, 189, 190, 209, 223, 225, 229, 242, 252]
    exp_IDs_list_test_ds2 = [6, 9, 12, 13, 22, 45, 63, 75, 103, 105, 109, 123, 126, 128, 150, 153, 154, 161, 166, 167, 182, 203, 214, 215, 220, 234, 237, 247, 249, 252, 257, 258, 262, 271, 275, 277, 278, 294, 337, 348, 366, 373, 387, 390, 391, 413, 420, 426, 430, 436, 441, 445, 446, 451, 455, 466, 468, 482, 492, 495, 505, 507, 514, 525, 580, 585, 599, 602, 612, 620, 625, 639, 640, 641, 667, 676, 688, 690, 698]
        
    for num in range(0, len(exp_IDs_list_test_ds1)):
    #for num in range(0, len(exp_IDs_list_test_ds2)):
    #for num in range(0, num_iter):
        print('iteration: ', num)

        lime_time_avg = 0
        gan_time_avg = 0
        
        # optional instance selection - deterministic
        expID = exp_IDs_list_test_ds1[num]
        #expID = exp_IDs_list_test_ds2[num]
        #expID = 203

        # random instance selection
        #import random
        #expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete) # expID se trazi iz local_costmap_info

        # call LIME    
        time_before = time.time()
        exp_nav.explain_instance(expID)
        time_after = time.time()
        lime_time_avg += time_after - time_before
        #print('LIME exp time: ', time_after - time_before)           


        # call GAN
        # Prepare data for GAN
        time_before = time.time()
        index = expID
        offset = num_of_first_rows_to_delete

        # Get local costmap
        local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
        
        # Make image a np.array deepcopy of local_costmap_original
        image = np.array(copy.deepcopy(local_costmap_original))

        #'''
        # Turn inflated area to free space and 100s to 99s
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if 99 > image[i, j] > 0:
                    image[i, j] = 0
                elif image[i, j] == 100:
                    image[i, j] = 99
        #'''

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        image = image * 1.0

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image.astype('float64'), aspect='auto')
        fig.savefig('costmap_original.png', transparent=False)
        fig.clf()
        '''                

        # Get flipped input image if wanted
        if flipped == True:    
            image_flipped = np.flip(image, axis=1)
        elif flipped == False:
            image_flipped = image

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
        if flipped == True:
            localCostmapIndex_x_odom = 160 - int((odom_x - localCostmapOriginX) / localCostmapResolution)
        elif flipped == False:
            localCostmapIndex_x_odom = int((odom_x - localCostmapOriginX) / localCostmapResolution)
        localCostmapIndex_y_odom = int((odom_y - localCostmapOriginY) / localCostmapResolution)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        x_odom_index = [localCostmapIndex_x_odom]
        y_odom_index = [localCostmapIndex_y_odom]

        # save robot odometry orientation to class variables
        #odom_z = odom_tmp.iloc[0, 2] 
        #odom_w = odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        #[yaw_odom, pitch_odom, roll_odom] = quaternion_to_euler(0.0, 0.0, odom_z, odom_w)
        '''
        if flipped == True:
            yaw_sign = math.copysign(1, self.yaw_odom)
            self.yaw_odom = -1 * yaw_sign * (math.pi - abs(self.yaw_odom))
        '''
        # find yaw angles projections on x and y axes and save them to class variables
        #yaw_odom_x = math.cos(yaw_odom)
        #yaw_odom_y = math.sin(yaw_odom)

        # get local plan
        local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
        local_plan_tmp = local_plan_tmp.iloc[:, 1:]
        # indices of local plan's poses in local costmap
        local_plan_x_list = []
        local_plan_y_list = []
        for i in range(1, local_plan_tmp.shape[0]):
            if flipped == True:
                x_temp = 160 - int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
            elif flipped == False:
                x_temp = int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
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
            if flipped == True:
                x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
            elif flipped == False:
                x_temp = int((plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)    
            y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                #print('x_temp: ', x_temp)
                #print('y_temp: ', y_temp)
                #print('\n')
                plan_x_list.append(x_temp)
                plan_y_list.append(y_temp)
        # '''

        # plot global plan        
        ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
        # plot local plan
        ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')
        # plot robots' location and orientation
        ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
        #ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

        fig.savefig('input.png', transparent=False)
        fig.clf()

        from GAN import gan            
        gan.predict()

        time_after = time.time()
        gan_time_avg += time_after - time_before

        print('LIME time: ', lime_time_avg / num_iter)
        print('\n')
        print('GAN time: ', gan_time_avg / num_iter)
        print('\n')

        with open("times.csv", "a") as myfile:
            myfile.write(str(lime_time_avg) + "," + str(gan_time_avg) + "\n")

        segments = exp_nav.getSegmentsForGanLimeEval(image)
        #print('exp_nav.exp: ', exp_nav.exp)
        #plt.imshow(segments)
        #plt.savefig('SEGMENTS.png')

        if flipped == True:
            segments = np.flip(segments, axis=1)

        #pd.DataFrame(segments).to_csv('SEGMENTS.csv', index=False)


        # RGB evaluation
        import PIL.Image
        import os
        if flipped == True:
            path1 = os.getcwd() + '/flipped_explanation.png'
        elif flipped == False:
            path1 = os.getcwd() + '/explanation.png'
        exp_lime_orig = PIL.Image.open(path1).convert('RGB')
        path1 = os.getcwd() + '/GAN.png'
        exp_gan_orig = PIL.Image.open(path1).convert('RGB')

        exp_lime = np.array(exp_lime_orig)
        #print('exp_lime.shape: ', exp_lime.shape)
        exp_gan = np.array(exp_gan_orig)
        #print('exp_gan.shape: ', exp_gan.shape)

        '''
        pd.DataFrame(exp_lime[:,:,0]).to_csv("exp_lime_R.csv")
        pd.DataFrame(exp_lime[:,:,1]).to_csv("exp_lime_G.csv")
        pd.DataFrame(exp_lime[:,:,2]).to_csv("exp_lime_B.csv")
        '''

        '''
        pd.DataFrame(exp_gan[:,:,0]).to_csv("exp_gan_R.csv")
        pd.DataFrame(exp_gan[:,:,1]).to_csv("exp_gan_G.csv")
        pd.DataFrame(exp_gan[:,:,2]).to_csv("exp_gan_B.csv")
        '''

        #seg_unique = np.unique(segments)

        # weighted eval
        color_coverage_percent = []

        weights = []

        avg_R_list = []
        avg_G_list = []
        avg_B_list = []

        diff_R_list = []
        diff_G_list = []
        diff_B_list = []

        diff_list = []
        
        avg_diff_list = []
        
        for e in exp_nav.exp:
            if abs(e[1]) >= 0.0:
                
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0
                
                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0

                weights.append(abs(e[1]))

                for row in range(0, segments.shape[0]):
                    for columns in range(0, segments.shape[1]):
                        if segments[row, columns] == e[0]:
                            '''
                            print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                            print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                            print('\n')
                            '''

                            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                            if lime_color_name == gan_color_name:
                                same_color_count += 1

                            color_count += 1
                            count_R += 1
                            count_G += 1
                            count_B += 1
                            count_avg += 1

                            if int(exp_lime[row, columns, 0]) != 0:
                                diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
                            else:
                                count_R -= 1     

                            if int(exp_lime[row, columns, 1]) != 0:
                                diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
                            else:
                                count_G -= 1    
                            
                            if int(exp_lime[row, columns, 2]) != 0:
                                diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
                            else:
                                count_B -= 1    

                            #avg_avg += (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3

                            temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                            if temp_avg_sum != 0:
                                avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                                - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
                            else:
                                count_avg -= 1   

                if count_R == 0:
                    count_R = 1

                if count_G == 0:
                    count_G = 1

                if count_B == 0:
                    count_B = 1

                if count_avg == 0:
                    count_avg = 1

                if color_count == 0:
                    color_count = 1    

                color_coverage_percent.append(100 * same_color_count / color_count)
                
                diff_R /= count_R
                diff_G /= count_G
                diff_B /= count_B
                
                diff_R_list.append(diff_R)
                diff_G_list.append(diff_G)
                diff_B_list.append(diff_B)

                diff = (diff_R + diff_G + diff_B) / 3
                diff_list.append(diff)

                avg_diff /= count_avg
                avg_diff_list.append(avg_diff)

        weights_sum = sum(weights)

        if weights_sum == 0.0:
            weights = [1.0] * len(weights)
            weights_sum = sum(weights)
        
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_coverage_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_coverage_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_R_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_R_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",") 


        avg_similarity_percentage = []
        for i in range(0, len(diff_G_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_G_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_B_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_B_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")           


        avg_similarity_percentage = []
        for i in range(0, len(diff_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_avg_similarity_percentage = []
        for i in range(0, len(avg_diff_list)):
            avg_avg_similarity_percentage.append(100 * (1.0 - avg_diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("percentages.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + "\n")



        # LOCAL PLAN eval 
        count_R = 0
        count_G = 0
        count_B = 0
        count_avg = 0

        same_color_count = 0
        color_count = 0
        
        avg_R = 0
        avg_G = 0
        avg_B = 0

        diff_R = 0
        diff_G = 0
        diff_B = 0

        avg_avg = 0
        avg_diff = 0

        for i in range(0, len(local_plan_x_list)):
            row = local_plan_y_list[i]
            columns = local_plan_x_list[i]

            '''
            print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
            print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
            print('\n')
            '''

            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
            if lime_color_name == gan_color_name:
                same_color_count += 1

            color_count += 1
            count_R += 1
            count_G += 1
            count_B += 1
            count_avg += 1

            if int(exp_lime[row, columns, 0]) != 0:
                diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
            else:
                count_R -= 1     

            if int(exp_lime[row, columns, 1]) != 0:
                diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
            else:
                count_G -= 1    
            
            if int(exp_lime[row, columns, 2]) != 0:
                diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
            else:
                count_B -= 1    

            temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
            if temp_avg_sum != 0:
                avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
            else:
                count_avg -= 1

        if count_R == 0:
            count_R = 1

        if count_G == 0:
            count_G = 1

        if count_B == 0:
            count_B = 1

        if count_avg == 0:
            count_avg = 1

        if color_count == 0:
            color_count = 1    

        color_coverage_percent = 100 * same_color_count / color_count
        
        diff_R /= count_R
        diff_G /= count_G
        diff_B /= count_B
        
        diff_R_list.append(diff_R)
        diff_G_list.append(diff_G)
        diff_B_list.append(diff_B)

        diff = (diff_R + diff_G + diff_B) / 3

        avg_diff /= count_avg
        with open("local_plan.csv", "a") as myfile:
            myfile.write(str(color_coverage_percent) + "," + str(100 * (1.0 - diff)) + "," + str(100 * (1.0 - avg_diff)) + "\n")


        # GLOBAL PLAN eval 
        count_R = 0
        count_G = 0
        count_B = 0
        count_avg = 0

        same_color_count = 0
        color_count = 0
        
        avg_R = 0
        avg_G = 0
        avg_B = 0

        diff_R = 0
        diff_G = 0
        diff_B = 0

        avg_avg = 0
        avg_diff = 0

        for i in range(0, len(plan_x_list)):
            row = plan_y_list[i]
            columns = plan_x_list[i]
            
            '''
            print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
            print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
            print('\n')
            '''

            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
            if lime_color_name == gan_color_name:
                same_color_count += 1

            color_count += 1
            count_R += 1
            count_G += 1
            count_B += 1
            count_avg += 1
        
            if int(exp_lime[row, columns, 0]) != 0:
                diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
            else:
                count_R -= 1     

            if int(exp_lime[row, columns, 1]) != 0:
                diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
            else:
                count_G -= 1    
            
            if int(exp_lime[row, columns, 2]) != 0:
                diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
            else:
                count_B -= 1    

            temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
            if temp_avg_sum != 0:
                avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
            else:
                count_avg -= 1

        if count_R == 0:
            count_R = 1

        if count_G == 0:
            count_G = 1

        if count_B == 0:
            count_B = 1

        if count_avg == 0:
            count_avg = 1

        if color_count == 0:
            color_count = 1    

        color_coverage_percent = 100 * same_color_count / color_count
        
        diff_R /= count_R
        diff_G /= count_G
        diff_B /= count_B
        
        diff_R_list.append(diff_R)
        diff_G_list.append(diff_G)
        diff_B_list.append(diff_B)

        diff = (diff_R + diff_G + diff_B) / 3

        avg_diff /= count_avg
        with open("global_plan.csv", "a") as myfile:
            myfile.write(str(color_coverage_percent) + "," + str(100 * (1.0 - diff)) + "," + str(100 * (1.0 - avg_diff)) + "\n")


        # OBSTACLES eval 
        count_R = 0
        count_G = 0
        count_B = 0
        count_avg = 0

        same_color_count = 0
        color_count = 0

        color_change_count = 0
        color_flip_count = 0
        color_turn_count = 0
        color_other_count = 0
        
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
                    row = i
                    columns = j
                    
                    '''
                    print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                    print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                    print('\n')
                    '''

                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1
                    else:
                        # if positive
                        if lime_color_name == 'aquamarine':
                            color_change_count += 1
                            if gan_color_name == 'violet':
                                color_flip_count += 1
                            elif gan_color_name == 'white':
                                color_turn_count += 1
                            else:
                                color_other_count += 1    
                        # if negative        
                        elif lime_color_name == 'violet':
                            color_change_count += 1
                            if gan_color_name == 'aquamarine':
                                color_flip_count += 1
                            elif gan_color_name == 'white':
                                color_turn_count += 1
                            else:
                                color_other_count += 1
                        elif lime_color_name == 'white':
                            color_change_count += 1
                            if gan_color_name == 'aquamarine' or gan_color_name == 'violet':
                                color_turn_count += 1
                            else:
                                color_other_count += 1                                        

                    color_count += 1
                    count_R += 1
                    count_G += 1
                    count_B += 1
                    count_avg += 1

                    if int(exp_lime[row, columns, 0]) != 0:
                        diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
                    else:
                        count_R -= 1     

                    if int(exp_lime[row, columns, 1]) != 0:
                        diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
                    else:
                        count_G -= 1    
                    
                    if int(exp_lime[row, columns, 2]) != 0:
                        diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
                    else:
                        count_B -= 1    

                    temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                    if temp_avg_sum != 0:
                        avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                        - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
                    else:
                        count_avg -= 1

        if count_R == 0:
            count_R = 1

        if count_G == 0:
            count_G = 1

        if count_B == 0:
            count_B = 1

        if count_avg == 0:
            count_avg = 1

        if color_count == 0:
            color_count = 1    

        color_coverage_percent = 100 * same_color_count / color_count

        color_flip_percent = 100 * color_flip_count / color_count #color_change_count
        color_turn_percent = 100 * color_turn_count / color_count #color_change_count
        color_other_percent = 100 * color_other_count / color_count #color_change_count
        
        diff_R /= count_R
        diff_G /= count_G
        diff_B /= count_B
        
        diff_R_list.append(diff_R)
        diff_G_list.append(diff_G)
        diff_B_list.append(diff_B)

        diff = (diff_R + diff_G + diff_B) / 3

        avg_diff /= count_avg
        with open("obstacles.csv", "a") as myfile:
                                myfile.write(str(color_coverage_percent) + "," + str(100 * (1.0 - diff)) + "," + str(100 * (1.0 - avg_diff)) + ","  + str(color_flip_percent) + ","  + str(color_turn_percent) + "," + str(color_other_percent) + "\n")


        # FREE SPACE eval 
        count_R = 0
        count_G = 0
        count_B = 0
        count_avg = 0

        same_color_count = 0
        color_count = 0

        color_change_count = 0
        color_flip_count = 0
        color_turn_count = 0
        color_other_count = 0
        
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
                    row = i
                    columns = j
                    
                    '''
                    print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                    print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                    print('\n')
                    '''

                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1
                    else:
                        # if positive
                        if lime_color_name == 'lightgreen':
                            color_change_count += 1
                            if gan_color_name == 'salmon':
                                color_flip_count += 1
                            elif gan_color_name == 'gray':
                                color_turn_count += 1
                            else:
                                color_other_count += 1    
                        # if negative        
                        elif lime_color_name == 'salmon':
                            color_change_count += 1
                            if gan_color_name == 'lightgreen':
                                color_flip_count += 1
                            elif gan_color_name == 'gray':
                                color_turn_count += 1
                            else:
                                color_other_count += 1
                        elif lime_color_name == 'gray':
                            color_change_count += 1
                            if gan_color_name == 'lightgreen' or gan_color_name == 'salmon':
                                color_turn_count += 1
                            else:
                                color_other_count += 1    

                    color_count += 1
                    count_R += 1
                    count_G += 1
                    count_B += 1
                    count_avg += 1

                    if int(exp_lime[row, columns, 0]) != 0:
                        diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
                    else:
                        count_R -= 1     

                    if int(exp_lime[row, columns, 1]) != 0:
                        diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
                    else:
                        count_G -= 1    
                    
                    if int(exp_lime[row, columns, 2]) != 0:
                        diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
                    else:
                        count_B -= 1    

                    temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                    if temp_avg_sum != 0:
                        avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                        - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
                    else:
                        count_avg -= 1

        if count_R == 0:
            count_R = 1

        if count_G == 0:
            count_G = 1

        if count_B == 0:
            count_B = 1

        if count_avg == 0:
            count_avg = 1

        if color_count == 0:
            color_count = 1    

        color_coverage_percent = 100 * same_color_count / color_count
        color_flip_percent = 100 * color_flip_count / color_count #color_change_count
        color_turn_percent = 100 * color_turn_count / color_count #color_change_count
        color_other_percent = 100 * color_other_count / color_count #color_change_count
        
        diff_R /= count_R
        diff_G /= count_G
        diff_B /= count_B
        
        diff_R_list.append(diff_R)
        diff_G_list.append(diff_G)
        diff_B_list.append(diff_B)

        diff = (diff_R + diff_G + diff_B) / 3

        avg_diff /= count_avg
        
        with open("free_space.csv", "a") as myfile:
            myfile.write(str(color_coverage_percent) + "," + str(100 * (1.0 - diff)) + "," + str(100 * (1.0 - avg_diff)) + ","  + str(color_flip_percent) + ","  + str(color_turn_percent) + "," + str(color_other_percent) + "\n")



        # obstacles weighted eval
        color_coverage_percent = []
        color_flip_percent = []
        color_turn_percent = []
        color_other_percent = []

        weights = []

        avg_R_list = []
        avg_G_list = []
        avg_B_list = []

        diff_R_list = []
        diff_G_list = []
        diff_B_list = []

        diff_list = []
        
        avg_diff_list = []
        
        for e in exp_nav.exp:
            if abs(e[1]) >= 0.0:
                
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                color_change_count = 0
                color_flip_count = 0
                color_turn_count = 0
                color_other_count = 0
                
                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0

                weights.append(abs(e[1]))

                obstacle = False

                for row in range(0, segments.shape[0]):
                    for columns in range(0, segments.shape[1]):
                        if segments[row, columns] == e[0]:
                            if image_flipped[row, columns] == 99:
                                obstacle = True
                                '''
                                print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                                print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                                print('\n')
                                '''

                                lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                                gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                                if lime_color_name == gan_color_name:
                                    same_color_count += 1
                                else:
                                    # if positive
                                    if lime_color_name == 'aquamarine':
                                        color_change_count += 1
                                        if gan_color_name == 'violet':
                                            color_flip_count += 1
                                        elif gan_color_name == 'white':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1    
                                    # if negative        
                                    elif lime_color_name == 'violet':
                                        color_change_count += 1
                                        if gan_color_name == 'aquamarine':
                                            color_flip_count += 1
                                        elif gan_color_name == 'white':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1
                                    elif lime_color_name == 'white':
                                        color_change_count += 1
                                        if gan_color_name == 'aquamarine' or gan_color_name == 'violet':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1                                        

                                color_count += 1
                                count_R += 1
                                count_G += 1
                                count_B += 1
                                count_avg += 1
            
                                if int(exp_lime[row, columns, 0]) != 0:
                                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
                                else:
                                    count_R -= 1     

                                if int(exp_lime[row, columns, 1]) != 0:
                                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
                                else:
                                    count_G -= 1    
                                
                                if int(exp_lime[row, columns, 2]) != 0:
                                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
                                else:
                                    count_B -= 1    

                                temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                                if temp_avg_sum != 0:
                                    avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
                                else:
                                    count_avg -= 1   

                if obstacle == True:
                    if count_R == 0:
                        count_R = 1

                    if count_G == 0:
                        count_G = 1

                    if count_B == 0:
                        count_B = 1

                    if count_avg == 0:
                        count_avg = 1

                    if color_count == 0:
                        color_count = 1    

                    color_coverage_percent.append(100 * same_color_count / color_count)
                    color_flip_percent.append(100 * color_flip_count / color_count)
                    color_turn_percent.append(100 * color_turn_count / color_count)
                    color_other_percent.append(100 * color_other_count / color_count)
                    
                    diff_R /= count_R
                    diff_G /= count_G
                    diff_B /= count_B
                    
                    diff_R_list.append(diff_R)
                    diff_G_list.append(diff_G)
                    diff_B_list.append(diff_B)

                    diff = (diff_R + diff_G + diff_B) / 3
                    diff_list.append(diff)

                    avg_diff /= count_avg
                    avg_diff_list.append(avg_diff)

                else:
                    color_coverage_percent.append(0.0)
                    color_flip_percent.append(0.0)
                    color_turn_percent.append(0.0)
                    color_other_percent.append(0.0)
                    diff_R_list.append(1.0)
                    diff_G_list.append(1.0)
                    diff_B_list.append(1.0)
                    diff_list.append(1.0)
                    avg_diff_list.append(1.0)    

        weights_sum = sum(weights)

        if weights_sum == 0.0:
            weights = [1.0] * len(weights)
            weights_sum = sum(weights)
        
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_coverage_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_coverage_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_R_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_R_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",") 


        avg_similarity_percentage = []
        for i in range(0, len(diff_G_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_G_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_B_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_B_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")           


        avg_similarity_percentage = []
        for i in range(0, len(diff_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_avg_similarity_percentage = []
        for i in range(0, len(avg_diff_list)):
            avg_avg_similarity_percentage.append(100 * (1.0 - avg_diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_flip_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_flip_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")

        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_turn_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_turn_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")

        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_other_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_other_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("obstacles_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + "\n")


        # free space weighted eval
        color_coverage_percent = []
        color_flip_percent = []
        color_turn_percent = []
        color_other_percent = []

        weights = []

        avg_R_list = []
        avg_G_list = []
        avg_B_list = []

        diff_R_list = []
        diff_G_list = []
        diff_B_list = []

        diff_list = []
        
        avg_diff_list = []
        
        for e in exp_nav.exp:
            if abs(e[1]) >= 0.0:
                
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                color_change_count = 0
                color_flip_count = 0
                color_turn_count = 0
                color_other_count = 0
                
                avg_R = 0
                avg_G = 0
                avg_B = 0

                diff_R = 0
                diff_G = 0
                diff_B = 0

                avg_avg = 0
                avg_diff = 0

                weights.append(abs(e[1]))

                free_space = False

                for row in range(0, segments.shape[0]):
                    for columns in range(0, segments.shape[1]):
                        if segments[row, columns] == e[0]:
                            if image_flipped[row, columns] == 0:
                                free_space = True
                                '''
                                print('lime_color_name: ', convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2])))
                                print('gan_color_name: ', convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2])))
                                print('\n')
                                '''

                                lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                                gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                                if lime_color_name == gan_color_name:
                                    same_color_count += 1
                                else:
                                    # if positive
                                    if lime_color_name == 'lightgreen':
                                        color_change_count += 1
                                        if gan_color_name == 'salmon':
                                            color_flip_count += 1
                                        elif gan_color_name == 'gray':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1    
                                    # if negative        
                                    elif lime_color_name == 'salmon':
                                        color_change_count += 1
                                        if gan_color_name == 'lightgreen':
                                            color_flip_count += 1
                                        elif gan_color_name == 'gray':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1
                                    elif lime_color_name == 'gray':
                                        color_change_count += 1
                                        if gan_color_name == 'lightgreen' or gan_color_name == 'salmon':
                                            color_turn_count += 1
                                        else:
                                            color_other_count += 1                                        

                                color_count += 1
                                count_R += 1
                                count_G += 1
                                count_B += 1
                                count_avg += 1
            
                                if int(exp_lime[row, columns, 0]) != 0:
                                    diff_R += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0])) / int(exp_lime[row, columns, 0])
                                else:
                                    count_R -= 1     

                                if int(exp_lime[row, columns, 1]) != 0:
                                    diff_G += abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1])) / int(exp_lime[row, columns, 1])
                                else:
                                    count_G -= 1    
                                
                                if int(exp_lime[row, columns, 2]) != 0:
                                    diff_B += abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2])) / int(exp_lime[row, columns, 2]) 
                                else:
                                    count_B -= 1    

                                temp_avg_sum = (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                                if temp_avg_sum != 0:
                                    avg_diff += abs( (int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3 
                                    - (int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3 ) / ( int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2]) ) / 3
                                else:
                                    count_avg -= 1   

                if free_space == True:
                    if count_R == 0:
                        count_R = 1

                    if count_G == 0:
                        count_G = 1

                    if count_B == 0:
                        count_B = 1

                    if count_avg == 0:
                        count_avg = 1

                    if color_count == 0:
                        color_count = 1    

                    color_coverage_percent.append(100 * same_color_count / color_count)
                    color_flip_percent.append(100 * color_flip_count / color_count)
                    color_turn_percent.append(100 * color_turn_count / color_count)
                    color_other_percent.append(100 * color_other_count / color_count)
                    
                    diff_R /= count_R
                    diff_G /= count_G
                    diff_B /= count_B
                    
                    diff_R_list.append(diff_R)
                    diff_G_list.append(diff_G)
                    diff_B_list.append(diff_B)

                    diff = (diff_R + diff_G + diff_B) / 3
                    diff_list.append(diff)

                    avg_diff /= count_avg
                    avg_diff_list.append(avg_diff)

                else:
                    color_coverage_percent.append(0.0)
                    color_flip_percent.append(0.0)
                    color_turn_percent.append(0.0)
                    color_other_percent.append(0.0)
                    diff_R_list.append(1.0)
                    diff_G_list.append(1.0)
                    diff_B_list.append(1.0)
                    diff_list.append(1.0)
                    avg_diff_list.append(1.0)    

        weights_sum = sum(weights)

        if weights_sum == 0.0:
            weights = [1.0] * len(weights)
            weights_sum = sum(weights)
        
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_coverage_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_coverage_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_R_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_R_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",") 


        avg_similarity_percentage = []
        for i in range(0, len(diff_G_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_G_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_similarity_percentage = []
        for i in range(0, len(diff_B_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_B_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")           


        avg_similarity_percentage = []
        for i in range(0, len(diff_list)):
            avg_similarity_percentage.append(100 * (1.0 - diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        avg_avg_similarity_percentage = []
        for i in range(0, len(avg_diff_list)):
            avg_avg_similarity_percentage.append(100 * (1.0 - avg_diff_list[i]))
        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += avg_avg_similarity_percentage[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(avg_avg_similarity_percentage[i] * weights[i] / weights_sum)

        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")


        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_flip_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_flip_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")

        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_turn_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_turn_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + ",")

        explanation_saved_percentage = 0.0
        explanation_saved_percentage_list = []
        weights_percentage = []
        for i in range(0, len(weights)):
            explanation_saved_percentage += color_other_percent[i] * weights[i] / weights_sum
            explanation_saved_percentage_list.append(color_other_percent[i] * weights[i] / weights_sum)
            weights_percentage.append(100 * weights[i] / weights_sum)
    
        with open("free_space_weighted.csv", "a") as myfile:
            myfile.write(str(explanation_saved_percentage) + "\n")



###### TKINTER START ######

from tkinter import *
# GUI architecture
root = Tk()
root.title("Navigation Explainer")

# Dropdown menu options
options_exp_alg = [
    "LIME",
    "Anchors"
]
# datatype of menu text
clicked_exp_alg = StringVar()  
# initial menu text
clicked_exp_alg.set( "Choose explanation algorithm" )  
# Create Dropdown menu
drop = OptionMenu( root , clicked_exp_alg , *options_exp_alg )
drop.pack()

options_exp_mode = [
    "image",
    "tabular",
    "tabular_costmap"
]
# datatype of menu text
clicked_exp_mode = StringVar()  
# initial menu text
clicked_exp_mode.set( "Choose LIME explanation method" )  
# Create Dropdown menu
drop = OptionMenu( root , clicked_exp_mode , *options_exp_mode )
drop.pack()

options_tab_mode = [
    "regression",
    "classification"
]
# datatype of menu text
clicked_tab_mode = StringVar()  
# initial menu text
clicked_tab_mode.set( "Choose LIME tabular explanation method" )  
# Create Dropdown menu
drop = OptionMenu( root , clicked_tab_mode , *options_tab_mode )
drop.pack()

def loadToGlobalVars():
    global explanation_alg, explanation_mode, tabular_mode
    explanation_alg = clicked_exp_alg.get()
    explanation_mode = clicked_exp_mode.get()
    tabular_mode = clicked_tab_mode.get()
    
    print('explanation algorithm: ', explanation_alg)
    print('explanation mode:', explanation_mode)
    print('tabular_mode: ', tabular_mode)

button_load = Button( root , text = "Confirm choices" , command = loadToGlobalVars ).pack()
  
buttonLimeSingle = Button(root, text='Run LIME single', height=3, width=25, command=LimeSingle, fg='black', bg='white')
#buttonLimeSingle.grid(row=0,column=0)
buttonLimeSingle.pack()

buttonCreateDataset = Button(root, text='Create dataset for LIME/GAN', height=3, width=25, command=CreateDataset, fg='black', bg='white')
#buttonCreateDataset.grid(row=1,column=0)
buttonCreateDataset.pack()

buttonEvaluateLIME = Button(root, text='Evaluate LIME', height=3, width=25, command=EvaluateLIME, fg='black', bg='white')
#buttonEvaluateLIME.grid(row=2,column=0)
buttonEvaluateLIME.pack()

buttonRunGAN = Button(root, text='Run GAN', height=3, width=25, command=RunGAN, fg='black', bg='white')
#buttonRunGAN.grid(row=3,column=0)
buttonRunGAN.pack()

buttonEvaluateLIMEvsGAN = Button(root, text='Evaluate LIME vs GAN', height=3, width=25, command=EvaluateLIMEvsGAN, fg='black', bg='white')
#buttonEvaluateLIMEvsGAN.grid(row=4,column=0)
buttonEvaluateLIMEvsGAN.pack()

root.mainloop()

###### TKINTER END ######


