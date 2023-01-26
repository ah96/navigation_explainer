#!/usr/bin/env python3

# Global variables
ds_id = 1
ds = 'ds' + str(ds_id)

print('dataset: ', ds)

# possible explanation algorithms: 'lime', 'anchors', 'shap'
explanation_alg = ''

# possible explanation modes: 'tabular', 'image', 'costmap'
explanation_mode = ''

# underlying model (explanation) modes: 'regression', 'classification'
underlying_model_mode = ''

import os
dirCurr = os.getcwd()

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
    

    # Calculating offset as the maximum of offsets of individual input files
    num_of_first_rows_to_delete = max(offsets)
    '''
    print('num_of_first_rows_to_delete: ', num_of_first_rows_to_delete)
    print('\n')
    '''


    # Delete entries with 'None' frame from local_costmap_info (cleaning the data)
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


    # Delete entries with 'None' frame from odom (cleaning the data)
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


    # Delete entries with 'None' frame from amcl_pose (cleaning the data)
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


    # Delete entries with 'None' frame from cmd_vel (cleaning the data)
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


    # Delete entries with 'None' frame from tf_odom_map (cleaning the data)
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


    # Delete entries with 'None' frame from tf_map_odom (cleaning the data)
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


def Single():
    import pandas as pd
    
    # create train and test dataframes
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()

    # some explanation variables
    output_class_name = ''
    num_samples = 100

    # Data loading
    from navigation_explainer import DataLoader
    
    # load input data
    odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data(ds)
    '''
    print("---input loaded---")
    print('\n')
    '''

    # load output data
    cmd_vel = DataLoader.load_output_data(ds)
    '''
    print("---output loaded---")
    print('\n')
    '''

    # preprocess data
    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints)
        
    if explanation_mode == 'tabular' or explanation_mode == 'tabular_costmap':
        from navigation_explainer import DatasetCreator
    
        # Select input for explanation algorithm
        X = odom.iloc[:, 6:8]  # input for explanation are odometry velocities
        # print(X)

        one_hot_encoding = True

        if underlying_model_mode == 'regression':
            import numpy as np

            # regression
            # Selecting the output for the explanation algorithm
            index_output_class = 1 # [0] - command linear velocity, [1] - command angular velocity
            y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
            #print(y)        
            output_class_name = y.columns.values[0]

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == False):
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

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == True):
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
    from navigation_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        underlying_model_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)
    
    print('\n\n\n\nEXPLANATION STARTS!!!')

    print('\nexpID range: ', (0, local_costmap_info.shape[0] - num_of_first_rows_to_delete))
    print('\nnum_of_first_rows_to_delete = ', num_of_first_rows_to_delete)    

    choose_random_instance = False

    if choose_random_instance == True:
        # random instance selection
        import random
        expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)
        print('\nexpID: ', expID)
    else:     
        # optional instance selection - deterministic
        expID = 88
        print('\nexpID: ', expID)

    import time
    start_total_exp_time = time.time()
    exp_nav.explain_instance(expID)
    end_total_exp_time = time.time()
    print('\nTotal explanation time: ', end_total_exp_time - start_total_exp_time)
    print('\nEND of EXPLANATION!!!')

def Evaluate():
    import pandas as pd
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()

    output_class_name = ''
    num_samples = 100

    # Data loading
    from navigation_explainer import DataLoader
    
    # load input data
    odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data(ds)
    '''
    print("---input loaded---")
    print('\n')
    '''

    # load output data
    cmd_vel = DataLoader.load_output_data(ds)
    '''
    print("---output loaded---")
    print('\n')
    '''

    # preprocess data
    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints)
        
    if explanation_mode == 'tabular' or explanation_mode == 'tabular_costmap':
        from navigation_explainer import DatasetCreator
    
        # Select input for explanation algorithm
        X = odom.iloc[:, 6:8]  # input for explanation are odometry velocities
        # print(X)

        one_hot_encoding = True

        if underlying_model_mode == 'regression':
            import numpy as np

            # regression
            # Selecting the output for the explanation algorithm
            index_output_class = 1 # [0] - command linear velocity, [1] - command angular velocity
            y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
            #print(y)        
            output_class_name = y.columns.values[0]

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == False):
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

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == True):
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
    from navigation_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        underlying_model_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)

    import time
    evaluation_sample_size = 50
    
    dirName = 'evaluation_results'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

    with open(dirCurr + '/' + dirName + "/explanations.txt", "a") as myfile:
        myfile.write('explain_instance_time\n')

    choose_random_instance = True
    for i in range(0, evaluation_sample_size):
        print('\ni = ', i)

        if choose_random_instance == True:
            # random instance selection
            print('\nexpID range: ', (0, local_costmap_info.shape[0] - num_of_first_rows_to_delete))
            import random
            expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)
            print('\nexpID: ', expID)
        else:     
            # optional instance selection - deterministic
            expID = 75 #DS1: #51 #78 #84 #144, #DS2: #260
            print('\nexpID: ', expID)

        with open(dirCurr + '/' + dirName + '/IDs.csv', "a") as myfile:
            myfile.write(str(expID) + '\n')
        myfile.close() 

        total_exp_start = time.time()
        exp_nav.explain_instance_evaluation(expID, i)
        total_exp_end = time.time()
        with open(dirCurr + '/' + dirName + "/explanations.txt", "a") as myfile:
            myfile.write(str(round(total_exp_end - total_exp_start, 2)) + '\n')

def CreateDataset():
    import pandas as pd
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()

    output_class_name = ''
    num_samples = 100

    # Data loading
    from navigation_explainer import DataLoader
    
    # load input data
    odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data(ds)
    '''
    print("---input loaded---")
    print('\n')
    '''

    # load output data
    cmd_vel = DataLoader.load_output_data(ds)
    '''
    print("---output loaded---")
    print('\n')
    '''

    # preprocess data
    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom, plan, teb_global_plan, teb_local_plan, footprints)
        
    if explanation_mode == 'tabular' or explanation_mode == 'tabular_costmap':
        from navigation_explainer import DatasetCreator
    
        # Select input for explanation algorithm
        X = odom.iloc[:, 6:8]  # input for explanation are odometry velocities
        # print(X)

        one_hot_encoding = True

        if underlying_model_mode == 'regression':
            import numpy as np

            # regression
            # Selecting the output for the explanation algorithm
            index_output_class = 1 # [0] - command linear velocity, [1] - command angular velocity
            y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
            #print(y)        
            output_class_name = y.columns.values[0]

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == False):
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

        elif (underlying_model_mode == 'classification') & (one_hot_encoding == True):
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
    from navigation_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        underlying_model_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)

    dirName = 'dataset_creation_results'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass    

    #'''
    with open(dirCurr + '/' + dirName + '/costmap_data.csv', "a") as myfile:
            myfile.write('picture_ID,width,height,origin_x,origin_y,resolution\n')

    with open(dirCurr + '/' + dirName + '/local_plan_coordinates.csv', "a") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open(dirCurr + '/' + dirName + '/global_plan_coordinates.csv', "a") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open(dirCurr + '/' + dirName + '/robot_coordinates.csv', "a") as myfile:
        myfile.write('picture_ID,position_x,position_y\n')
    #''' 

    dataset_size = local_costmap_info.shape[0] - num_of_first_rows_to_delete
    print('local_costmap_info.shape[0]: ', local_costmap_info.shape[0])
    print('num_of_first_rows_to_delete: ', num_of_first_rows_to_delete)
    print('dataset_size: ', dataset_size)

    choose_random_instance = True
    for i in range(174, dataset_size, 1):
        if choose_random_instance == False:
            # optional instance selection - deterministic
            expID = i
            print('expID: ', expID)

        else:
            # random instance selection
            import random    
            expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete) 

        exp_nav.explain_instance_dataset(expID, i)

def RunGAN():
    # Data loading
    from navigation_explainer import DataLoader
    
    # load input data
    odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data(ds)
    '''
    print("---input loaded---")
    print('\n')
    '''

    # load output data
    cmd_vel = DataLoader.load_output_data(ds)
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
    y_train = []
    y_test = []
    num_samples = 0

    # output_class_name - not important for LIME image
    #output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

    # Explanation
    from navigation_explainer import ExplainNavigation

    '''
    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        tabular_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)
    '''

    # optional instance selection - deterministic
    #expID = 28

    # rando1m instance selection
    import random
    expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)

    print('expID: ', expID)
    print('\n')

    index = expID
    offset = num_of_first_rows_to_delete

    import time
    before_gan_predict_start = time.time()

    # Get local costmap
    # Original costmap will be saved to self.local_costmap_original
    local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
    
    # Make image a np.array deepcopy of local_costmap_original
    import numpy as np
    import copy
    image = np.array(copy.deepcopy(local_costmap_original))

    # '''
    # Turn inflated area to free space and 100s to 99s
    image[image == 100] = 99
    image[image != 99] = 0
    # '''

    # Turn every local costmap entry from int to float, so the segmentation algorithm works okay - here probably not needed
    image = image * 1.0

    free_space_shade = 180
    obstacle_shade = 255
    from skimage.color import gray2rgb
    image = gray2rgb(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = free_space_shade
            elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = obstacle_shade

    # plot costmap with plans
    import matplotlib.pyplot as plt
    fig = plt.figure(frameon=False)
    w = 1.6
    h = 1.6
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image.astype(np.uint8), aspect='auto')

    import pandas as pd

    costmap_info_tmp = local_costmap_info.iloc[index, :]
    costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
    costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

    # save costmap info to class variables
    localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
    localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
    localCostmapResolution = costmap_info_tmp.iloc[0, 0]
    #localCostmapHeight = costmap_info_tmp.iloc[0, 2]
    #localCostmapWidth = costmap_info_tmp.iloc[0, 1]

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
        if 0 <= x_temp < costmap_size and 0 <= y_temp < costmap_size:
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
        if 0 <= x_temp < costmap_size and 0 <= y_temp < costmap_size:
            plan_x_list.append(x_temp)
            plan_y_list.append(y_temp)
    
    ax.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
    ax.scatter(local_plan_x_list, local_plan_y_list, c='yellow', marker='o')
    # plot robots' location, orientation and local plan
    ax.scatter(x_odom_index, y_odom_index, c='white', marker='o')
    # za novu verziju GAN-a isključiti crtanje orijentacije
    #ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')
    # '''
    
    dirName = 'GAN_results'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

    fig.savefig(dirCurr + '/' + dirName + '/input.png', transparent=False)
    fig.clf()

    before_gan_predict_end = time.time()
    before_gan_predict_time = before_gan_predict_end - before_gan_predict_start
    print('\nbefore_gan_predict_time = ', before_gan_predict_time)
 
    #import time
    gan_predict_start = time.time()
    from GAN import gan            
    gan.predict()
    gan_predict_end = time.time()
    gan_predict_time = gan_predict_end - gan_predict_start
    print('\ngan_predict_time = ', gan_predict_time)

    print('\nEND!!!')

def EvaluateLIMEvsGAN():
    ds1_test = [3,18,36,49,60,105,132,156,174]
    ds2_test = [0,12,111]
    ds10_test = [375,420,425,555,605]
    dss_test = [ds1_test,ds2_test,ds10_test]
    dss_names = ['ds1','ds2','ds10']

    '''
    # test indices for IROS original
    ds8_test = [170,185,190,205,215,220,225,250,275,280,285]
    ds9_test = [10,25,45,110,145]
    ds10_test = [5,355,375,420,425,555,605,620]
    dss_test = [ds8_test,ds9_test,ds10_test]
    #
    '''

    ''' 
    # test indices for RAAD   
    ds1_test = [1, 6, 12, 17, 20, 35, 44, 52, 58, 64, 66, 71] * 3
    ds2_test_ = [81, 86, 91, 97, 100, 103, 116, 118, 133, 136, 139, 150, 151, 163, 166, 169, 180, 183, 201, 204, 209, 214, 216, 218, 221, 237, 246, 249, 256, 259, 260]
    ds2_test = [(i-81)*3 for i in ds2_test_]
    ds3_test_ = [272, 282, 286, 288, 289, 293, 295, 297, 302, 304, 307, 309, 317, 330, 333, 337, 338, 339]
    ds3_test = [(i-264)*3 for i in ds3_test_]
    ds4_test_ = [351, 360, 365, 372, 382, 387, 399, 401, 405, 411, 415, 421, 426, 436, 446, 453, 458, 461, 465, 474, 475, 485, 496, 507, 519, 523, 530, 541, 546, 547, 548, 553, 555, 578, 582, 588, 590, 593, 595]
    ds4_test = [(i-342)*3 for i in ds4_test_]
    dss_test = [ds1_test,ds2_test,ds3_test,ds4_test]
    '''

    new_eval = True

    redni_broj_slike = 0
        
    for ID in range(1, len(dss_test)+1): 
        ds_id = ID
        ds = dss_names[ID-1] #'ds' + str(ds_id + 7)
        print('\nds = ', ds)

        # Data loading
        from navigation_explainer import DataLoader
        
        # load input data
        odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data(ds)
        '''
        print("---input loaded---")
        print('\n')
        '''

        # load output data
        cmd_vel = DataLoader.load_output_data(ds)
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
        y_train = []
        y_test = []
        num_samples = 0

        # output_class_name - not important for LIME image
        output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - 'cmd_vel_ang_z'

        # Explanation
        from navigation_explainer import ExplainNavigation

        exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                            current_goal, local_costmap_data, local_costmap_info,
                                                            amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                            underlying_model_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                            X_train, X_test, y_train, y_test, num_samples, plot=True)


        dirName = 'GANvsLIME_results'
        import os
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass
        
        if new_eval == True:
            if ID == 1:
                with open(dirCurr + '/' + dirName + "/times.csv", "a") as myfile:
                        myfile.write("lime,gan\n")

                with open(dirCurr + '/' + dirName + "/measures_whole_explanation.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")

                with open(dirCurr + '/' + dirName + "/measures_robot_position.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")

                with open(dirCurr + '/' + dirName + "/measures_local_plan.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")

                with open(dirCurr + '/' + dirName + "/measures_global_plan.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")

                with open(dirCurr + '/' + dirName + "/measures_obstacles.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")

                with open(dirCurr + '/' + dirName + "/measures_free_space.csv", "a") as myfile:
                        myfile.write("cie76,cie94,ciede2000,cmc,euc,euc_norm\n")
            
            num_iter = 1
            
            lime_time_avg = 0
            gan_time_avg = 0
                
            for num in range(0, len(dss_test[ID-1])):
                print('iteration: ', num)
                print('ds: ', ds_id)

                lime_time_avg = 0
                gan_time_avg = 0
                
                # optional instance selection - deterministic
                expID = dss_test[ID-1][num]

                print('expID: ', expID)


                # call LIME
                import numpy as np
                import pandas as pd
                import copy  
                import time  
                time_before_lime = time.time()
                exp_nav.explain_instance(expID)
                time_after_lime = time.time()
                lime_time_avg += time_after_lime - time_before_lime
                #print('LIME exp time: ', time_after - time_before)           


                # call GAN
                # Prepare data for GAN
                time_before_gan = time.time()
                index = expID
                offset = num_of_first_rows_to_delete

                # Get local costmap
                local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
                
                # Make image a np.array deepcopy of local_costmap_original
                image = np.array(copy.deepcopy(local_costmap_original))

                # '''
                # Turn inflated area to free space and 100s to 99s
                image[image == 100] = 99
                image[image != 99] = 0
                # '''

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay - here probably not needed
                image = image * 1.0
                
                #'''
                # get costmap info
                costmap_info_tmp = local_costmap_info.iloc[index, :]
                costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
                costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

                # save costmap info to class variables
                localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
                localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
                localCostmapResolution = costmap_info_tmp.iloc[0, 0]
                #localCostmapHeight = costmap_info_tmp.iloc[0, 2]
                #localCostmapWidth = costmap_info_tmp.iloc[0, 1]

                # get odometry info
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

                # get local plan
                local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
                local_plan_tmp = local_plan_tmp.iloc[:, 1:]
                # indices of local plan's poses in local costmap
                local_plan_x_list = []
                local_plan_y_list = []
                for i in range(1, local_plan_tmp.shape[0]):
                    x_temp = int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)
                    if 0 <= x_temp < costmap_size and 0 <= y_temp < costmap_size:
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
                plan_x_list = []
                plan_y_list = []
                for i in range(0, plan_tmp_tmp.shape[0], 3):
                    x_temp = int((plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)    
                    y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        #print('x_temp: ', x_temp)
                        #print('y_temp: ', y_temp)
                        #print('\n')
                        plan_x_list.append(x_temp)
                        plan_y_list.append(y_temp)

                #'''

                from GAN import gan            
                gan.predict()

                time_after_gan = time.time()
                gan_time_avg += time_after_gan - time_before_gan

                print('LIME time: ', lime_time_avg / num_iter)
                print('\n')
                print('GAN time: ', gan_time_avg / num_iter)
                print('\n')

                with open(dirCurr + '/' + dirName + "/times.csv", "a") as myfile:
                    myfile.write(str(lime_time_avg) + "," + str(gan_time_avg) + "\n")

                # evaluation
                import PIL.Image
                import os
                path1 = os.getcwd() + '/explanation_results' + '/explanation.png'
                exp_lime_orig = PIL.Image.open(path1).convert('RGB')
                path1 = os.getcwd() + '/' + 'GAN_results' + '/GAN.png'
                exp_gan_orig = PIL.Image.open(path1).convert('RGB')

                exp_lime = np.array(exp_lime_orig)
                print('exp_lime.shape: ', exp_lime.shape)
                exp_gan = np.array(exp_gan_orig)
                print('exp_gan.shape: ', exp_gan.shape)

                from PIL import Image

                im = Image.fromarray(exp_lime)
                im.save(os.getcwd() + '/' + dirName + "/lime_" + str(redni_broj_slike) + ".png")

                im = Image.fromarray(exp_gan)
                im.save(os.getcwd() + '/' + dirName + "/gan_" + str(redni_broj_slike) + ".png")

                redni_broj_slike += 1
                
                from skimage import color

                #cie76,cie94,ciede2000,cmc,euc
                whole_explanation_cie76 = 0.0
                whole_explanation_cie94 = 0.0
                whole_explanation_ciede2000 = 0.0
                whole_explanation_cmc = 0.0
                whole_explanation_euc = 0.0
                whole_explanation_euc_norm = 0.0
                whole_explanation_counter = 0

                robot_position_cie76 = 0.0
                robot_position_cie94 = 0.0
                robot_position_ciede2000 = 0.0
                robot_position_cmc = 0.0
                robot_position_euc = 0.0
                robot_position_euc_norm = 0.0
                robot_position_counter = 0

                local_plan_cie76 = 0.0
                local_plan_cie94 = 0.0
                local_plan_ciede2000 = 0.0
                local_plan_cmc = 0.0
                local_plan_euc = 0.0
                local_plan_euc_norm = 0.0
                local_plan_counter = 0

                global_plan_cie76 = 0.0
                global_plan_cie94 = 0.0
                global_plan_ciede2000 = 0.0
                global_plan_cmc = 0.0
                global_plan_euc = 0.0
                global_plan_euc_norm = 0.0
                global_plan_counter = 0

                obstacles_cie76 = 0.0
                obstacles_cie94 = 0.0
                obstacles_ciede2000 = 0.0
                obstacles_cmc = 0.0
                obstacles_euc = 0.0
                obstacles_euc_norm = 0.0
                obstacles_counter = 0

                free_space_cie76 = 0.0
                free_space_cie94 = 0.0
                free_space_ciede2000 = 0.0
                free_space_cmc = 0.0
                free_space_euc = 0.0
                free_space_euc_norm = 0.0
                free_space_counter = 0
                
                for i in range(0, 160):
                    for j in range(0, 160):
                        lime_color = color.rgb2lab(exp_lime[i, j, :])
                        #print('\nlime_color = ', lime_color)
                        gan_color = color.rgb2lab(exp_gan[i, j, :])
                        #print('gan_color = ', gan_color)

                        from colormath.color_objects import LabColor
                        lime_color = LabColor(lab_l= lime_color[0], lab_a = lime_color[1], lab_b = lime_color[2])
                        #print('\nlime_color = ', lime_color)
                        gan_color = LabColor(lab_l= gan_color[0], lab_a = gan_color[1], lab_b = gan_color[2])
                        #print('gan_color = ', gan_color)
            
                        from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000, delta_e_cmc
                        cie76 = delta_e_cie1976(gan_color, lime_color)
                        cie94 = delta_e_cie1994(gan_color, lime_color)
                        ciede2000 = delta_e_cie2000(gan_color, lime_color)
                        cmc = delta_e_cmc(gan_color, lime_color)
                        euc = math.sqrt((exp_gan[i, j, 0] - exp_lime[i, j, 0])**2 + (exp_gan[i, j, 1] - exp_lime[i, j, 1])**2 + (exp_gan[i, j, 2] - exp_lime[i, j, 2])**2)               
                        euc_norm = euc / math.sqrt(3*(255**2))
 
                        whole_explanation_cie76 += cie76
                        whole_explanation_cie94 += cie94
                        whole_explanation_ciede2000 += ciede2000
                        whole_explanation_cmc += cmc
                        whole_explanation_euc += euc
                        whole_explanation_euc_norm += euc_norm
                        whole_explanation_counter += 1

                        if i == y_odom_index[0] and j == x_odom_index[0]:
                            robot_position_cie76 += cie76
                            robot_position_cie94 += cie94
                            robot_position_ciede2000 += ciede2000
                            robot_position_cmc += cmc
                            robot_position_euc += euc
                            robot_position_euc_norm += euc_norm
                            robot_position_counter += 1

                        elif i in local_plan_y_list and j in local_plan_x_list:
                            local_plan_cie76 += cie76
                            local_plan_cie94 += cie94
                            local_plan_ciede2000 += ciede2000
                            local_plan_cmc += cmc
                            local_plan_euc += euc
                            local_plan_euc_norm += euc_norm
                            local_plan_counter += 1

                        elif i in exp_nav.transformed_plan_ys and j in exp_nav.transformed_plan_xs:
                            global_plan_cie76 += cie76
                            global_plan_cie94 += cie94
                            global_plan_ciede2000 += ciede2000
                            global_plan_cmc += cmc
                            global_plan_euc += euc
                            global_plan_euc_norm += euc_norm
                            global_plan_counter += 1

                        elif image[i, j] == 99:
                            obstacles_cie76 += cie76
                            obstacles_cie94 += cie94
                            obstacles_ciede2000 += ciede2000
                            obstacles_cmc += cmc
                            obstacles_euc += euc
                            obstacles_euc_norm += euc_norm
                            obstacles_counter += 1

                        elif image[i, j] == 0:
                            free_space_cie76 += cie76
                            free_space_cie94 += cie94
                            free_space_ciede2000 += ciede2000
                            free_space_cmc += cmc
                            free_space_euc += euc
                            free_space_euc_norm += euc_norm
                            free_space_counter += 1    

                whole_explanation_cie76 /= whole_explanation_counter
                whole_explanation_cie94 /= whole_explanation_counter
                whole_explanation_ciede2000 /= whole_explanation_counter
                whole_explanation_cmc /= whole_explanation_counter
                whole_explanation_euc /= whole_explanation_counter
                whole_explanation_euc_norm /= whole_explanation_counter

                robot_position_cie76 /= robot_position_counter
                robot_position_cie94 /= robot_position_counter
                robot_position_ciede2000 /= robot_position_counter
                robot_position_cmc /= robot_position_counter
                robot_position_euc /= robot_position_counter
                robot_position_euc_norm /= robot_position_counter

                local_plan_cie76 /= local_plan_counter
                local_plan_cie94 /= local_plan_counter
                local_plan_ciede2000 /= local_plan_counter
                local_plan_cmc /= local_plan_counter
                local_plan_euc /= local_plan_counter
                local_plan_euc_norm /= local_plan_counter

                global_plan_cie76 /= global_plan_counter
                global_plan_cie94 /= global_plan_counter
                global_plan_ciede2000 /= global_plan_counter
                global_plan_cmc /= global_plan_counter
                global_plan_euc /= global_plan_counter
                global_plan_euc_norm /= global_plan_counter

                obstacles_cie76 /= obstacles_counter
                obstacles_cie94 /= obstacles_counter
                obstacles_ciede2000 /= obstacles_counter
                obstacles_cmc /= obstacles_counter
                obstacles_euc /= obstacles_counter
                obstacles_euc_norm /= obstacles_counter

                free_space_cie76 /= free_space_counter
                free_space_cie94 /= free_space_counter
                free_space_ciede2000 /= free_space_counter
                free_space_cmc /= free_space_counter
                free_space_euc /= free_space_counter
                free_space_euc_norm /= free_space_counter

                with open(os.getcwd() + '/' + dirName + "/measures_whole_explanation.csv", "a") as myfile:
                    myfile.write(str(whole_explanation_cie76) + ',' + str(whole_explanation_cie94) + ',' + str(whole_explanation_ciede2000) 
                    + ',' + str(whole_explanation_cmc) + ',' + str(whole_explanation_euc) + ',' + str(whole_explanation_euc_norm) + ',' + "\n")

                with open(os.getcwd() + '/' + dirName + "/measures_robot_position.csv", "a") as myfile:
                    myfile.write(str(robot_position_cie76) + ',' + str(robot_position_cie94) + ',' + str(robot_position_ciede2000) 
                    + ',' + str(robot_position_cmc) + ',' + str(robot_position_euc) + ',' + str(robot_position_euc_norm) + ',' + "\n")

                with open(os.getcwd() + '/' + dirName + "/measures_local_plan.csv", "a") as myfile:
                    myfile.write(str(local_plan_cie76) + ',' + str(local_plan_cie94) + ',' + str(local_plan_ciede2000) 
                    + ',' + str(local_plan_cmc) + ',' + str(local_plan_euc) + ',' + str(local_plan_euc_norm) + ',' + "\n")                    

                with open(os.getcwd() + '/' + dirName + "/measures_global_plan.csv", "a") as myfile:
                    myfile.write(str(global_plan_cie76) + ',' + str(global_plan_cie94) + ',' + str(global_plan_ciede2000) 
                    + ',' + str(global_plan_cmc) + ',' + str(global_plan_euc) + ',' + str(global_plan_euc_norm) + ',' + "\n")

                with open(os.getcwd() + '/' + dirName + "/measures_obstacles.csv", "a") as myfile:
                    myfile.write(str(obstacles_cie76) + ',' + str(obstacles_cie94) + ',' + str(obstacles_ciede2000) 
                    + ',' + str(obstacles_cmc) + ',' + str(obstacles_euc) + ',' + str(obstacles_euc_norm) + ',' + "\n")

                with open(os.getcwd() + '/' + dirName + "/measures_free_space.csv", "a") as myfile:
                    myfile.write(str(free_space_cie76) + ',' + str(free_space_cie94) + ',' + str(free_space_ciede2000) 
                    + ',' + str(free_space_cmc) + ',' + str(free_space_euc) + ',' + str(free_space_euc_norm) + ',' + "\n")

        else:
            from test_color import create_dict_my, convert_rgb_to_names_my
            create_dict_my()

            import pandas as pd
            import time
            import numpy as np
            import copy
            import matplotlib.pyplot as plt

            if ID == 1:
                with open(os.getcwd() + '/' + dirName + "/times.csv", "a") as myfile:
                        myfile.write("lime,gan\n")

                with open(os.getcwd() + '/' + dirName + "/weights.csv", "a") as myfile:
                        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")        

                with open(os.getcwd() + '/' + dirName + "/segments.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/local_plan.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/global_plan.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/obstacles.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/free_space.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/robot_position.csv", "a") as myfile:
                        myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

                with open(os.getcwd() + '/' + dirName + "/gan_times.csv", "a") as myfile:
                        myfile.write("predict_time\n")

                with open(os.getcwd() + '/' + dirName + "/R_avg_lime.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/R_avg_gan.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/R_diff.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/G_avg_lime.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/G_avg_gan.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/G_diff.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/B_avg_lime.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/B_avg_gan.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

                with open(os.getcwd() + '/' + dirName + "/B_diff.csv", "a") as myfile:
                    myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

            num_iter = 1
            
            lime_time_avg = 0
            gan_time_avg = 0
                
            for num in range(0, len(dss_test[ID-1])):
                print('iteration: ', num)
                print('ds: ', ds_id)

                lime_time_avg = 0
                gan_time_avg = 0
                
                # optional instance selection - deterministic
                expID = dss_test[ID-1][num]

                print('expID: ', expID)


                # call LIME    
                time_before_lime = time.time()
                exp_nav.explain_instance(expID)
                time_after_lime = time.time()
                lime_time_avg += time_after_lime - time_before_lime
                #print('LIME exp time: ', time_after - time_before)           


                # call GAN
                # Prepare data for GAN
                time_before_gan = time.time()
                index = expID
                offset = num_of_first_rows_to_delete

                # Get local costmap
                local_costmap_original = local_costmap_data.iloc[(index) * costmap_size:(index + 1) * costmap_size, :]
                
                # Make image a np.array deepcopy of local_costmap_original
                image = np.array(copy.deepcopy(local_costmap_original))

                # '''
                # Turn inflated area to free space and 100s to 99s
                image[image == 100] = 99
                image[image != 99] = 0
                # '''tabular_mode

                # Turn every local costmap entry from int to float, so the segmentation algorithm works okay - here probably not needed
                image = image * 1.0
                
                #'''
                # get costmap info
                costmap_info_tmp = local_costmap_info.iloc[index, :]
                costmap_info_tmp = pd.DataFrame(costmap_info_tmp).transpose()
                costmap_info_tmp = costmap_info_tmp.iloc[:, 1:]

                # save costmap info to class variables
                localCostmapOriginX = costmap_info_tmp.iloc[0, 3]
                localCostmapOriginY = costmap_info_tmp.iloc[0, 4]
                localCostmapResolution = costmap_info_tmp.iloc[0, 0]
                #localCostmapHeight = costmap_info_tmp.iloc[0, 2]
                #localCostmapWidth = costmap_info_tmp.iloc[0, 1]

                # get odometry info
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
                
                #if flipped == True:
                #    yaw_sign = math.copysign(1, self.yaw_odom)
                #    self.yaw_odom = -1 * yaw_sign * (math.pi - abs(self.yaw_odom))
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
                    x_temp = int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)
                    y_temp = int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)
                    if 0 <= x_temp < costmap_size and 0 <= y_temp < costmap_size:
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
                plan_x_list = []
                plan_y_list = []
                for i in range(0, plan_tmp_tmp.shape[0], 3):
                    x_temp = int((plan_tmp_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution)    
                    y_temp = int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution)    
                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        #print('x_temp: ', x_temp)
                        #print('y_temp: ', y_temp)
                        #print('\n')
                        plan_x_list.append(x_temp)
                        plan_y_list.append(y_temp)

                #'''

                from GAN import gan            
                gan.predict()

                time_after_gan = time.time()
                gan_time_avg += time_after_gan - time_before_gan

                print('LIME time: ', lime_time_avg / num_iter)
                print('\n')
                print('GAN time: ', gan_time_avg / num_iter)
                print('\n')

                with open(os.getcwd() + '/' + dirName + "/times.csv", "a") as myfile:
                    myfile.write(str(lime_time_avg) + "," + str(gan_time_avg) + "\n")

                segments = exp_nav.segments
                #print('\nexp_nav.exp: ', exp_nav.exp)
                #plt.imshow(segments)
                #plt.savefig('SEGMENTS.png')

                # RGB evaluation
                import PIL.Image
                import os
                path1 = os.getcwd() + '/explanation_results' + '/explanation.png'
                exp_lime_orig = PIL.Image.open(path1).convert('RGB')
                path1 = os.getcwd() + '/' + 'GAN_results' + '/GAN.png'
                exp_gan_orig = PIL.Image.open(path1).convert('RGB')

                exp_lime = np.array(exp_lime_orig)
                #print('exp_lime.shape: ', exp_lime.shape)
                exp_gan = np.array(exp_gan_orig)
                #print('exp_gan.shape: ', exp_gan.shape)

                #exp_lime = exp_nav.temp_img.astype(np.uint8)

                #seg_unique = np.unique(segments)
                #print('seg_unique = ', seg_unique)

                # SEGMENTS eval STARTS
                weights = []
                weights_raw = []

                color_coverage_percent = []

                R_abs_list_lime = []
                G_abs_list_lime = []
                B_abs_list_lime = []

                R_abs_list_gan = []
                G_abs_list_gan = []
                B_abs_list_gan = []

                diff_R_abs_list = []
                diff_G_abs_list = []
                diff_B_abs_list = []

                diff_abs_from_RGB_list = []
                
                channel_avg_diff_list = []
                
                for e in exp_nav.exp[0:-1]:
                    print('\ne = ', e)
                    # if weight is greater than 0
                    if abs(e[1]) >= 0.0:
                        
                        count_R = 0
                        count_G = 0
                        count_B = 0
                        count_avg = 0

                        same_color_count = 0
                        color_count = 0

                        R_abs_lime = 0
                        G_abs_lime = 0
                        B_abs_lime = 0

                        R_abs_gan = 0
                        G_abs_gan = 0
                        B_abs_gan = 0

                        diff_R_abs = 0
                        diff_G_abs = 0
                        diff_B_abs = 0

                        channel_avg_diff = 0

                        # add segment weight
                        weights.append(abs(e[1]))
                        weights_raw.append(e[1])

                        for row in range(0, segments.shape[0]):
                            for columns in range(0, segments.shape[1]):
                                # if a segment pixel
                                if segments[row, columns] == e[0]:
                                    if row not in local_plan_y_list and row not in plan_y_list and row != y_odom_index[0] and columns not in local_plan_x_list and columns not in plan_x_list and columns != x_odom_index[0]:
            
                                        # increase counts
                                        count_R += 1
                                        count_G += 1
                                        count_B += 1
                                        count_avg += 1
                                        color_count += 1

                                        # compare colors
                                        lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                                        gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                                        if lime_color_name == gan_color_name:
                                            same_color_count += 1

                                        # R channel
                                        R_abs_lime += float(exp_lime[row, columns, 0])
                                        R_abs_gan += float(exp_gan[row, columns, 0])
                                        diff_R_abs += abs(float(exp_gan[row, columns, 0]) - float(exp_lime[row, columns, 0]))
                                        
                                        # G channel
                                        G_abs_lime += float(exp_lime[row, columns, 1])
                                        G_abs_gan += float(exp_gan[row, columns, 1])
                                        diff_G_abs = abs(float(exp_gan[row, columns, 1]) - float(exp_lime[row, columns, 1]))
                                            
                                        # B channel
                                        print('\nexp_lime[row, columns, 2] = ', exp_lime[row, columns, 2])
                                        B_abs_lime += float(exp_lime[row, columns, 2])
                                        B_abs_gan += float(exp_gan[row, columns, 2])
                                        diff_B_abs = abs(float(exp_gan[row, columns, 2]) - float(exp_lime[row, columns, 2]))
                                        
                                        # average channel intensity
                                        channel_avg_lime = float(float(exp_lime[row, columns, 0]) + float(exp_lime[row, columns, 1]) + float(exp_lime[row, columns, 2])) / 3
                                        channel_avg_gan = float(float(exp_gan[row, columns, 0]) + float(exp_gan[row, columns, 1]) + float(exp_gan[row, columns, 2])) / 3
                                        
                                        channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)  

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

                        R_abs_lime /= count_R
                        G_abs_lime /= count_G
                        B_abs_lime /= count_B

                        R_abs_gan /= count_R
                        G_abs_gan /= count_G
                        B_abs_gan /= count_B

                        if B_abs_lime > 0:
                            B_abs_gan = abs(B_abs_gan - B_abs_lime)
                            B_abs_lime = 0

                        diff_R_abs /= count_R
                        diff_G_abs /= count_G
                        diff_B_abs /= count_B

                        R_abs_list_lime.append(R_abs_lime)
                        G_abs_list_lime.append(G_abs_lime)
                        B_abs_list_lime.append(B_abs_lime)

                        R_abs_list_gan.append(R_abs_gan)
                        G_abs_list_gan.append(G_abs_gan)
                        B_abs_list_gan.append(B_abs_gan)

                        diff_R_abs_list.append(diff_R_abs)
                        diff_G_abs_list.append(diff_G_abs)
                        diff_B_abs_list.append(diff_B_abs)

                        diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3
                        diff_abs_from_RGB_list.append(diff_abs_from_RGB)

                        channel_avg_diff /= count_avg
                        channel_avg_diff_list.append(channel_avg_diff)

                weights_sum = sum(weights)

                if weights_sum == 0.0:
                    weights = [1.0] * len(weights)
                    weights_sum = sum(weights)

                print('\nweights = ', weights)
                print('\nweights_sum = ', weights_sum)    
                
                explanation_saved_percentage = 0.0
                explanation_saved_percentage_list = []
                weights_percentage = []
                for i in range(0, len(weights)):
                    explanation_saved_percentage += color_coverage_percent[i] * weights[i] / weights_sum
                    explanation_saved_percentage_list.append(color_coverage_percent[i] * weights[i] / weights_sum)
                    weights_percentage.append(100 * weights[i] / weights_sum)

                print('\ndiff_R_abs_list = ', diff_R_abs_list)
                print('\ndiff_G_abs_list = ', diff_G_abs_list)
                print('\ndiff_B_abs_list = ', diff_B_abs_list)

                with open(os.getcwd() + '/' + dirName + "/weights.csv", "a") as myfile:
                    for id in range(0, len(weights_raw) - 1):
                        myfile.write(str(weights_raw[id]) + ",")
                    myfile.write(str(weights_raw[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/R_avg_lime.csv", "a") as myfile:
                    for id in range(0, len(R_abs_list_lime) - 1):
                        myfile.write(str(R_abs_list_lime[id]) + ",")
                    myfile.write(str(R_abs_list_lime[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/G_avg_lime.csv", "a") as myfile:
                    for id in range(0, len(G_abs_list_lime) - 1):
                        myfile.write(str(G_abs_list_lime[id]) + ",")
                    myfile.write(str(G_abs_list_lime[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/B_avg_lime.csv", "a") as myfile:
                    for id in range(0, len(B_abs_list_lime) - 1):
                        myfile.write(str(B_abs_list_lime[id]) + ",")
                    myfile.write(str(B_abs_list_lime[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/R_avg_gan.csv", "a") as myfile:
                    for id in range(0, len(R_abs_list_gan) - 1):
                        myfile.write(str(R_abs_list_gan[id]) + ",")
                    myfile.write(str(R_abs_list_gan[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/G_avg_gan.csv", "a") as myfile:
                    for id in range(0, len(G_abs_list_gan) - 1):
                        myfile.write(str(G_abs_list_gan[id]) + ",")
                    myfile.write(str(G_abs_list_gan[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/B_avg_gan.csv", "a") as myfile:
                    for id in range(0, len(B_abs_list_gan) - 1):
                        myfile.write(str(B_abs_list_gan[id]) + ",")
                    myfile.write(str(B_abs_list_gan[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/R_diff.csv", "a") as myfile:
                    for id in range(0, len( diff_R_abs_list) - 1):
                        myfile.write(str( diff_R_abs_list[id]) + ",")
                    myfile.write(str( diff_R_abs_list[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/G_diff.csv", "a") as myfile:
                    for id in range(0, len( diff_G_abs_list) - 1):
                        myfile.write(str( diff_G_abs_list[id]) + ",")
                    myfile.write(str( diff_G_abs_list[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/B_diff.csv", "a") as myfile:
                    for id in range(0, len( diff_B_abs_list) - 1):
                        myfile.write(str( diff_B_abs_list[id]) + ",")
                    myfile.write(str( diff_B_abs_list[-1]) + "\n")

                with open(os.getcwd() + '/' + dirName + "/segments.csv", "a") as myfile:
                    myfile.write(str(explanation_saved_percentage) + ",")
                    myfile.write(str(sum(diff_R_abs_list) / len(diff_R_abs_list)) + ",")
                    myfile.write(str(sum(diff_G_abs_list) / len(diff_G_abs_list)) + ",")
                    myfile.write(str(sum(diff_B_abs_list) / len(diff_B_abs_list)) + ",")
                    myfile.write(str(sum(diff_abs_from_RGB_list) / len(diff_abs_from_RGB_list)) + ",") 
                    myfile.write(str(sum(channel_avg_diff_list) / len(channel_avg_diff_list)) + "\n")
                # SEGMENTS eval ENDS

            
                # LOCAL PLAN eval STARTS 
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                diff_R_abs = 0
                diff_G_abs = 0
                diff_B_abs = 0

                channel_avg_diff = 0

                for i in range(0, len(local_plan_x_list)):
                    row = local_plan_y_list[i]
                    columns = local_plan_x_list[i]

                    # increase counts
                    count_R += 1
                    count_G += 1
                    count_B += 1
                    count_avg += 1
                    color_count += 1

                    # compare colors
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    # absolute R channel difference
                    diff_R_abs += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))
                        
                    # absolute G channel difference
                    diff_G_abs = abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))
                        
                    # absolute B channel difference
                    diff_B_abs = abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2]))
                    
                    # average channel intensity
                    channel_avg_lime = float(int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                    channel_avg_gan = float(int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3
                    
                    channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)

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
                
                diff_R_abs /= count_R
                diff_G_abs /= count_G
                diff_B_abs /= count_B

                diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3

                channel_avg_diff /= count_avg

                with open(os.getcwd() + '/' + dirName + "/local_plan.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + ",")
                    myfile.write(str(diff_R_abs) + ",")
                    myfile.write(str(diff_G_abs) + ",")
                    myfile.write(str(diff_B_abs) + ",")
                    myfile.write(str(diff_abs_from_RGB) + ",") 
                    myfile.write(str(channel_avg_diff) + "\n")
                # LOCAL PLAN eval ENDS


                # GLOBAL PLAN eval STARTS 
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                diff_R_abs = 0
                diff_G_abs = 0
                diff_B_abs = 0

                channel_avg_diff = 0

                for i in range(0, len(plan_x_list)):
                    row = plan_y_list[i]
                    columns = plan_x_list[i]

                    # increase counts
                    count_R += 1
                    count_G += 1
                    count_B += 1
                    count_avg += 1
                    color_count += 1

                    # compare colors
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    # absolute R channel difference
                    diff_R_abs += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))
                        
                    # absolute G channel difference
                    diff_G_abs = abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))
                        
                    # absolute B channel difference
                    diff_B_abs = abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2]))
                    
                    # average channel intensity
                    channel_avg_lime = float(int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                    channel_avg_gan = float(int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3
                    
                    channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)

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
                
                diff_R_abs /= count_R
                diff_G_abs /= count_G
                diff_B_abs /= count_B

                diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3

                channel_avg_diff /= count_avg

                with open(os.getcwd() + '/' + dirName + "/global_plan.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + ",")
                    myfile.write(str(diff_R_abs) + ",")
                    myfile.write(str(diff_G_abs) + ",")
                    myfile.write(str(diff_B_abs) + ",")
                    myfile.write(str(diff_abs_from_RGB) + ",") 
                    myfile.write(str(channel_avg_diff) + "\n")
                # GLOBAL PLAN eval ENDS


                # ROBOT POSITION eval STARTS 
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                diff_R_abs = 0
                diff_G_abs = 0
                diff_B_abs = 0

                channel_avg_diff = 0

                for i in range(0, 1):
                    row = y_odom_index[0]
                    print('\ny_odom_index = ', y_odom_index)
                    columns = x_odom_index[0]
                    print('\nx_odom_index = ', x_odom_index)

                    # increase counts
                    count_R += 1
                    count_G += 1
                    count_B += 1
                    count_avg += 1
                    color_count += 1

                    # compare colors
                    lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                    gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                    if lime_color_name == gan_color_name:
                        same_color_count += 1

                    # absolute R channel difference
                    diff_R_abs += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))
                        
                    # absolute G channel difference
                    diff_G_abs = abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))
                        
                    # absolute B channel difference
                    diff_B_abs = abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2]))
                    
                    # average channel intensity
                    channel_avg_lime = float(int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                    channel_avg_gan = float(int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3
                    
                    channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)

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
                
                diff_R_abs /= count_R
                diff_G_abs /= count_G
                diff_B_abs /= count_B

                diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3

                channel_avg_diff /= count_avg

                with open(os.getcwd() + '/' + dirName + "/robot_position.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + ",")
                    myfile.write(str(diff_R_abs) + ",")
                    myfile.write(str(diff_G_abs) + ",")
                    myfile.write(str(diff_B_abs) + ",")
                    myfile.write(str(diff_abs_from_RGB) + ",") 
                    myfile.write(str(channel_avg_diff) + "\n")
                # ROBOT POSITION eval ENDS


                
                # OBSTACLES eval STARTS
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                diff_R_abs = 0
                diff_G_abs = 0
                diff_B_abs = 0

                channel_avg_diff = 0

                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        if image[i, j] == 99:
                            row = i
                            columns = j
                            
                            # increase counts
                            count_R += 1
                            count_G += 1
                            count_B += 1
                            count_avg += 1
                            color_count += 1

                            # compare colors
                            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                            if lime_color_name == gan_color_name:
                                same_color_count += 1

                            # absolute R channel difference
                            diff_R_abs += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))
                                
                            # absolute G channel difference
                            diff_G_abs = abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))
                                
                            # absolute B channel difference
                            diff_B_abs = abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2]))
                            
                            # average channel intensity
                            channel_avg_lime = float(int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                            channel_avg_gan = float(int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3
                            
                            channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)                                        

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
                
                diff_R_abs /= count_R
                diff_G_abs /= count_G
                diff_B_abs /= count_B

                diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3

                channel_avg_diff /= count_avg

                with open(os.getcwd() + '/' + dirName + "/obstacles.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + ",")
                    myfile.write(str(diff_R_abs) + ",")
                    myfile.write(str(diff_G_abs) + ",")
                    myfile.write(str(diff_B_abs) + ",")
                    myfile.write(str(diff_abs_from_RGB) + ",") 
                    myfile.write(str(channel_avg_diff) + "\n")
                # OBSTACLES eval ENDS


                # FREE SPACE eval STARTS
                count_R = 0
                count_G = 0
                count_B = 0
                count_avg = 0

                same_color_count = 0
                color_count = 0

                diff_R_abs = 0
                diff_G_abs = 0
                diff_B_abs = 0

                channel_avg_diff = 0

                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        if image[i, j] == 0:
                            row = i
                            columns = j
                            
                            # increase counts
                            count_R += 1
                            count_G += 1
                            count_B += 1
                            count_avg += 1
                            color_count += 1

                            # compare colors
                            lime_color_name =  convert_rgb_to_names_my((exp_lime[row, columns, 0],exp_lime[row, columns, 1],exp_lime[row, columns, 2]))
                            gan_color_name =  convert_rgb_to_names_my((exp_gan[row, columns, 0],exp_gan[row, columns, 1],exp_gan[row, columns, 2]))
                            if lime_color_name == gan_color_name:
                                same_color_count += 1

                            # absolute R channel difference
                            diff_R_abs += abs(int(exp_gan[row, columns, 0]) - int(exp_lime[row, columns, 0]))
                                
                            # absolute G channel difference
                            diff_G_abs = abs(int(exp_gan[row, columns, 1]) - int(exp_lime[row, columns, 1]))
                                
                            # absolute B channel difference
                            diff_B_abs = abs(int(exp_gan[row, columns, 2]) - int(exp_lime[row, columns, 2]))
                            
                            # average channel intensity
                            channel_avg_lime = float(int(exp_lime[row, columns, 0]) + int(exp_lime[row, columns, 1]) + int(exp_lime[row, columns, 2])) / 3
                            channel_avg_gan = float(int(exp_gan[row, columns, 0]) + int(exp_gan[row, columns, 1]) + int(exp_gan[row, columns, 2])) / 3
                            
                            channel_avg_diff += abs(channel_avg_gan - channel_avg_lime)    

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
                
                diff_R_abs /= count_R
                diff_G_abs /= count_G
                diff_B_abs /= count_B

                diff_abs_from_RGB = (diff_R_abs + diff_G_abs + diff_B_abs) / 3

                channel_avg_diff /= count_avg

                with open(os.getcwd() + '/' + dirName + "/free_space.csv", "a") as myfile:
                    myfile.write(str(color_coverage_percent) + ",")
                    myfile.write(str(diff_R_abs) + ",")
                    myfile.write(str(diff_G_abs) + ",")
                    myfile.write(str(diff_B_abs) + ",")
                    myfile.write(str(diff_abs_from_RGB) + ",") 
                    myfile.write(str(channel_avg_diff) + "\n")
                # FREE SPACE eval ENDS


# GUI
def tkinterStart():
    import tkinter
    # GUI architecture
    root = tkinter.Tk()
    root.title("Navigation Explainer")

    # Dropdown menu options
    options_exp_alg = [
        "LIME",
        "Anchors",
        "SHAP"
    ]
    # datatype of menu text
    clicked_exp_alg = tkinter.StringVar()  
    # initial menu text
    clicked_exp_alg.set( "Choose explanation algorithm" )  
    # Create Dropdown menu
    drop = tkinter.OptionMenu( root , clicked_exp_alg , *options_exp_alg )
    drop.pack()

    options_exp_mode = [
        "image",
        "tabular",
        "tabular_costmap"
    ]
    # datatype of menu text
    clicked_exp_mode = tkinter.StringVar()  
    # initial menu text
    clicked_exp_mode.set( "Choose explanation method" )  
    # Create Dropdown menu
    drop = tkinter.OptionMenu( root , clicked_exp_mode , *options_exp_mode )
    drop.pack()

    options_model_mode = [
        "regression",
        "classification",
        "regression_normalized_around_deviation",
        "regression_normalized"
    ]
    # datatype of menu text
    clicked_model_mode = tkinter.StringVar()  
    # initial menu text
    clicked_model_mode.set( "Choose underlying model" )  
    # Create Dropdown menu
    drop = tkinter.OptionMenu( root , clicked_model_mode , *options_model_mode )
    drop.pack()

    def loadToGlobalVars():
        global explanation_alg, explanation_mode, underlying_model_mode
        explanation_alg = clicked_exp_alg.get()
        explanation_mode = clicked_exp_mode.get()
        underlying_model_mode = clicked_model_mode.get()
        
        print('\nexplanation algorithm: ', explanation_alg)
        print('explanation mode:', explanation_mode)
        print('underlying_model_mode: ', underlying_model_mode)

    button_load = tkinter.Button( root , text = "Confirm choices" , command = loadToGlobalVars ).pack()
    
    buttonSingle = tkinter.Button(root, text='Run single', height=3, width=25, command=Single, fg='black', bg='white')
    #buttonSingle.grid(row=0,column=0)
    buttonSingle.pack()

    buttonEvaluate = tkinter.Button(root, text='Evaluate', height=3, width=25, command=Evaluate, fg='black', bg='white')
    #buttonEvaluate.grid(row=2,column=0)
    buttonEvaluate.pack()

    buttonCreateDataset = tkinter.Button(root, text='Create dataset for GAN', height=3, width=25, command=CreateDataset, fg='black', bg='white')
    #buttonCreateDataset.grid(row=1,column=0)
    buttonCreateDataset.pack()

    buttonRunGAN = tkinter.Button(root, text='Run GAN', height=3, width=25, command=RunGAN, fg='black', bg='white')
    #buttonRunGAN.grid(row=3,column=0)
    buttonRunGAN.pack()

    buttonEvaluateLIMEvsGAN = tkinter.Button(root, text='Evaluate LIME vs GAN', height=3, width=25, command=EvaluateLIMEvsGAN, fg='black', bg='white')
    #buttonEvaluateLIMEvsGAN.grid(row=4,column=0)
    buttonEvaluateLIMEvsGAN.pack()

    root.mainloop()

tkinterStart()

