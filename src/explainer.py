#!/usr/bin/env python3

# Global variables
ds_id = 1
ds = 'ds' + str(ds_id)

print('dataset: ', ds)

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

def Single():
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
                                                        tabular_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)
    
    print('\nexpID range: ', (0, local_costmap_info.shape[0] - num_of_first_rows_to_delete))
    print('\nnum_of_first_rows_to_delete = ', num_of_first_rows_to_delete)    

    choose_random_instance = True

    if choose_random_instance == True:
        # random instance selection
        #print('\nexpID range: ', (0, local_costmap_info.shape[0] - num_of_first_rows_to_delete))
        import random
        expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete)
        print('\nexpID: ', expID)
    else:     
        # optional instance selection - deterministic
        expID = 150 #DS1: #51 #78 #84 #144; #DS2: #260
        print('\nexpID: ', expID)

    import time
    start_total_exp_time = time.time()
    exp_nav.explain_instance(expID)
    end_total_exp_time = time.time()
    print('\nTotal explanation time: ', end_total_exp_time - start_total_exp_time)
    print('\nEND of EXPLANATION!!!')
    #exp_nav.testSegmentation(expID)

def Evaluate():
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
                                                        tabular_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)

    import time
    evaluation_sample_size = 50
    
    with open("explanations.txt", "a") as myfile:
        myfile.write('explain_instance_time\n')
    
    for i in range(0, evaluation_sample_size):
        print('\ni = ', i)
        choose_random_instance = True

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

        with open('IDs.csv', "a") as myfile:
            myfile.write(str(expID) + '\n')
        myfile.close() 

        total_exp_start = time.time()
        exp_nav.explain_instance_evaluation(expID, i)
        total_exp_end = time.time()
        with open("explanations.txt", "a") as myfile:
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
    from lime_explainer import DataLoader
    
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
                                                        tabular_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples)

    #'''
    with open('costmap_data.csv', "a") as myfile:
            myfile.write('picture_ID,width,height,origin_x,origin_y,resolution\n')

    with open('local_plan_coordinates.csv', "a") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open('global_plan_coordinates.csv', "a") as myfile:
            myfile.write('picture_ID,position_x,position_y\n')

    with open('robot_coordinates.csv', "a") as myfile:
        myfile.write('picture_ID,position_x,position_y\n')
    #''' 

    dataset_size = local_costmap_info.shape[0] - num_of_first_rows_to_delete
    print('local_costmap_info.shape[0]: ', local_costmap_info.shape[0])
    print('num_of_first_rows_to_delete: ', num_of_first_rows_to_delete)
    print('dataset_size: ', dataset_size)
    #import random    
    for i in range(0, dataset_size, 1):
        # optional instance selection - deterministic
        expID = i
        print('expID: ', expID)

        # random instance selection
        #expID = random.randint(0, local_costmap_info.shape[0] - num_of_first_rows_to_delete) 

        exp_nav.explain_instance_dataset(expID, i)
        #exp_nav.testSegmentation(expID)

def RunGAN():
    # Data loading
    from lime_explainer import DataLoader
    
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
    from lime_explainer import ExplainNavigation

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

    gray_shade = 180
    white_shade = 255
    from skimage.color import gray2rgb
    image = gray2rgb(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = gray_shade
            elif image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 99:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = white_shade

    import os
    path_core = os.getcwd()

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
    
    fig.savefig('input.png', transparent=False)
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
    # Data loading
    from lime_explainer import DataLoader
    
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
    from lime_explainer import ExplainNavigation

    exp_nav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                        current_goal, local_costmap_data, local_costmap_info,
                                                        amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info,
                                                        tabular_mode, explanation_mode, explanation_alg, num_of_first_rows_to_delete, footprints, output_class_name,
                                                        X_train, X_test, y_train, y_test, num_samples, plot=False)

    from test_color import create_dict_my, convert_rgb_to_names_my
    create_dict_my()

    import pandas as pd
    import time
    import numpy as np
    import copy
    import matplotlib.pyplot as plt

    with open("times.csv", "a") as myfile:
            myfile.write("lime,gan\n")

    with open("weights.csv", "a") as myfile:
            myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")        

    with open("segments.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("local_plan.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("global_plan.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("obstacles.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("free_space.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("robot_position.csv", "a") as myfile:
            myfile.write("color_similarity_percentage,R_abs,G_abs,B_abs,abs_from_RGB,channel_abs\n")

    with open("gan_times.csv", "a") as myfile:
            myfile.write("predict_time\n")

    with open("R_avg_lime.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("R_avg_gan.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("R_diff.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("G_avg_lime.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("G_avg_gan.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("G_diff.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("B_avg_lime.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("B_avg_gan.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    with open("B_diff.csv", "a") as myfile:
        myfile.write("w1,w2,w3,w4,w5,w6,w7,w8\n")

    num_iter = 1
    
    lime_time_avg = 0
    gan_time_avg = 0

    #R_PERC = [0.0] * 10
    #G_PERC = [0.0] * 10
    #B_PERC = [0.0] * 10

    exp_IDs_list_test_ds1 = [5, 23, 44, 52, 75, 88, 94, 104, 118, 128, 136, 150, 151, 189, 190, 209, 223, 225, 229, 242, 252]
    exp_IDs_list_test_ds2 = [6, 9, 12, 13, 22, 45, 63, 75, 103, 105, 109, 123, 126, 128, 150, 153, 154, 161, 166, 167, 182, 203, 214, 215, 220, 234, 237, 247, 249, 252, 257, 258, 262, 271, 275, 277, 278, 294, 337, 348, 366, 373, 387, 390, 391, 413, 420, 426, 430, 436, 441, 445, 446, 451, 455, 466, 468, 482, 492, 495, 505, 507, 514, 525, 580, 585, 599, 602, 612, 620, 625, 639, 640, 641, 667, 676, 688, 690, 698]
        
    for num in range(10, 11):
    #for num in range(0, len(exp_IDs_list_test_ds2)):
    #for num in range(0, num_iter):
        print('iteration: ', num)

        lime_time_avg = 0
        gan_time_avg = 0
        
        # optional instance selection - deterministic
        expID = exp_IDs_list_test_ds1[num]
        #expID = exp_IDs_list_test_ds2[num]
        #expID = 203

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

        with open("times.csv", "a") as myfile:
            myfile.write(str(lime_time_avg) + "," + str(gan_time_avg) + "\n")

        segments = exp_nav.segments
        #print('\nexp_nav.exp: ', exp_nav.exp)
        #plt.imshow(segments)
        #plt.savefig('SEGMENTS.png')

        # RGB evaluation
        import PIL.Image
        import os
        path1 = os.getcwd() + '/explanation.png'
        exp_lime_orig = PIL.Image.open(path1).convert('RGB')
        path1 = os.getcwd() + '/GAN.png'
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

        with open("weights.csv", "a") as myfile:
            for id in range(0, len(weights_raw) - 1):
                myfile.write(str(weights_raw[id]) + ",")
            myfile.write(str(weights_raw[-1]) + "\n")

        with open("R_avg_lime.csv", "a") as myfile:
            for id in range(0, len(R_abs_list_lime) - 1):
                myfile.write(str(R_abs_list_lime[id]) + ",")
            myfile.write(str(R_abs_list_lime[-1]) + "\n")

        with open("G_avg_lime.csv", "a") as myfile:
            for id in range(0, len(G_abs_list_lime) - 1):
                myfile.write(str(G_abs_list_lime[id]) + ",")
            myfile.write(str(G_abs_list_lime[-1]) + "\n")

        with open("B_avg_lime.csv", "a") as myfile:
            for id in range(0, len(B_abs_list_lime) - 1):
                myfile.write(str(B_abs_list_lime[id]) + ",")
            myfile.write(str(B_abs_list_lime[-1]) + "\n")

        with open("R_avg_gan.csv", "a") as myfile:
            for id in range(0, len(R_abs_list_gan) - 1):
                myfile.write(str(R_abs_list_gan[id]) + ",")
            myfile.write(str(R_abs_list_gan[-1]) + "\n")

        with open("G_avg_gan.csv", "a") as myfile:
            for id in range(0, len(G_abs_list_gan) - 1):
                myfile.write(str(G_abs_list_gan[id]) + ",")
            myfile.write(str(G_abs_list_gan[-1]) + "\n")

        with open("B_avg_gan.csv", "a") as myfile:
            for id in range(0, len(B_abs_list_gan) - 1):
                myfile.write(str(B_abs_list_gan[id]) + ",")
            myfile.write(str(B_abs_list_gan[-1]) + "\n")

        with open("R_diff.csv", "a") as myfile:
            for id in range(0, len( diff_R_abs_list) - 1):
                myfile.write(str( diff_R_abs_list[id]) + ",")
            myfile.write(str( diff_R_abs_list[-1]) + "\n")

        with open("G_diff.csv", "a") as myfile:
            for id in range(0, len( diff_G_abs_list) - 1):
                myfile.write(str( diff_G_abs_list[id]) + ",")
            myfile.write(str( diff_G_abs_list[-1]) + "\n")

        with open("B_diff.csv", "a") as myfile:
            for id in range(0, len( diff_B_abs_list) - 1):
                myfile.write(str( diff_B_abs_list[id]) + ",")
            myfile.write(str( diff_B_abs_list[-1]) + "\n")

        with open("segments.csv", "a") as myfile:
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

        with open("local_plan.csv", "a") as myfile:
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

        with open("global_plan.csv", "a") as myfile:
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

        with open("robot_position.csv", "a") as myfile:
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

        with open("obstacles.csv", "a") as myfile:
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

        with open("free_space.csv", "a") as myfile:
            myfile.write(str(color_coverage_percent) + ",")
            myfile.write(str(diff_R_abs) + ",")
            myfile.write(str(diff_G_abs) + ",")
            myfile.write(str(diff_B_abs) + ",")
            myfile.write(str(diff_abs_from_RGB) + ",") 
            myfile.write(str(channel_avg_diff) + "\n")
        # FREE SPACE eval ENDS


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
clicked_exp_mode.set( "Choose explanation method" )  
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
clicked_tab_mode.set( "Choose tabular explanation method" )  
# Create Dropdown menu
drop = OptionMenu( root , clicked_tab_mode , *options_tab_mode )
drop.pack()

def loadToGlobalVars():
    global explanation_alg, explanation_mode, tabular_mode
    explanation_alg = clicked_exp_alg.get()
    explanation_mode = clicked_exp_mode.get()
    tabular_mode = clicked_tab_mode.get()
    
    print('\nexplanation algorithm: ', explanation_alg)
    print('explanation mode:', explanation_mode)
    print('tabular_mode: ', tabular_mode)

button_load = Button( root , text = "Confirm choices" , command = loadToGlobalVars ).pack()
  
buttonSingle = Button(root, text='Run single', height=3, width=25, command=Single, fg='black', bg='white')
#buttonSingle.grid(row=0,column=0)
buttonSingle.pack()

buttonEvaluate = Button(root, text='Evaluate', height=3, width=25, command=Evaluate, fg='black', bg='white')
#buttonEvaluate.grid(row=2,column=0)
buttonEvaluate.pack()

buttonCreateDataset = Button(root, text='Create dataset for GAN', height=3, width=25, command=CreateDataset, fg='black', bg='white')
#buttonCreateDataset.grid(row=1,column=0)
buttonCreateDataset.pack()

buttonRunGAN = Button(root, text='Run GAN', height=3, width=25, command=RunGAN, fg='black', bg='white')
#buttonRunGAN.grid(row=3,column=0)
buttonRunGAN.pack()

buttonEvaluateLIMEvsGAN = Button(root, text='Evaluate LIME vs GAN', height=3, width=25, command=EvaluateLIMEvsGAN, fg='black', bg='white')
#buttonEvaluateLIMEvsGAN.grid(row=4,column=0)
buttonEvaluateLIMEvsGAN.pack()

root.mainloop()

###### TKINTER END ######


