#!/usr/bin/env python3

# consider moving explainer.py out of ROS package

# Defining parameters

# test type: 'single', 'evaluation'
test_type = 'single'

# possible explanation algorithms: 'lime', 'shap', 'anchors'
explanation_alg = 'lime'

# possible explanation modes: 'tabular', 'image', 'tabular_costmap', 'text'
explanation_mode = 'image'

# tabular explanation modes: 'regression', 'classification'
tabular_mode = 'regression'

# one hot encoding: 'True' or 'False' - needed for tabular classification
one_hot_encoding = True

# header of the output class/column
output_class_name = 'beginning' # just to see if it was changed properly

# set number of samples (does not define/affect the number of samples in LIME image)
num_samples = 256




if explanation_alg == 'lime':

    # consider moving data DataLoader outside lime
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

    # Delete entries with 'None' frame
    # Detect number of entries with 'None' frame based on local_costmap_info
    num_of_first_rows_to_delete = len(local_costmap_info[local_costmap_info['frame'] == 'None'])
    '''
    print('num_of_first_rows_to_delete:')
    print(num_of_first_rows_to_delete)
    print('\n')
    '''

    # Delete entries with 'None' frame from local_costmap_info
    local_costmap_info.drop(index=local_costmap_info.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('local_costmap_info after deleting entries with None frame:')
    print(local_costmap_info)
    print('\n')
    '''

    # Delete entries with 'None' frame from odom
    odom.drop(index=odom.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('odom after deleting entries with None frame:')
    print(odom)
    print('\n')
    '''

    # Delete entries with 'None' frame from amcl_pose
    amcl_pose.drop(index=amcl_pose.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('amcl_pose after deleting entries with None frame:')
    print(amcl_pose)
    print('\n')
    '''

    # Delete entries with 'None' frame from cmd_vel
    cmd_vel.drop(index=cmd_vel.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('cmd_vel after deleting entries with None frame:')
    print(cmd_vel)
    print('\n')
    '''

    # Delete entries with 'None' frame from tf_odom_map
    tf_odom_map.drop(index=tf_odom_map.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('tf_odom_map after deleting entries with None frame:')
    print(tf_odom_map)
    print('\n')
    '''

    # Delete entries with 'None' frame from tf_map_odom
    tf_map_odom.drop(index=tf_map_odom.index[:num_of_first_rows_to_delete], axis=0, inplace=True)
    '''
    print('tf_map_odom after deleting entries with None frame:')
    print(tf_map_odom)
    print('\n')
    '''

    # Deletion of entries with 'None' frame from plans and footprints has not yet been implemented,
    # because after deleting rows from dataframes, indexes retain their values,
    # so that further plans' and footprints' instances can be indexed on the same way.


    # Dataset creation
    X_train = []
    X_test = []

    # If the LIME tabular is to be used
    # ' cmd_vel_ang_z' - sometime in the future to correct this gap - delete it
    if explanation_mode == 'tabular':

'''

    from lime_explainer import DatasetCreator

    # Select input for explanation algorithm
    X = odom.iloc[:,6:8] # input for explanation are odometry velocities
    #print(X)

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

# if LIME image will be used
elif explanationMode == 'image':
    # optional selection - deterministic
    expID = 15

    # Representative situations/costmaps
    # New datasets:
    # Dataset1: #60, #165
    # Dataset2: #15 #163, #599
    # Old datasets:
    # Dataset1: 163
    # Dataset2:
    # Dataset3:
    # Dataset4: #100 #190
    # Dataset HARL Workshop 2021 paper: #71
    
    # random selection
    #import random
    #expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info
    
    output_class_name = cmd_vel.columns.values[0] # [0] - 'cmd_vel_lin_x'  or [1] - ' cmd_vel_ang_z'

# Explanation
from lime_explainer import ExplainNavigation

expNav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, 
amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train, X_test, mode, explanationMode, expID, num_samples, output_class_name, numOfFirstRowsToDelete, footprints)

if testType == 'single':
    expNav.explain_instance(expID)
    #expNav.testSegmentation()
    #expNav.testLocalCostmap()

elif testType == 'evaluation':
    expID = 15
    expNav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                      current_goal, local_costmap_data, local_costmap_info,
                                                      amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train,
                                                      X_test, mode, explanationMode, expID, num_samples,
                                                      output_class_name, numOfFirstRowsToDelete, footprints)
    expNav.explain_instance_evaluation(expID, 1)

    expID = 163
    expNav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                      current_goal, local_costmap_data, local_costmap_info,
                                                      amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train,
                                                      X_test, mode, explanationMode, expID, num_samples,
                                                      output_class_name, numOfFirstRowsToDelete, footprints)
    expNav.explain_instance_evaluation(expID, 2)

    expID = 599
    expNav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan,
                                                      current_goal, local_costmap_data, local_costmap_info,
                                                      amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train,
                                                      X_test, mode, explanationMode, expID, num_samples,
                                                      output_class_name, numOfFirstRowsToDelete, footprints)
    expNav.explain_instance_evaluation(expID, 3)

    import time
    for i in range(0, 50):
        expID = random.randint(0, local_costmap_info.shape[0])
        for j in range(0, 9):
            start = time.time()
            expNav.explain_instance(expID, j, i)
            end = time.time()
            with open("explanations.txt", "a") as myfile:
                myfile.write('- ' + str(round(end - start, 4)) + '\n')
        with open("explanations.txt", "a") as myfile:
                myfile.write('\n')
    
'''

