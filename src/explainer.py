#!/usr/bin/env python3

# Defining parameters - global variables

# test type: 'single', 'dataset_creation', 'evaluation', 'GAN'
test_type = 'evaluation'

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

def preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom):
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

    return num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom




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

    num_of_first_rows_to_delete, local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom = preprocess_data(local_costmap_info, odom, amcl_pose, cmd_vel, tf_odom_map, tf_map_odom)
    
    # Dataset creation
    X_train = []
    X_test = []

    # if LIME image
    if explanation_mode == 'image':
        # output_class_name - not important for LIME image
        output_class_name = cmd_vel.columns.values[0]  # [0] - 'cmd_vel_lin_x'  or [1] - ' cmd_vel_ang_z'

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
            expID = 60

            # random instance selection
            # import random
            # expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

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
            evaluation_size = 1
            for i in range(0, evaluation_size):
                # optional instance selection - deterministic
                expID = 60

                # random instance selection
                # import random
                # expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

                for j in range(0, 3):
                    start = time.time()
                    exp_nav.explain_instance_evaluation(expID, j)
                    end = time.time()
                    with open("explanations.txt", "a") as myfile:
                        myfile.write('- ' + str(round(end - start, 4)) + '\n')
                
                with open("explanations.txt", "a") as myfile:
                        myfile.write('\n')        

        elif test_type == 'GAN':
            # optional instance selection - deterministic
            expID = 15

            # random instance selection
            # import random
            # expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info

            index = expID
            offset = num_of_first_rows_to_delete
            costmap_size = 160

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
            image_flipped = np.flip(image, axis=1)

            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(image_flipped.astype('float64'))  # , aspect='auto')

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
            
            # plot robots' location, orientation and local plan
            ax.scatter(x_odom_index, y_odom_index, c='black', marker='o')
            ax.quiver(x_odom_index, y_odom_index, yaw_odom_x, yaw_odom_y, color='black')

            local_plan_tmp = teb_local_plan.loc[teb_local_plan['ID'] == index + offset]
            local_plan_tmp = local_plan_tmp.iloc[:, 1:]
            # indices of local plan's poses in local costmap
            local_plan_x_list = []
            local_plan_y_list = []
            for i in range(1, local_plan_tmp.shape[0]):
                local_plan_x_list.append(160 - int((local_plan_tmp.iloc[i, 0] - localCostmapOriginX) / localCostmapResolution))
                local_plan_y_list.append(int((local_plan_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution))
            ax.scatter(local_plan_x_list, local_plan_y_list, c='red', marker='o')

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
                if 0 <= x_temp <= 159:
                    plan_x_list.append(x_temp)
                    plan_y_list.append(
                        int((plan_tmp_tmp.iloc[i, 1] - localCostmapOriginY) / localCostmapResolution))
            plt.scatter(plan_x_list, plan_y_list, c='blue', marker='o')
            # '''

            plt.savefig('input.png', transparent=False)
            plt.close()
            plt.clf()

            from models import create_model
            #model = create_model(opt)  # create a model given opt.model and other options
            #model.setup(opt)  # regular setup: load and print networks; create schedulers
            #model.set_input(data)  # unpack data from data loader
            #model.test()  # run inference
            #visuals = model.get_current_visuals()  # get image results
            #img_path = model.get_image_paths()  # get image paths














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

