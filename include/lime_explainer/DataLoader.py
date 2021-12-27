#!/usr/bin/env python3

import pandas as pd

# Load output data
def load_output_data():
    cmd_vel = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Output/cmd_vel.csv')
    #cmd_vel.head()
    '''
    print('cmd_vel:')
    print(cmd_vel)
    print('\n')
    '''

    '''
    print('cmd_vel.shape: ', cmd_vel.shape)
    print('\n')
    '''
    
    return cmd_vel


# Load input data
def load_input_data():
    odom = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/odom.csv')
    #odom.head()
    '''
    print('odom:')
    print(odom)
    print('\n')
    '''

    '''
    print('odom.shape: ', odom.shape)
    print('\n')
    '''


    teb_global_plan = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/teb_global_plan.csv')
    #teb_global_plan.head()
    '''
    print('teb_global_plan:')
    print(teb_global_plan)
    print('\n')
    '''

    '''
    print('teb_global_plan.shape: ', teb_global_plan.shape)
    print('\n')
    ''' 


    teb_local_plan = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/teb_local_plan.csv')
    #teb_local_plan.head()
    '''
    print('teb_local_plan:')
    print(teb_local_plan)
    print('\n')
    '''
    
    '''
    print('teb_local_plan.shape: ', teb_local_plan.shape)
    print('\n')
    '''

    
    current_goal = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/current_goal.csv')
    #current_goal.head()
    '''
    print('current_goal:')
    print(current_goal)
    print('\n')
    '''

    '''
    print('current_goal.shape: ', current_goal.shape)
    print('\n')
    '''

    
    local_costmap_info = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/local_costmap_info.csv')
    #local_costmap_info.head()
    '''
    print('local_costmap_info:')
    print(local_costmap_info)
    print('\n')
    '''

    '''
    print('local_costmap_info.shape: ', local_costmap_info.shape)
    print('\n')
    '''

    
    local_costmap_data = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/local_costmap_data.csv',
                                     header=None)
    #local_costmap_data.head()
    '''
    print('local_costmap_data:')
    print(local_costmap_data)
    print('\n')
    '''

    '''
    print('local_costmap_data.shape: ', local_costmap_data.shape)
    print('\n')
    '''

    if local_costmap_data.shape[1] > local_costmap_info.iloc[0, 2]:
        # drop last column in the local_costmap_data - NaN data (',')
        local_costmap_data = local_costmap_data.iloc[:, :-1]
        '''
        print('local_costmap_data after dropping NaN data:')
        print(local_costmap_data)
        print('\n')
        '''

        '''
        print('local_costmap_data.shape after dropping NaN data: ', local_costmap_data.shape)
        print('\n')
        '''


    plan = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/plan.csv')
    #plan.head()
    '''
    print('plan:')
    print(plan)
    print('\n')
    '''

    '''
    print('plan.shape: ', plan.shape)
    print('\n')
    '''


    amcl_pose = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/amcl_pose.csv')
    #amcl_pose.head()
    '''
    print('amcl_pose:')
    print(amcl_pose)
    print('\n')
    '''

    '''
    print('amcl_pose.shape: ', amcl_pose.shape)
    print('\n')
    '''


    tf_odom_map = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/tf_odom_map.csv')
    #tf_odom_map.head()
    '''
    print('tf_odom_map:')
    print(tf_odom_map)
    print('\n')
    '''

    '''
    print('tf_odom_map.shape: ', tf_odom_map.shape)
    print('\n')
    '''


    tf_map_odom = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/tf_map_odom.csv')
    #tf_map_odom.head()
    '''
    print('tf_map_odom:')
    print(tf_map_odom)
    print('\n')
    '''

    '''
    print('tf_map_odom.shape: ', tf_map_odom.shape)
    print('\n')
    '''


    map_info = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/map_info.csv')
    #map_info.head()
    '''
    print('map_info:')
    print(map_info)
    print('\n')
    '''

    '''
    print('map_info.shape: ', map_info.shape)
    print('\n')
    '''


    map_data = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/map_data.csv', header=None)
    #map_data.head()
    '''
    print('map_data:')
    print(map_data)
    print('\n')
    '''
    
    '''
    print('map_data.shape: ', map_data.shape)
    print('\n')
    '''

    if map_data.shape[1] > map_info.iloc[0, 2]:
        # drop last column in the map_data - NaN data (',')
        map_data = map_data.iloc[:, :-1]
        '''
        print('map_data after dropping NaN data:')
        print(map_data)
        print('\n')
        '''

        '''
        print('map_data.shape after dropping NaN data: ', map_data.shape)
        print('\n')
        '''


    
    footprints = pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/footprints.csv')
    #footprints.head()
    '''
    print('footprints:')
    print(footprints)
    print('\n')
    '''

    '''
    print('footprints.shape: ', footprints.shape)
    print('\n')
    '''


    return odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints
