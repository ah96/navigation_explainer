#!/usr/bin/env python3


# Defining parameters

# possible explanation modes: 'tabular', 'image', 'tabular_costmap'
explanationMode = 'image'

# possible (tabular) modes: 'regression', 'classification'
mode = 'regression'

# one hot encoding - needed for LIME tabular classification
one_hot_encoding = True

# header of the output class/column
output_class_name = 'beginning' # just to differ if it was changed properly

# choose number of samples
num_samples = 256



# Data loading

from lime_explainer import DataLoader

# load output data
cmd_vel = DataLoader.load_output_data()

#print("output loaded")

#load input data
odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, footprints = DataLoader.load_input_data()

#print("input loaded")

# Izbaciti unose sa 'None' frejmom

# Detektuj broj unosa sa 'None' frejmom na osnovu local_costmap_info
numOfFirstRowsToDelete = len(local_costmap_info[local_costmap_info['frame'] == 'None'])
#print(numOfFirstRowsToDelete)

# Izbaci unose iz local_costmap_info
local_costmap_info.drop(index=local_costmap_info.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(local_costmap_info)

# Izbaci unose iz odom
odom.drop(index=odom.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(odom)

# Izbaci unose iz tf_odom_map
tf_odom_map.drop(index=tf_odom_map.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(tf_odom_map)

# Izbaci unose iz tf_map_odom
tf_map_odom.drop(index=tf_map_odom.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(tf_map_odom)

# Izbaci unose iz amcl_pose
amcl_pose.drop(index=amcl_pose.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(amcl_pose)

# Izbaci unose iz cmd_vel
cmd_vel.drop(index=cmd_vel.index[:numOfFirstRowsToDelete], axis=0, inplace=True)
#print(cmd_vel)


#Izbacivanje iz planova i footprinta zasad nije implementirano, jer nakon izbacivanja redova iz datafejmova indeksi zadrzavaju svoje vrijednosti,
# tako da se dalje mogu plan i footprint instance na isti nacin indeksirati


# Dataset creation

X_train = []
X_test = []

# Ako ce se koristiti LIME tabular
# ' cmd_vel_ang_z' - nekad u buducnosti ispraviti ovaj razmak - izbrisati ga
if explanationMode == 'tabular':
    from lime_explainer import DatasetCreator

    # Biranje ulaza za explanation algorithm
    X = odom.iloc[:,6:8] # ulaz za objasnjenje su odometrijske brzine
    #print(X)


    if mode == 'regression':

        import numpy as np

        # regression
        # Biranje izlaza za explanation algorithm
        index_output_class = 1 # [0] - komandna linearna brzina, [1] - komandna ugaona brzina
        y = cmd_vel.iloc[:,index_output_class:index_output_class+1]
        #print(y)        
        output_class_name = y.columns.values[0] # 'cmd_vel_lin_x' - [0] or ' cmd_vel_ang_z' - [0]

    elif (mode == 'classification') & (one_hot_encoding == False):

        import numpy as np
        
        # classification        
        # left-right-straight logika
        conditions = [
            (cmd_vel[' cmd_vel_ang_z'] >= 0),
            (cmd_vel[' cmd_vel_ang_z'] < 0)
            ]

        values = ['left', 'right']

        cmd_vel['direction'] = np.select(conditions, values)
        
        # Biranje izlaza za explanation algorithm
        index_output_class = 2 # [2] - direction
        y = cmd_vel.iloc[:,index_output_class:index_output_class+1] # izlaz za objasnjavanje je direction
        #print(y)
        output_class_name = y.columns.values[0] # 'direction' - [0]

    elif (mode == 'classification') & (one_hot_encoding == True):

        import numpy as np
        
        # random forest classification - one-hot encoding        
        # left-right-straight logika
        conditions = [
            (cmd_vel[' cmd_vel_ang_z'] >= 0),
            (cmd_vel[' cmd_vel_ang_z'] < 0)
            ]

        # one-hot left-right kodiranje
        valuesLeft = [1.0, 0.0]

        cmd_vel['left'] = np.select(conditions, valuesLeft)

        valuesRight = [0.0, 1.0]

        cmd_vel['right'] = np.select(conditions, valuesRight)
        
        # Biranje izlaza za explanation algorithm
        index_output_class = 2 # 'left' & 'right'
        y = cmd_vel.iloc[:,index_output_class:index_output_class+2] # izlaz za objasnjavanje je direction, odnosno left, right and straight one-hot enkodirano
        #print(y)
        output_class_name = y.columns.values[0] # 'left' - [0] or 'right' - [1]

    
    import random
    #randomNum  = random.randint(0, 100)
    randomNum = 42
    
    slice_ratio = 0.01 # very small slice_ratio ensures that almost all data is put in X_train
    
    X_train, X_test, y_train, y_test = DatasetCreator.split_test_train(X, y, slice_ratio, randomNum) # Imena redova (indexi) i nakon mijesanja ostaju ocuvana - very good


'''  
    # Osiguranje da ovi podaci budu Datafrejmovi, ali nije potrebno, jer vec jesu. Nek stoji radi ispisa pri mogucem debuggingu
    import pandas as pd
    X_train = pd.DataFrame(X_train)
    print(X_train)
    y_train = pd.DataFrame(y_train)
    print(y_train)
    
    X_test = pd.DataFrame(X_test)
    print(X_test)
    y_test = pd.DataFrame(y_test)
    print(y_test)
'''


# Ako ce se koristiti LIME image or LIME tabular sa costmap ulazom
if (explanationMode == 'image') | (explanationMode == 'tabular_costmap'):
    import pandas as pd
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame() 
    y_test = pd.DataFrame()
    

# Biranje expID
# choose/generate expID - redni broj reda u X_test ili X_train iz kojeg se vadi index
if explanationMode == 'tabular':
    # odabir po zelji - deterministicki
    expID = 86

    # Dataset1:
    # Dataset2:
    # Dataset3:
    
    # random odabir
    #import random
    #expID = random.randint(0, X_train.shape[0]) # expID se trazi iz X_train

else:
    # odabir po zelji - deterministicki
    expID = 71

    # Dataset1:
    # Dataset2:
    # Dataset3: #48 #92
    
    # random odabir
    #import random
    #expID = random.randint(0, local_costmap_info.shape[0]) # expID se trazi iz local_costmap_info
    
    output_class_name = cmd_vel.columns.values[0] # [0] - 'cmd_vel_lin_x'  or [1] - ' cmd_vel_ang_z'

# Explanation
from lime_explainer import ExplainNavigation

expNav = ExplainNavigation.ExplainRobotNavigation(cmd_vel, odom, plan, teb_global_plan, teb_local_plan, current_goal, local_costmap_data, local_costmap_info, 
amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, X_train, X_test, mode, explanationMode, expID, num_samples, output_class_name, numOfFirstRowsToDelete, footprints)

expNav.explain_instance(expID)

expNav.testSegmentation()

