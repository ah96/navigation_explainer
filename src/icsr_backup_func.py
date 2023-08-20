def explain_textual_only_icsr(self):
        if self.red_object_countdown_textual_only > 0:
            self.red_object_countdown_textual_only -= 1
            return
        elif self.red_object_countdown_textual_only == 0:
            self.red_object_countdown_textual_only = -1
            self.red_object_value_textual_only = -1

                # TEST THE DEVIATION
        if len(self.global_plan_history) > 1:
            # test if there is deviation between current and previous
            deviation_between_global_plans_textual = False
            deviation_threshold = 10.0
            
            self.globalPlan_goalPose_indices_history_hold = copy.deepcopy(self.globalPlan_goalPose_indices_history[-2:])
            #self.global_plan_history_hold = copy.deepcopy(self.global_plan_history)
            self.global_plan_current_hold = copy.deepcopy(self.global_plan_history[-1])
            if len(self.global_plan_history) > 1:
                self.global_plan_previous_hold = copy.deepcopy(self.global_plan_history[-2])
            
                min_plan_length = min(len(self.global_plan_current_hold.poses), len(self.global_plan_previous_hold.poses))
                
                # calculate deivation
                global_dev = 0
                for i in range(0, min_plan_length):
                    dev_x = self.global_plan_current_hold.poses[i].pose.position.x - self.global_plan_previous_hold.poses[i].pose.position.x
                    dev_y = self.global_plan_current_hold.poses[i].pose.position.y - self.global_plan_previous_hold.poses[i].pose.position.y
                    local_dev = dev_x**2 + dev_y**2
                    global_dev += local_dev
                global_dev = math.sqrt(global_dev)
                #print('\nDEVIATION BETWEEN GLOBAL PLANS!!! = ', global_dev)
                #print('OBJECT_MOVED_ID = ', self.last_object_moved_ID)
            
                if global_dev > deviation_threshold:
                    #print('DEVIATION BETWEEN GLOBAL PLANS!!! = ', global_dev)
                    deviation_between_global_plans_textual = True

            # check if the last two global plans have the same goal pose
            same_goal_pose = False
            if len(self.globalPlan_goalPose_indices_history_hold) > 1:
                if self.globalPlan_goalPose_indices_history_hold[-1][1] == self.globalPlan_goalPose_indices_history_hold[-2][1]:
                    same_goal_pose = True

            # if deviation happened and some object was moved
            if deviation_between_global_plans_textual and same_goal_pose:
                #print('TESTIRA se moguca devijacija')
                if self.last_object_moved_ID > 0 and self.red_object_countdown == -1: #self.last_object_moved_ID in neighborhood_objects_IDs
                    # define the red object
                    self.red_object_value_textual_only = copy.deepcopy(self.last_object_moved_ID)
                    self.red_object_countdown_textual_only = 12
                    self.last_object_moved_ID = -1

                    obj_pos_x = self.ontology[self.red_object_value_textual_only-1][3]
                    obj_pos_y = self.ontology[self.red_object_value_textual_only-1][4]

                    d_x = obj_pos_x - self.robot_pose_map.position.x
                    d_y = obj_pos_y - self.robot_pose_map.position.y

                    angle = np.arctan2(d_y, d_x)
                    [angle_ref,pitch,roll] = quaternion_to_euler(self.robot_pose_map.orientation.x,self.robot_pose_map.orientation.y,self.robot_pose_map.orientation.z,self.robot_pose_map.orientation.w)
                    angle = angle - angle_ref
                    if angle >= PI:
                        angle -= 2*PI
                    elif angle < -PI:
                        angle += 2*PI
                    qsr_value = getIntrinsicQsrValue(angle)

                    self.text_exp = 'I am deviating because the ' + self.ontology[self.red_object_value_textual_only - 1][1] + ', which is to my ' + qsr_value + ', was moved.'
                    return    

        # define local explanation window around robot
        around_robot_size_x = 1.5
        around_robot_size_y = 1.5
        if self.extrovert == False:
            around_robot_size_x = 2.5
            around_robot_size_y = 2.5

        # find the objects/obstacles in the robot's local neighbourhood
        robot_pose = self.robot_pose_map
        x_min = robot_pose.position.x - around_robot_size_x
        y_min = robot_pose.position.y - around_robot_size_y
        x_max = robot_pose.position.x + around_robot_size_x
        y_max = robot_pose.position.y + around_robot_size_y
        #print('(x_min,x_max,y_min,y_max) = ', (x_min,x_max,y_min,y_max))

        x_min_pixel = int((x_min - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)        
        x_max_pixel = int((x_max - self.global_semantic_map_origin_x) / self.global_semantic_map_resolution)
        y_min_pixel = int((y_min - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        y_max_pixel = int((y_max - self.global_semantic_map_origin_y) / self.global_semantic_map_resolution)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        explanation_size_y = self.global_semantic_map_size[0]
        explanation_size_x = self.global_semantic_map_size[1]
        #print('(self.explanation_size_x,self.explanation_size_y)',(self.explanation_size_y,self.explanation_size_x))

        x_min_pixel = max(0, x_min_pixel)
        x_max_pixel = min(explanation_size_x - 1, x_max_pixel)
        y_min_pixel = max(0, y_min_pixel)
        y_max_pixel = min(explanation_size_y - 1, y_max_pixel)
        #print('(x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel) = ', (x_min_pixel,x_max_pixel,y_min_pixel,y_max_pixel))
        
        global_semantic_map_complete_copy = copy.deepcopy(self.global_semantic_map_complete)
        neighborhood_objects_IDs = np.unique(global_semantic_map_complete_copy[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel])
        if 0 in neighborhood_objects_IDs:
            neighborhood_objects_IDs = neighborhood_objects_IDs[1:]
        neighborhood_objects_IDs = [int(item) for item in neighborhood_objects_IDs]
        #print('neighborhood_objects_IDs =', neighborhood_objects_IDs)

        neighborhood_objects_distances = []
        neighborhood_objects_spatials = []
        neighborhood_objects_names = []
        for ID in neighborhood_objects_IDs:
            obj_pos_x = self.ontology[ID-1][3]
            obj_pos_y = self.ontology[ID-1][4]

            d_x = obj_pos_x - robot_pose.position.x
            d_y = obj_pos_y - robot_pose.position.y

            dist = math.sqrt(d_x**2 + d_y**2)
            neighborhood_objects_distances.append(dist)

            angle = np.arctan2(d_y, d_x)
            [angle_ref,pitch,roll] = quaternion_to_euler(robot_pose.orientation.x,robot_pose.orientation.y,robot_pose.orientation.z,robot_pose.orientation.w)
            angle = angle - angle_ref
            if angle >= PI:
                angle -= 2*PI
            elif angle < -PI:
                angle += 2*PI
            qsr_value = getIntrinsicQsrValue(angle)
            neighborhood_objects_spatials.append(qsr_value)
            neighborhood_objects_names.append(self.ontology[ID-1][1])
            #print('tiago passes ' + qsr_value + ' of the ' + self.ontology[ID-1][1])
        #print(len(neighborhood_objects_spatials))
            
        # FORM THE TEXTUAL EXPLANATION
        if len(neighborhood_objects_names) > 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])

            if len(left_objects) != 0 and len(right_objects) != 0:
                self.text_exp = 'I am passing by '

                if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left'
                elif len(left_objects) == 2:
                    self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left'
                elif len(left_objects) == 1:
                    self.text_exp += left_objects[0] + ' to my left'

                self.text_exp += ' and '

                if len(right_objects) > 2:
                    for i in range(0, len(right_objects) - 2):    
                        self.text_exp += right_objects[i] + ', '
                    i += 1
                    self.text_exp += right_objects[i] + ' and '
                    i += 1
                    self.text_exp += right_objects[i] + ' to my right.'
                elif len(right_objects) == 2:
                    self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                elif len(right_objects) == 1:
                    self.text_exp += right_objects[0] + ' to my right.'
            else:
                self.text_exp = 'I am passing by '

                if len(left_objects) != 0:
                    if len(left_objects) > 2:
                        for i in range(0, len(left_objects) - 2):    
                            self.text_exp += left_objects[i] + ', '
                        i += 1
                        self.text_exp += left_objects[i] + ' and '
                        i += 1
                        self.text_exp += left_objects[i] + ' to my left.'
                    elif len(left_objects) == 2:
                        self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
                    elif len(left_objects) == 1:
                        self.text_exp += left_objects[0] + ' to my left.'

                if len(right_objects) != 0:
                    if len(right_objects) > 2:
                        for i in range(0, len(right_objects) - 2):    
                            self.text_exp += right_objects[i] + ', '
                        i += 1
                        self.text_exp += right_objects[i] + ' and '
                        i += 1
                        self.text_exp += right_objects[i] + ' to my right.'
                    elif len(right_objects) == 2:
                        self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
                    elif len(right_objects) == 1:
                        self.text_exp += right_objects[0] + ' to my right.'
        
        elif len(neighborhood_objects_names) == 2:
            left_objects = []
            right_objects = []
            for i in range(0, len(neighborhood_objects_names)):
                if neighborhood_objects_spatials[i] == 'left':
                    left_objects.append(neighborhood_objects_names[i])
                else:
                    right_objects.append(neighborhood_objects_names[i])
            
            if len(left_objects) == 1 and len(right_objects) == 1:
                self.text_exp = 'I am passing by '
                self.text_exp += left_objects[0] + ' to my left and ' + right_objects[1] + ' to my right.'
            elif len(left_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += right_objects[0] + ' and ' + right_objects[1] + ' to my right.'
            elif len(right_objects) == 0:
                self.text_exp = 'I am passing by ' 
                self.text_exp += left_objects[0] + ' and ' + left_objects[1] + ' to my left.'
        
        elif len(neighborhood_objects_names) == 1:
            if neighborhood_objects_spatials[0] == 'left' or neighborhood_objects_spatials[0] == 'right':
                self.text_exp = 'I am passing by '    
                self.text_exp += neighborhood_objects_names[0] + ' to my ' + neighborhood_objects_spatials[0] + '.'
        #print(self.text_exp)