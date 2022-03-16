from . import anchor_base
import numpy as np
import sklearn
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import itertools
        

class AnchorImage(object):
    """bla"""
    def __init__(self, distribution_path=None,
                 transform_img_fn=None, n=1000, dummys=None, white=None,
                 segmentation_fn=None):
        """"""
        self.hide = True
        self.white = white
        self.segmentation = self.sm_only_obstacles                                           
        
        if dummys is not None:
            self.hide = False
            self.dummys = dummys
        elif distribution_path:
            self.hide = False
            import os
            import skimage

            if not transform_img_fn:
                def transform_img(path):
                    img = skimage.io.imread(path)
                    short_egde = min(img.shape[:2])
                    yy = int((img.shape[0] - short_egde) / 2)
                    xx = int((img.shape[1] - short_egde) / 2)
                    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
                    return skimage.transform.resize(crop_img, (224, 224))

                def transform_imgs(paths):
                    out = []
                    for i, path in enumerate(paths):
                        if i % 100 == 0:
                            print(i)
                        out.append(transform_img(path))
                    return out
                transform_img_fn = transform_imgs
            all_files = os.listdir(distribution_path)
            all_files = np.random.choice(
                all_files, size=min(n, len(all_files)), replace=False)
            paths = [os.path.join(distribution_path, f) for f in all_files]
            self.dummys = transform_img_fn(paths)

    def get_sample_fn(self, image, classifier_fn, costmap_info, map_info, tf_odom_map, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, 
                    plan_x_list, plan_y_list, lime=False):

        # save original 2D costmap
        image_orig = copy.deepcopy(image[:,:,0])
        
        # find segments
        segments = self.segmentation(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
        
        # create perturbations
        features = list(np.unique(segments))
        n_features = len(features) - 1 #=9-1
        rows = np.array([1]*n_features)
        # create costmap perturbations
        imgs = []    
        temp = copy.deepcopy(image_orig)
        zeros = np.where(rows == 0)[0]
        for z in zeros:
            temp[segments == z + 1] = 0.0
        imgs.append(temp)
        output = classifier_fn(np.array(imgs)[0,:,:])
        print('\nclassifier_fn output = ', output)
        true_label = output[0][0]
        print('\ntrue_label = ', true_label)
        #true_label = np.argmax(output[0])
        #true_label = np.argmax(classifier_fn(np.expand_dims(image, 0))[0])

        def sample_fn(present, num_samples, compute_labels=True):
            print('\nsample_fn started!')
                            
            # this if is validated to True only at the beginning when coverage is calculated
            if not compute_labels:
                # num_samples = 2**n_features
                if num_samples == 2**n_features:
                    lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
                    data = np.array(lst).reshape((num_samples, n_features))
                    data[0, :] = 1
                    data[-1, :] = 0 # only if I use my perturbation
                    return [], data, []
                else:
                    data = np.random.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
                    return [], data, []    
            elif compute_labels:
                imgs = []
                if num_samples == 2**n_features:
                    lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
                    rows = np.array(lst).reshape((num_samples, n_features))
                    rows[0, :] = 1
                    rows[-1, :] = 0 # only if I use my perturbation
                    for row in rows:
                        temp = copy.deepcopy(image_orig)
                        zeros = np.where(row == 0)[0]
                        for z in zeros:
                            temp[segments == z + 1] = 0.0
                        imgs.append(temp)
                    preds = classifier_fn(np.array(imgs))
                    return rows, rows, preds
                else:
                    rows = np.random.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
                    rows[:, present] = 1
                    for row in rows:
                        temp = copy.deepcopy(image_orig)
                        zeros = np.where(row == 0)[0]
                        for z in zeros:
                            temp[segments == z + 1] = 0.0
                        imgs.append(temp)
                    preds = classifier_fn(np.array(imgs))
                    return rows, rows, preds
            
        def lime_sample_fn(num_samples, batch_size=50):
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            data = np.zeros((num_samples, n_features))
            labels = []
            imgs = []
            sizes = np.random.randint(0, n_features, num_samples)
            all_features = range(n_features)
            # for row in data:
            for i, size in enumerate(sizes):
                row = np.ones(n_features)
                chosen = np.random.choice(all_features, size)
                # print chosen, size,
                row[chosen] = 0
                data[i] = row
                # print row
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
            # return imgs, np.array(labels)
            return data, np.array(labels)

        if lime:
            return segments, lime_sample_fn

        def sample_fn_dummy(present, num_samples, compute_labels=True):
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []
            data = np.zeros((num_samples, n_features))
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            if len(present) < 5:
                data = np.random.choice(
                    [0, 1], num_samples * n_features, p=[.8, .2]).reshape(
                        (num_samples, n_features))
            data[:, present] = 1
            chosen = np.random.choice(range(len(self.dummys)), data.shape[0],
                                      replace=True)
            labels = []
            imgs = []
            for d, r in zip(data, chosen):
                temp = copy.deepcopy(image)
                zeros = np.where(d == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                if self.white:
                    temp[mask] = 1
                else:
                    temp[mask] = self.dummys[r][mask]
                imgs.append(temp)
                # pred = np.argmax(classifier_fn(temp.to_nn())[0])
                # print self.class_names[pred]
                # labels.append(int(pred == true_label))
            # import time
            # a = time.time()
            imgs = np.array(imgs)
            preds = classifier_fn(imgs)
            # print (time.time() - a) / preds.shape[0]
            imgs = []
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            raw_data = np.hstack((data, chosen.reshape(-1, 1)))
            return raw_data, data, np.array(labels)

        sample = sample_fn if self.hide else sample_fn_dummy
        return segments, sample

    def explain_instance(self, image, classifier_fn, 
                        costmap_info, map_info, tf_odom_map, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list,
                        threshold=0.95, delta=0.1, tau=0.15, batch_size=100, **kwargs):
        
        # classifier_fn is a predict_proba
        segments, sample = self.get_sample_fn(image, classifier_fn, costmap_info, map_info, tf_odom_map, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
                
        exp, best_tuples = anchor_base.AnchorBaseBeam.anchor_beam(sample, delta=delta, epsilon=tau, batch_size=batch_size, desired_confidence=threshold, **kwargs)
        print('\nBEFORE_HOEFFDING_EXP = ', exp)
        #print('\ntype(exp) = ', type(exp))
        
        return segments, self.get_exp_from_hoeffding(image, exp), best_tuples

    def sm_only_obstacles(self, image, img_rgb, x_odom, y_odom, devDistance_x, sign_x, devDistance_y, sign_y, devDistance, plan_x_list, plan_y_list):
        # import needed libraries
        from skimage.segmentation import slic
        import matplotlib.pyplot as plt
        import numpy as np
        import copy
        
        print('\nsm_only_obstacles started')
        
        # show original image
        img = copy.deepcopy(image)

        sm_only_obstacles_start = time.time()

        # Find segments_2
        segments_slic = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                            start_label=1, mask=None)

        #'''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_slic.astype('float64'), aspect='auto')
        fig.savefig('segments_slic.png', transparent=False)
        fig.clf()
        #'''

        segments = np.zeros(img.shape, np.uint8)

        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)
        #print('\nabs(k) = ', abs(k)) 

        # make one free space segment
        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        num_of_wanted_obstacle_segments = 8

        num_of_existing_obstacle_segments = 0
        # add existing obstacle segments        
        for i in np.unique(segments_slic):
            temp = img[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            if np.all(img[segments_slic == i] == 99) or count_of_99_s > 0.9 * temp.shape[0]:
                segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_existing_obstacle_segments += 1

        #print('\nnum_of_existing_obstacle_segments: ', num_of_existing_obstacle_segments)

        num_of_wanted_obstacle_segments = max(num_of_wanted_obstacle_segments, num_of_existing_obstacle_segments)        

        if num_of_wanted_obstacle_segments > num_of_existing_obstacle_segments > 0:
            # divide segment obstacles    
            current_segs_labels = np.unique(segments)[1:]        

            #print('\nnumber of wanted segments: ', num_of_wanted_obstacle_segments)
            #print('number of current segments: ', num_of_existing_obstacle_segments)

            current_segs_sizes = []

            if num_of_existing_obstacle_segments < num_of_wanted_obstacle_segments:
                for i in range(0, num_of_existing_obstacle_segments):
                    #print(len(segments[segments == seg_labels[i]]))
                    current_segs_sizes.append(len(segments[segments == current_segs_labels[i]]))

            #print('\nsizes of original segments: ', current_segs_sizes)
            #print('labels of original segments: ', current_segs_labels)
            sorted_segs_labels = [x for _, x in sorted(zip(current_segs_sizes, current_segs_labels))]
            sorted_segs_labels.reverse()
            #print('labels of sorted segments: ', sorted_segs_labels)

            num_segs_missing = num_of_wanted_obstacle_segments - num_of_existing_obstacle_segments
            #print('\nnumber of segments missing: ', num_segs_missing)

            # if a number of missing segments is smaller or equal than the number of existing segments
            # only divide existing segments into 2
            if num_segs_missing <= num_of_existing_obstacle_segments:
                label_current = len(sorted_segs_labels) + 1
                # for loop from zero until the number of missing segments   
                for i in range(0, num_segs_missing):
                    temp = segments[segments == sorted_segs_labels[i]]
                    #print('temp = ', temp)

                    # check obstacle shape
                    w_min = 161
                    w_max = -1
                    h_min = 161
                    h_max = -1
                    for j in range(0, segments.shape[0]):
                        for q in range(0, segments.shape[1]):
                            if segments[j, q] == sorted_segs_labels[i]:
                                if j > h_max:
                                    h_max = j
                                if j < h_min:
                                    h_min = j
                                if q > w_max:
                                    w_max = q
                                if q < w_min:
                                    w_min = q           

                    #print('\n(h_min, h_max): ', (h_min, h_max))
                    #print('(w_min, w_max): ', (w_min, w_max))

                    height = h_max - h_min + 1
                    #print('\nheight', height)
                    width = w_max - w_min + 1
                    #print('width', width)
           
                    # if upright
                    if abs(k) >= 1:
                        # divide by height
                        if height > width:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == sorted_segs_labels[i]] = temp
                        
                        # divide by width    
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1

                            '''
                            # second method            
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            finished = False
                            for q in range(0, segments.shape[1]):
                                if finished == True:
                                    break
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        segments[j, q] = label_current
                                        counter += 1
                                        if counter == int(num_of_pixels / 2 + 0.5):
                                            #label_current += 1
                                            finished = True
                                            break
                            '''

                    # if to the side
                    elif abs(k) < 1:
                        # divide by height
                        if width > height:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == sorted_segs_labels[i]] = temp
                        
                        # divide by width   
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1
                    
                    label_current += 1

            # if a number of missing segments is greater than the number of existing segments
            else:
                num_of_new_seg_per_old_seg = int(num_segs_missing / num_of_existing_obstacle_segments)
                #print('num_of_new_seg_per_old_seg = ', num_of_new_seg_per_old_seg)
                
                # if a number of new segment per old segments is integer and same for all old segments
                # divide segments in num_of_new_seg_per_old_seg new segments
                if num_of_new_seg_per_old_seg == num_segs_missing / num_of_existing_obstacle_segments:
                    #print('\nnumber of new segments per existing segment: ', num_of_new_seg_per_old_seg)

                    label_current = len(sorted_segs_labels) + 1
                    
                    for i in range(0, num_of_existing_obstacle_segments):
                        temp = segments[segments == sorted_segs_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            # divide by height
                            if height > width:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break
                        # if to the side
                        else:
                            if width > height:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                # if a number of new segment per old segments is integer and same for all old segments
                else:
                    #print('NON-CIO BROJ')
                    whole_part = int(num_segs_missing / num_of_existing_obstacle_segments)
                    #print('whole part = ', whole_part)
                    rest = num_segs_missing % num_of_existing_obstacle_segments
                    #print('rest = ', rest)

                    put_rest_to_biggest_segment = True

                    num_of_new_seg_per_old_seg_list = [whole_part] * num_of_existing_obstacle_segments

                    if put_rest_to_biggest_segment == False:
                        for i in range(0, rest):
                            num_of_new_seg_per_old_seg_list[i] += 1
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)    
                    else:
                        num_of_new_seg_per_old_seg_list[0] += rest
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)

                    #print('num_of_new_seg_per_old_seg_list = ', num_of_new_seg_per_old_seg_list)

                    label_current = len(sorted_segs_labels) + 1
                    #print('\nlabel_current = ', label_current)
                    
                    for i in range(0, num_of_existing_obstacle_segments):
                        temp = segments[segments == sorted_segs_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height > width:
                                #print('height > width')
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                #print('height <= width')
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                #print('step = ', step)
                                label_original = temp[0]
                                #print('label_original = ', label_original)
                                num_of_pixels = len(temp)
                                #print('num_of_pixels = ', num_of_pixels)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                        # if to the side
                        else:
                            #print('SIDE')
                            if width > height:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

        #'''
        fig = plt.figure(frameon=False)
        w = 1.6 * 3
        h = 1.6 * 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments_final.png', transparent=False)
        fig.clf()
        #'''

        # fix labels of segments
        seg_labels = np.unique(segments)
        for i in range(0, len(seg_labels)):
            label = seg_labels[i]
            if label != i:
                segments[segments == label] = i

        #print('\nnp.unique(segments): ', np.unique(segments))
        #print('\nlen(np.unique(segments)): ', len(np.unique(segments)))

        sm_only_obstacles_end = time.time()
        sm_only_obstacles_time = sm_only_obstacles_end - sm_only_obstacles_start
        print('\nsm_only_obstacles runtime: ', sm_only_obstacles_time)

        print('\nsm_only_obstacles ended')

        if len(seg_labels) != num_of_wanted_obstacle_segments + 1:
            print('ERROR!!!')
            return segments_slic

        return segments

    def get_exp_from_hoeffding(self, image, hoeffding_exp):
        """
        bla
        """
        ret = []

        features = hoeffding_exp['feature']
        means = hoeffding_exp['mean']
        if 'negatives' not in hoeffding_exp:
            negatives_ = [np.array([]) for x in features]
        else:
            negatives_ = hoeffding_exp['negatives']
        for f, mean, negatives in zip(features, means, negatives_):
            train_support = 0
            name = ''
            if negatives.shape[0] > 0:
                unique_negatives = np.vstack({
                    tuple(row) for row in negatives})
                distances = sklearn.metrics.pairwise_distances(
                    np.ones((1, negatives.shape[1])),
                    unique_negatives)
                negative_arrays = (unique_negatives
                                   [np.argsort(distances)[0][:4]])
                negatives = []
                for n in negative_arrays:
                    negatives.append(n)
            else:
                negatives = []
            ret.append((f, name, mean, negatives, train_support))
        return ret