"""
Functions for explaining classifiers that use Image data.
"""
from cProfile import label
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm

import time

from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negatively and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """

        #print('get_image_and_mask starting')

        #'''
        # testing
        import matplotlib.pyplot as plt
        #print('self.local_exp: ', self.local_exp)
        seg_unique = np.unique(self.segments)
        #print('self.segments_unique: ', seg_unique)
        #'''

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]

        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            #temp = self.image.copy()
            temp = np.zeros(self.image.shape)
            #temp[segments == 1] = 0.0
        if positive_only:
            fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
            #print('fs: ', fs)
        if negative_only:
            fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only == True and negative_only == False:
            for f in fs:
                if image[segments == f].all() == 0.0:
                    temp[segments == f, 1] = np.max(image) #image[segments == f].copy()
                    if hide_rest == False:
                        temp[segments == f, 0] = 0.0
                        temp[segments == f, 2] = 0.0
                else:
                    temp[segments == f, 1] = np.max(image)  # image[segments == f].copy()
                    temp[segments == f, 2] = np.max(image)  # image[segments == f].copy()
                    if hide_rest == False:
                        temp[segments == f, 0] = 0.0
                        # temp[segments == f, 2] = 0.0
                mask[segments == f] = 1
            #print('get_image_and_mask ending')
            return temp, mask, exp
        if positive_only == False and negative_only == True:
            for f in fs:
                if image[segments == f].all() == 0.0:
                    temp[segments == f, 0] = np.max(image) #image[segments == f].copy()
                    if hide_rest == False:
                        temp[segments == f, 1] = 0.0
                        temp[segments == f, 2] = 0.0
                else:
                    temp[segments == f, 0] = np.max(image)  # image[segments == f].copy()
                    temp[segments == f, 2] = np.max(image)  # image[segments == f].copy()
                    if hide_rest == False:
                        temp[segments == f, 1] = 0.0
                        #temp[segments == f, 2] = 0.0
                mask[segments == f] = 1
            #print('get_image_and_mask ending')
            return temp, mask, exp
        else:
            counter_local = 0
            import pandas as pd

            w_sum = 0.0
            w_s = []
            for f, w in exp[:num_features]:
                w_sum += abs(w)
                w_s.append(abs(w))
            max_w = max(w_s)
            if max_w == 0:
                max_w = 1    

            #pd.DataFrame(segments).to_csv('segments_lime_image.csv', index=False)
            for f, w in exp[:num_features]:
                #print('(f, w): ', (f, w))
                #print('w_sum: ', w_sum)

                #if f == 0:
                #    print(image[segments == f])

                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                #print('(f, w, c): ', (f, w, c))
                mask[segments == f] = 120 + 10*(f+1) #f+1 #-1 if w < 0 else 1
                #temp[segments == f] = image[segments == f].copy()
                val_low = 0.0
                val_high = 255.0

                gray_shade = 180

                color_free_space = False

                # if free space
                if image[segments == f].all() == 0.0:
                    # if positive weight
                    if c == 1:
                        if color_free_space == True:
                            temp[segments == f, 0] = 0 #180.0  # c is channel, RGB - 012
                            temp[segments == f, 1] = val_low + (val_high - val_low) * (abs(w) / max_w) #180.0 #val_low + (val_high - val_low) * (abs(w) / max_w) #float(np.max(image)) * w / w_sum # c is channel, RGB - 012
                            temp[segments == f, 2] = 0 #180.0
                        else:
                            temp[segments == f, 0] = gray_shade  # c is channel, RGB - 012
                            temp[segments == f, 1] = gray_shade #val_low + (val_high - val_low) * (abs(w) / max_w) #float(np.max(image)) * w / w_sum # c is channel, RGB - 012
                            temp[segments == f, 2] = gray_shade
                    # if negative weight
                    else:
                        if color_free_space == True:
                            temp[segments == f, 0] = val_low + (val_high - val_low) * (abs(w) / max_w) #180.0 #val_low + (val_high - val_low) * (abs(w) / max_w) #float(np.max(image)) * w / w_sum  # c is channel, RGB - 012
                            temp[segments == f, 1] = 0 #180.0  # c is channel, RGB - 012
                            temp[segments == f, 2] = 0 #180.0
                        else:
                            temp[segments == f, 0] = gray_shade  # c is channel, RGB - 012
                            temp[segments == f, 1] = gray_shade #val_low + (val_high - val_low) * (abs(w) / max_w) #float(np.max(image)) * w / w_sum # c is channel, RGB - 012
                            temp[segments == f, 2] = gray_shade
                # if obstacle
                else:
                    # if positive weight
                    if c == 1:
                        temp[segments == f, 0] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 1] = val_low + (val_high - val_low) * (abs(w) / max_w)  #120 + 100 * (abs(w) / max_w) #float(np.max(image)) * w / w_sum #float(np.max(image)) * w / w_sum  # c is channel, RGB - 012
                        temp[segments == f, 2] = 0.0 #50 + 170 * (abs(w) / max_w) #float(np.max(image)) * w / w_sum #1.0
                    # if negative weight
                    else:
                        temp[segments == f, 0] = val_low + (val_high - val_low) * (abs(w) / max_w) #float(np.max(image)) * w / w_sum  # c is channel, RGB - 012
                        temp[segments == f, 1] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 2] = 0.0 #50 + 170 * (abs(w) / max_w) #float(np.max(image)) * w / w_sum #1.0

                counter_local += 1

            #temp = temp.astype(np.uint8)
            #mask = mask.astype(np.uint8)    
            
            '''
            pd.DataFrame(temp[:,:,0]).to_csv('R.csv')
            pd.DataFrame(temp[:,:,1]).to_csv('G.csv')
            pd.DataFrame(temp[:,:,2]).to_csv('B.csv')

            from PIL import Image
            im = Image.fromarray(temp.astype(np.uint8))
            im.save("temp.png")

            im = Image.fromarray(mask.astype(np.uint8))
            im.save("mask.png")

            import matplotlib.pyplot as plt
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(temp.astype(np.uint8), aspect='auto')
            fig.savefig('temp_plt.png')
            fig.clf()
            '''
            
            #print('get_image_and_mask ending')
            return temp, mask, exp


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def sm1(self, img_rgb):
        print('\nsm1 starts')

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = img_rgb[:, :, 0]

        start = time.time()

        # segments_1 - good obstacles
        # Find segments_1
        #'''
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        #'''
        
        '''
        segments_1 = slic(img_rgb, n_segments=8, compactness=0.1, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''

        print('np.unique(segments_1): ', np.unique(segments_1))

        #'''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1.astype('float64'), aspect='auto')
        fig.savefig('segments1.png', transparent=False)
        fig.clf()
        #'''

        early_return = False
        if early_return == True:
            end = time.time()
            print('\nsm1 runtime: ', end-start)
            print('\nsm1 ends')
            return segments_1 - 1                  

        # Find segments_2
        #'''
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        #'''

        '''                  
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''

        #'''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        #'''

        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)

        # make obstacles on segments_2 nice - not needed
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 99:
                   segments_2[i, j] = segments_1[i, j] + segments_unique_2.shape[0]
        
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k # k+1 must be in order for regionprops() function to work correctly # k mora da bi perturbacija radila za obstacles
                
        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1.astype('float64'), aspect='auto')
        fig.savefig('segments12.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments_1)
        '''
        
        end = time.time()
        print('\nsm1 runtime: ', end-start)

        print('\nsm1 ends')

        return segments_1

    def sm2(self, image, img_rgb, x_odom, y_odom):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=4, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                            start_label=1, mask=None)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        '''

        segments = np.zeros(img.shape, np.uint8)

        ctr = 0
        segments[0:(y_odom + 25), 0:(x_odom - 25)] = ctr
        ctr = ctr + 1
        segments[0:(y_odom + 25), (x_odom - 25):(x_odom + 25)] = ctr
        ctr = ctr + 1
        segments[0:(y_odom + 25), (x_odom + 25):160] = ctr
        ctr = ctr + 1
        segments[(y_odom + 25):160, :] = ctr
        ctr = ctr + 1
        #segments[65:95, 65:95] = ctr
        #ctr = ctr + 1

        #print('np.unique(segments_2): ', np.unique(segments_2))
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        end = time.time()

        print("\sm2 runtime: ", end - start)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments12.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments)
        '''

        return segments

    def sm3(self, image, img_rgb, x_odom, y_odom, devDistance, sign):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=4, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                            start_label=1, mask=None)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        '''

        segments = np.zeros(img.shape, np.uint8)

        devDistance = int(devDistance)

        footprint_radius = 13
        add = 15

        if sign < 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - devDistance - add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance - add):(x_odom + footprint_radius + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1
        elif sign > 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - add):(x_odom + footprint_radius + devDistance + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + devDistance + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1


        #print('np.unique(segments_2): ', np.unique(segments_2))
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        end = time.time()

        print("\sm3 runtime: ", end - start)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments12.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments)
        '''

        return segments

    def sm4(self, image, img_rgb, x_odom, y_odom, devDistance, sign):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=4, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                            start_label=1, mask=None)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        '''

        segments = np.zeros(img.shape, np.uint8)

        devDistance = int(devDistance)

        footprint_radius = 13
        add = 15

        if sign < 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - devDistance - add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance - add):(x_odom - footprint_radius - devDistance)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance):(x_odom + footprint_radius + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1
        elif sign > 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - add):(x_odom + footprint_radius + devDistance + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + devDistance + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1


        #print('np.unique(segments_2): ', np.unique(segments_2))
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        end = time.time()

        print("\sm4 runtime: ", end - start)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments12.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments)
        '''

        return segments

    def sm5(self, image, img_rgb, x_odom, y_odom, devDistance, sign):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=4, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                            start_label=1, mask=None)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        '''

        segments = np.zeros(img.shape, np.uint8)

        devDistance = int(devDistance)

        footprint_radius = 13
        add = 15

        if sign < 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - devDistance)] = ctr
            ctr = ctr + 1
            #segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance - add):(x_odom - footprint_radius - devDistance)] = ctr
            #ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance):(x_odom + footprint_radius + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1
        elif sign > 0:
            ctr = 0
            segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom - footprint_radius - add):(x_odom + footprint_radius + devDistance + add)] = ctr
            ctr = ctr + 1
            segments[0:(y_odom + 25), (x_odom + footprint_radius + devDistance + add):160] = ctr
            ctr = ctr + 1
            segments[(y_odom + 25):160, :] = ctr
            ctr = ctr + 1


        #print('np.unique(segments_2): ', np.unique(segments_2))
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        pd.DataFrame(segments).to_csv("segments.csv")        

        end = time.time()

        print("\sm5 runtime: ", end - start)

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments12.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments)
        '''

        return segments

    def sm6(self, image, img_rgb, x_odom, y_odom, devDistance_x, sign_x, devDistance_y, sign_y, devDistance, plan_x_list, plan_y_list):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=4, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                            start_label=1, mask=None)

        #'''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments2.png', transparent=False)
        fig.clf()
        #'''

        segments = np.zeros(img.shape, np.uint8)

        devDistance_x = int(devDistance_x)
        devDistance_x = abs(devDistance_x)
        devDistance_y = int(devDistance_y)
        devDistance_y = abs(devDistance_y)
        devDistance = int(devDistance)

        footprint_radius = 13
        add = 15

        print('plan_x_list[-1]: ', plan_x_list[-1])
        print('plan_y_list[-1]: ', plan_y_list[-1])
        print('x_odom: ', x_odom)
        print('y_odom: ', y_odom)

        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)
        print('k = ', k)
        print('plan_x_list: ', plan_x_list)
        print('plan_y_list: ', plan_y_list)

        delta_y = plan_y_list[-1] - y_odom
        delta_x = plan_x_list[-1] - x_odom  

        if abs(k) >= 1:
            print('VODORAVNO!')
            if delta_y <= 0:
                print('GORE!')
                if sign_x <= 0:
                    '''
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
                    '''
                    ctr = 0
                    segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - devDistance_x - add)] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom + 25), (x_odom - footprint_radius - devDistance_x - add):(x_odom + footprint_radius + add)] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom + 25), (x_odom + footprint_radius + add):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + 25):160, :] = ctr
                    ctr = ctr + 1
                elif sign_x > 0:
                    '''
                    ctr = 0
                    segments[:, :] = ctr
                    '''
                    #'''
                    ctr = 0
                    segments[0:(y_odom + 25), 0:(x_odom - footprint_radius - add)] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom + 25), (x_odom - footprint_radius - add):(x_odom + footprint_radius + devDistance_x + add)] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom + 25), (x_odom + footprint_radius + devDistance_x + add):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + 25):160, :] = ctr
                    ctr = ctr + 1
                    #'''
            else:
                print('DOLJE!')
                if sign_x <= 0:
                    ctr = 0
                    segments[(y_odom - 25):160, 0:(x_odom - footprint_radius - devDistance_x - add)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - 25):160, (x_odom - footprint_radius - devDistance_x - add):(x_odom + footprint_radius + add)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - 25):160, (x_odom + footprint_radius + add):160] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom - 25), :] = ctr
                    ctr = ctr + 1
                elif sign_x > 0:
                    ctr = 0
                    segments[(y_odom - 25):160, 0:(x_odom - footprint_radius - add)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - 25):160, (x_odom - footprint_radius - add):(x_odom + footprint_radius + devDistance_x + add)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - 25):160, (x_odom + footprint_radius + devDistance_x + add):160] = ctr
                    ctr = ctr + 1
                    segments[0:(y_odom - 25), :] = ctr
                    ctr = ctr + 1    
        else:    
            print('HORIZONTALNO!')
            if delta_x <= 0:
                print('LIJEVO!')
                if sign_y <= 0:
                    ctr = 0
                    segments[0:(y_odom - footprint_radius - devDistance_y - add), 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - footprint_radius - devDistance_y - add):(y_odom + footprint_radius + add), 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + footprint_radius + add):160, 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[:, (x_odom + 25):160] = ctr
                    ctr = ctr + 1
                elif sign_y > 0:
                    ctr = 0
                    segments[0:(y_odom - footprint_radius - add), 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - footprint_radius - add):(y_odom + footprint_radius + devDistance_y + add), 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + footprint_radius + devDistance_y + add):160, 0:(x_odom + 25)] = ctr
                    ctr = ctr + 1
                    segments[:, (x_odom + 25):160] = ctr
                    ctr = ctr + 1
            else:
                print('DESNO!')
                if sign_y <= 0:
                    ctr = 0
                    segments[0:(y_odom - footprint_radius - devDistance_y - add), (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - footprint_radius - devDistance_y - add):(y_odom + footprint_radius + add), (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + footprint_radius + add):160, (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[:, 0:(x_odom - 25)] = ctr
                    ctr = ctr + 1
                elif sign_y > 0:
                    ctr = 0
                    segments[0:(y_odom - footprint_radius - add), (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom - footprint_radius - add):(y_odom + footprint_radius + devDistance_y + add), (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[(y_odom + footprint_radius + devDistance_y + add):160, (x_odom - 25):160] = ctr
                    ctr = ctr + 1
                    segments[:, 0:(x_odom - 25)] = ctr
                    ctr = ctr + 1    

        '''
        fig = plt.figure(frameon=False)
        w = 1.6 #* 3
        h = 1.6 #* 3
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.scatter(x_, y_, c='r')
        fig.savefig('HELP.png', transparent=False)
        fig.clf()
        '''
                
        #print('np.unique(segments_2): ', np.unique(segments_2))
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        end = time.time()

        print("\sm3 runtime: ", end - start)

        #'''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segmentsFINAL.png', transparent=False)
        fig.clf()
        segments_unique = np.unique(segments)
        #'''

        return segments

    def sm7(self, image, img_rgb, x_odom, y_odom, devDistance_x, sign_x, devDistance_y, sign_y, devDistance, plan_x_list, plan_y_list):
        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        start = time.time()

        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
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
        ax.imshow(segments_2.astype('float64'), aspect='auto')
        fig.savefig('segments_slic.png', transparent=False)
        fig.clf()
        #'''

        segments = np.zeros(img.shape, np.uint8)

        devDistance_x = int(devDistance_x)
        devDistance_x = abs(devDistance_x)
        devDistance_y = int(devDistance_y)
        devDistance_y = abs(devDistance_y)
        devDistance = int(devDistance)

        footprint_radius = 13
        add = 15

        '''
        print('plan_x_list[-1]: ', plan_x_list[-1])
        print('plan_y_list[-1]: ', plan_y_list[-1])
        print('x_odom: ', x_odom)
        print('y_odom: ', y_odom)
        '''

        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)

        '''
        print('k = ', k)
        print('plan_x_list: ', plan_x_list)
        print('plan_y_list: ', plan_y_list)
        '''

        delta_y = plan_y_list[-1] - y_odom
        delta_x = plan_x_list[-1] - x_odom  

        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        '''
        if abs(k) >= 1:
            print('VODORAVNO!')
            if delta_y <= 0:
                print('GORE!')
                if sign_x <= 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1

                elif sign_x > 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
            else:
                print('DOLJE!')
                if sign_x <= 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
                elif sign_x > 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1    
        else:    
            print('HORIZONTALNO!')
            if delta_x <= 0:
                print('LIJEVO!')
                if sign_y <= 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
                elif sign_y > 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
            else:
                print('DESNO!')
                if sign_y <= 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1
                elif sign_y > 0:
                    ctr = 0
                    segments[:, :] = ctr
                    ctr = ctr + 1    
        '''
                
        for i in np.unique(segments_2):
            if np.all(img[segments_2 == i] == 99):
                #print('obstacle')
                segments[segments_2 == i] = ctr
                ctr = ctr + 1

        seg_labels = np.unique(segments)[1:]        

        num_of_seg = len(seg_labels)
        num_of_wanted_seg = 8

        seg_sizes = []

        if num_of_seg < num_of_wanted_seg:
            for i in range(0, num_of_seg):
                print(len(segments[segments == seg_labels[i]]))
                seg_sizes.append(len(segments[segments == seg_labels[i]]))

        print('segm_sizes: ', seg_sizes)
        print('segm_labels: ', seg_labels)
        seg_labels = [x for _, x in sorted(zip(seg_sizes, seg_labels))]
        seg_labels.reverse()
        print('segm_sizes: ', seg_sizes)
        print('segm_labels: ', seg_labels)

        seg_missing = num_of_wanted_seg - num_of_seg
        print('seg_missing: ', seg_missing)
        if seg_missing < num_of_seg:
            label_current = len(seg_labels) + 1
            for i in range(0, seg_missing):
                temp = segments[segments == seg_labels[i]]
                print('temp = ', temp)
                for j in range(0, int(len(temp) / 2)):
                    temp[j] = label_current
                label_current += 1
                segments[segments == seg_labels[i]] = temp
        else:
            label_current = len(seg_labels) + 1 #1
            num_of_new_seg_per_old_seg = int(seg_missing / num_of_seg) + 1
            print('num_of_new_seg_per_old_seg: ', num_of_new_seg_per_old_seg)
            for i in range(0, num_of_seg):
                temp = segments[segments == seg_labels[i]]
                print('temp_before = ', temp)
                counter_local = 1
                print('counter_local * int(len(temp) / num_of_new_seg_per_old_seg): ', counter_local * int(len(temp) / num_of_new_seg_per_old_seg))
                print('len(temp): ', len(temp))
                for j in range(0, len(temp)):
                    temp[j] = label_current
                    if j == counter_local * int(len(temp) / num_of_new_seg_per_old_seg):
                        print('IN')
                        label_current += 1
                        counter_local += 1
                print('temp_after = ', temp)        
                segments[segments == seg_labels[i]] = temp           

        end = time.time()

        print("\sm7 runtime: ", end - start)

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

        print('np.unique(segments): ', np.unique(segments))

        return segments

    def semantic(self, image, img_rgb, x_odom, y_odom, devDistance, sign, costmap_info, map_info, tf_odom_map):
        print('\nsemantic starts')

        from skimage.segmentation import slic
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import time

        #semantic_map = np.array(pd.read_csv('~/amar_ws/semantic_map.csv'))
        #semantic_map = np.array(pd.read_csv('~/amar_ws/map_sem.csv'))

        #start = time.time()

        # Find segments
        #segments = self.sm1(img_rgb)
        #segments = self.sm2(image, img_rgb, x_odom, y_odom)
        #segments = self.sm3(image, img_rgb, x_odom, y_odom, devDistance, sign)
        #segments = self.sm4(image, img_rgb, x_odom, y_odom, devDistance, sign)
        #segments = self.sm5(image, img_rgb, x_odom, y_odom, devDistance, sign)
        segments = self.sm6(image, img_rgb, x_odom, y_odom, devDistance, sign)
        
        print('\nsemantic ends')

        return segments        

    def semantic_segment(self, image, img_rgb, costmap_info, map_info, tf_odom_map):
        print('\nsemantic_segment starts')

        from skimage.segmentation import slic
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import time

        semantic_map = np.array(pd.read_csv('~/amar_ws/src/lime_explainer/include/lime_explainer/Input/semantic_global_map.csv'))

        start = time.time()

        # Find segments
        segments = slic(img_rgb, n_segments=8, compactness=100, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                            start_label=1, mask=None)
        #pd.DataFrame(segments).to_csv('segments.csv')

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments.astype('float64'), aspect='auto')
        fig.savefig('segments.png', transparent=False)
        fig.clf()
        '''

        obstacles = []                    

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if image[i, j] == 99:
                    obstacles.append(segments[i, j])

        obstacles = np.unique(obstacles)            
        #print('obstacles: ', obstacles)
        #print('obstacles: ', obstacles.shape)

        localCostmapOriginX = costmap_info.iloc[0, 3]
        localCostmapOriginY = costmap_info.iloc[0, 4]
        localCostmapRes = costmap_info.iloc[0, 0]

        mapOriginX = map_info.iloc[0, 4]
        mapOriginY = map_info.iloc[0, 5]
        mapRes = map_info.iloc[0, 1]

        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([tf_odom_map.iloc[0, 3], tf_odom_map.iloc[0, 4], tf_odom_map.iloc[0, 5], tf_odom_map.iloc[0, 6]])
        r_array = np.asarray(r.as_matrix())
        t = np.array([tf_odom_map.iloc[0, 0], tf_odom_map.iloc[0, 1], tf_odom_map.iloc[0, 2]])

        upper_left = np.array([localCostmapOriginX, localCostmapOriginY, 0.0]).dot(r_array) + t
        upper_left_x = int((upper_left[0] - mapOriginX) / mapRes)
        upper_left_y = int((upper_left[1] - mapOriginY) / mapRes)

        upper_right = np.array([159 * localCostmapRes + localCostmapOriginX, 0 * localCostmapRes + localCostmapOriginY, 0.0]).dot(r_array) + t
        upper_right_x = int((upper_right[0] - mapOriginX) / mapRes)
        upper_right_y = int((upper_right[1] - mapOriginY) / mapRes)

        down_left = np.array([0 * localCostmapRes + localCostmapOriginX, 159 * localCostmapRes + localCostmapOriginY, 0.0]).dot(r_array) + t
        down_left_x = int((down_left[0] - mapOriginX) / mapRes)
        down_left_y = int((down_left[1] - mapOriginY) / mapRes)

        down_right = np.array([159 * localCostmapRes + localCostmapOriginX, 159 * localCostmapRes + localCostmapOriginY, 0.0]).dot(r_array) + t
        down_right_x = int((down_right[0] - mapOriginX) / mapRes)
        down_right_y = int((down_right[1] - mapOriginY) / mapRes)

        semantic = False
        for i in range(upper_left_x, upper_right_x + 1):
            for j in range(upper_left_y, down_left_y + 1):
                if semantic_map[i, j] >= 100:
                    semantic = True
                    break   

        semantic_segments = segments             

        #print('(i, j) = ', (i, j))
        #print('semantic: ', semantic)
        if semantic == True:
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    odom_x = i * localCostmapRes + localCostmapOriginX
                    odom_y = j * localCostmapRes + localCostmapOriginY

                    p = np.array([odom_x, odom_y, 0.0])
                    pnew = p.dot(r_array) + t

                    map_x = int((pnew[0] - mapOriginX) / mapRes)
                    #print('map_x: ', map_x)
                    map_y = int((pnew[1] - mapOriginY) / mapRes)
                    #print('map_y: ', map_y)

                    #print(semantic_map[map_x, map_y])
                    if -1 < map_x < semantic_map.shape[0] and -1 < map_y < semantic_map.shape[1]: 
                        semantic_segments[i, j] = semantic_map[map_x, map_y]
            
        for i in range(0, obstacles.shape[0]):
            #print('obstacles[i]: ', obstacles[i])
            semantic_segments[segments == obstacles[i]] = 99 - 10 * obstacles[i]

        for i in range(0, len(np.unique(segments))):
            segments[segments == np.unique(segments)[i]] = i

        end = time.time()
        print('\nsemantic_segment runtime: ', end-start)
        
        #pd.DataFrame(semantic_segments).to_csv('semantic_segments.csv')

        '''
        fig = plt.figure(frameon=False)
        #w = 1.6 #* 3
        #h = 1.6 #* 3
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(semantic_segments.astype('float64'), aspect='auto')
        fig.savefig('semantic_segments.png', transparent=False)
        fig.clf()
        '''

        print('\nsemantic_segment ends')

        return np.array(semantic_segments)        

    def explain_instance(self, image, classifier_fn, costmap_info, map_info, tf_odom_map, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """

        import copy
        image_orig = copy.deepcopy(image)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # my change
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            segments = segmentation_fn(image)
        elif segmentation_fn == 'custom_segmentation':
            #segments = self.sm1(image)
            #segments = self.sm2(image_orig, image, x_odom, y_odom)
            #segments = self.sm3(image_orig, image, x_odom, y_odom, devDistance, sum)
            #segments = self.sm4(image_orig, image, x_odom, y_odom, devDistance, sum)
            #segments = self.sm5(image_orig, image, x_odom, y_odom, devDistance, sum)
            #segments = self.sm6(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            segments = self.sm7(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
        elif segmentation_fn == 'semantic_segmentation':
            segments = self.sm7(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.sm6(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.semantic(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, costmap_info, map_info, tf_odom_map)
            #segments = self.semantic_segment(image_orig, image, costmap_info, map_info, tf_odom_map)    
        else:
            segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        #distance_metric = 'jaccard'
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp, segments

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """

        #print('data_labels starts')

        n_features = np.unique(segments).shape[0]

        '''
        # Original perturbation
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        #print('data: ', data)
        #print('data.shape: ', data.shape)
        '''

        #'''
        # My perturbation - test all possible combinations
        num_samples = 2 ** n_features
        import itertools
        lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
        data = np.array(lst).reshape((num_samples, n_features))
        #print('lst: ', lst)
        #print('len(lst): ', len(lst))
        #print('data: ', data)
        #print('data.shape: ', data.shape)
        #'''

        labels = []

        show_free_space = False

        #print("data before: ", data)
        if show_free_space == True:
            data[0, :] = 1
            data[-1, :] = 0 # only if I use my perturbation
        else:
            print('data.shape[0]: ', data.shape[0])
            data = data[int(data.shape[0]/2):, :]    
            #for i in range(0, data.shape[0]):
            #    data[i, 0] = 1
        #print("data after: ", data)
        #import pandas as pd
        #pd.DataFrame(data).to_csv('data.csv', index=False, header=False)

        imgs = []
        rows = tqdm(data) if progress_bar else data
        #print('rows: ', rows)
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            #print('mask.shape: ', mask.shape)
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

        #print('data_labels ends')

        #print('data: ', data)
        #print('labels: ', np.array(labels))

        return data, np.array(labels)




    def explain_instance_evaluation(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_segments=1,
                         num_segments_current=1,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # my change
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            segments = segmentation_fn(image)
        elif segmentation_fn == 'custom_segmentation':
            start = time.time()
            segments = self.mySlic(image)
            end = time.time()
            segmentation_time = end - start
            #segmentation_time = round(end - start, 3)
        else:
            segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels, classifier_fn_time, planner_time = self.data_labels_evaluation(image, fudged_image, segments,
                                        classifier_fn, num_segments, num_segments_current,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp, segments, segmentation_time, classifier_fn_time, planner_time

    def data_labels_evaluation(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_segments,
                    num_segments_current,
                    batch_size=10,
                    progress_bar=True):

        #print('num_segments: ', num_segments)
        #print('num_segments_current: ', num_segments_current)

        import itertools
        lst = list(map(list, itertools.product([0, 1], repeat=num_segments)))
        data = np.array(lst).reshape((2**num_segments, num_segments))
        data[0, :] = 1
        data[-1, :] = 0

        labels = []

        if num_segments != num_segments_current:
            data_copy = copy.deepcopy(data)
            data = data_copy[np.random.choice(len(data_copy), 2**num_segments_current, replace=False)]
            data[0, :] = 1

        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                #start = time.time()
                preds = classifier_fn(np.array(imgs))
                #end = time.time()
                #classifier_fn_time = round(end - start, 3)
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            start = time.time()
            preds, planner_time = classifier_fn(np.array(imgs))
            end = time.time()
            classifier_fn_time = end - start
            #classifier_fn_time = round(end - start, 3)
            labels.extend(preds)

        #print('data_labels ends')

        return data, np.array(labels), classifier_fn_time, planner_time
