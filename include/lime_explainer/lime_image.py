"""
Functions for explaining classifiers that use Image data.
"""
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

        '''
        # plot for the HARL Workshop 2021 paper
        exp_list = []
        for i in range(0, len(exp)):
            exp_list.append(list(exp[i]))
            exp_list[i][1] = 0.0
        exp_list[1][1] = 0.9
        print('exp_list: ', exp_list)
        exp = []
        for i in range(0, len(exp_list)):
            exp.append(tuple(exp_list[i]))
        print('exp: ', exp)
        '''

        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
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
            #pd.DataFrame(segments).to_csv('segments_lime_image.csv', index=False)
            for f, w in exp[:num_features]:
                #print('(f, w): ', (f, w))

                #if f == 0:
                #    print(image[segments == f])

                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                #temp[segments == f] = image[segments == f].copy()
                # if free space
                if image[segments == f].all() == 0.0:
                    # if positive weight
                    if c == 1:
                        temp[segments == f, 1] = float(np.max(image)) / 2**counter_local # c is channel, RGB - 012
                        temp[segments == f, 0] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 2] = 0.0
                    # if negative weight
                    else:
                        temp[segments == f, 0] = float(np.max(image)) / 2**counter_local  # c is channel, RGB - 012
                        temp[segments == f, 1] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 2] = 0.0
                # if obstacle
                else:
                    # if positive weight
                    if c == 1:
                        temp[segments == f, 1] = float(np.max(image)) / 2**counter_local  # c is channel, RGB - 012
                        temp[segments == f, 0] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 2] = float(np.max(image)) / 2**counter_local #1.0
                    # if negative weight
                    else:
                        temp[segments == f, 0] = float(np.max(image)) / 2**counter_local  # c is channel, RGB - 012
                        temp[segments == f, 1] = 0.0  # c is channel, RGB - 012
                        temp[segments == f, 2] = float(np.max(image)) / 2**counter_local #1.0

                counter_local += 1

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

    def mySlic(self, img_rgb):

        #print('mySlic starts')

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = img_rgb[:, :, 0]
        '''
        # Save segments_1 as a picture
        plt.imshow(img)
        plt.savefig('mySlic_img.png')
        plt.clf()
        '''

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''
        # plot segments_1 with centroids and labels
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_1 as a picture
        plt.imshow(segments_1)
        plt.savefig('mySlic_segments_1.png')
        plt.clf()
        '''
        # find segments_unique_1
        segments_unique_1 = np.unique(segments_1)
        #print('segments_unique_1: ', segments_unique_1)
        #print('segments_unique_1.shape: ', segments_unique_1.shape)


        # Find segments_2
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        #print('segments_unique_2: ', segments_unique_2)
        #print('segments_unique_2.shape: ', segments_unique_2.shape)
        # make obstacles on segments_2 nice - not needed
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 99:
                   segments_2[i, j] = segments_1[i, j] + segments_unique_2.shape[0]
        '''
        # plot segments_2 with centroids and labels
        regions = regionprops(segments_2)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_2 as a picture
        plt.imshow(segments_2)
        plt.savefig('mySlic_segments_2.png')
        plt.clf()
        '''
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        #print('segments_unique_2: ', segments_unique_2)
        #print('segments_unique_2.shape: ', segments_unique_2.shape)
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        '''
        # plot segments with centroids and labels/weights
        plt.imshow(segments_1)
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_1 as a picture before nice segment numbering
        plt.savefig('mySlic_segments_beforeNiceNumbering.png')
        plt.clf()
        '''
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        #print('segments_unique: ', segments_unique)
        #print('segments_unique.shape: ', segments_unique.shape)

        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1 # k+1 must be in order for regionprops() function to work correctly
        # find segments_unique after nice segment numbering

        segments_unique = np.unique(segments_1)
        #print('segments_unique (with nice numbering): ', segments_unique)
        #print('segments_unique.shape (with nice numbering): ', segments_unique.shape)
        '''
        # plot segments with centroids and labels/weights
        plt.imshow(segments_1)
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v-1))
            i = i + 1

        # Save segments as a picture
        plt.savefig('mySlic_segments.png')
        plt.clf()
        '''
        segments_1 = segments_1 - 1

        #import pandas as pd
        #pd.DataFrame(segments_1).to_csv('segments_original.csv', index=False)

        #print('mySlic ends')

        return segments_1

    def explain_instance(self, image, classifier_fn, labels=(1,),
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

        #print('batch_size: ', batch_size)
        #print('\n')

        '''
        # Plot picture
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.savefig('self.lime_image_costmap.png')
        plt.clf()
        '''

        #print('len(image.shape): ', len(image.shape))
        #print('\n')

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
            segments = self.mySlic(image)
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

        '''    
        # Plot picture
        import matplotlib.pyplot as plt
        plt.imshow(fudged_image)
        plt.savefig('self.lime_image_fudged_image.png')
        plt.clf()    
        '''

        top = labels
        #print('top: ', top)

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        #print('data: ', data)
        #print('labels: ', labels)

        #distance_metric = 'jaccard'
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        #print('distances: ', distances)

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
        return ret_exp

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

        #print("data before: ", data)
        data[0, :] = 1
        data[-1, :] = 0 # only if I use my perturbation
        #print("data after: ", data)
        #import pandas as pd
        #pd.DataFrame(data).to_csv('~/amar_ws/data.csv', index=False, header=False)

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
        return ret_exp, segmentation_time, classifier_fn_time, planner_time

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
