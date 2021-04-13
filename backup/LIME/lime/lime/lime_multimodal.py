"""
Functions for explaining classifiers that use Multimodal data.
Developed initially for the hateful memes challenge in mmf
"""

import copy
from functools import partial
import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm

from wrappers.scikit_image import SegmentationAlgorithm
from lime_base import LimeBase

from lime_text import IndexedString, TextDomainMapper
from exceptions import LimeError

from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from skimage.segmentation import mark_boundaries
#from object_detection import *


class MultiModalExplanation(object):
    def __init__(self, image, segments,
                 domain_mapper,
                 n_txt_features,
                 n_img_features,
                 n_detection_features,
                 detection_label,
                 mode='classification',
                 class_names=None,
                 random_state=None,
                 ):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.n_txt_features = n_txt_features
        self.n_img_features = n_img_features
        self.n_detection_features = n_detection_features
        self.detection_label = detection_label
        self.local_exp = {}
        self.intercept = {}
        self.score = {}
        self.local_pred = {}
        self.unsorted_weights = {}


        # divide explanations of the two modalities
        self.local_exp_img = {}
        self.local_exp_txt = {}
        self.local_exp_det = {}

        if mode == 'classification':
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):

        if label not in self.local_exp_img:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp_img[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) >= min_weight:
                    c = 0 if w < 0 else 1
                    mask[segments == f] = -1 if w < 0 else 1
                    temp[segments == f] = image[segments == f].copy()
                    temp[segments == f, c] = np.max(image)
            return temp, mask

    def get_feature_and_weight(self):
        return self.local_exp_img

    def as_list(self, label=1, **kwargs):
        label_to_use = label
        ans = self.domain_mapper.map_exp_ids(self.local_exp_txt[label_to_use], **kwargs)
        ans = [(x[0], float(x[1])) for x in ans]
        return ans

    def get_explanation(self, label, num_features=30, which_exp='positive'):

        # the explanation to display:
        '''

        :param label: label to explain
        :param num_features: how many top features to display
        :param positive: want features that encourage or discourage the dicision (label)
        :return:
            this_exp: raw feature and weight values
            readable_exp: a text description of the explanation
            txt_exp_list: text part of the explanation, ready to display
            temp, mask: image part of the explanation, ready to display
        '''
        this_exp = np.array(self.local_exp[label])

        if which_exp == 'positive':
            positives = this_exp[this_exp[:, 1] >= 0]
        else:   # negative
            positives = this_exp[this_exp[:, 1] < 0]

        if positives.shape[0] < num_features:
            num_features = positives.shape[0]
        top_exp = positives[:num_features]        
        top_exp_unique, top_idx = np.unique(top_exp[:, 0], return_index=True)

        txt_exp = top_exp[top_exp[:, 0] < self.n_txt_features]
        n_txt_exp = txt_exp.shape[0]
        txt_top_idx = []
        for txt_feature in txt_exp:
            txt_top_idx.append(top_idx[top_exp_unique==txt_feature[0]] + 1)

        img_exp = top_exp[self.n_txt_features <= top_exp[:, 0]]
        img_exp = img_exp[img_exp[:, 0] < (self.n_txt_features + self.n_img_features)]
        n_img_exp = img_exp.shape[0]
        img_top_idx = []
        for img_feature in img_exp:
            img_top_idx.append(top_idx[top_exp_unique==img_feature[0]] + 1)

        # detection features
        det_exp = top_exp[top_exp[:, 0] >= self.n_txt_features + self.n_img_features]
        n_det_exp = det_exp.shape[0]
        det_top_idx = []
        for det_feature in det_exp:
            det_top_idx.append(det_feature[0] - n_txt_exp - n_img_exp)  # index for retrieving the labels

        if n_det_exp != 0:
            readable_exp_det = f"Also, we have detected {n_det_exp} types " \
                            f"of objects from the input image that can be the reason for the decision, they are:"
            for i in det_top_idx:
                readable_exp_det += str(self.detection_label[i])
        else:
            readable_exp_det = "No objects in the image contributed to the model decision"

        readable_exp = f"among the top 20 features, {n_txt_exp} are from the text (the top"
        for i in txt_top_idx:
            readable_exp += str(i)

        readable_exp += f"), {n_img_exp} are from the image (the top"
        for j in img_top_idx:
            readable_exp += str(j)

        readable_exp += f"), displayed as follows"
        readable_exp += readable_exp_det

        txt_exp_list = np.array(self.as_list(label), dtype='object')

        # return explanations upon request
        if which_exp == 'positive':
            txt_list = txt_exp_list[txt_exp_list[:, 1] >= 0]
            temp, mask = self.get_image_and_mask(label, num_features=n_img_exp, positive_only=True)
        else:
            txt_list = txt_exp_list[txt_exp_list[:, 1] < 0]
            temp, mask = self.get_image_and_mask(label, num_features=n_img_exp, positive_only=False, negative_only=True)

        txt_list = txt_list[:n_txt_exp]
        img_boundary = mark_boundaries(temp, mask)
        im = Image.fromarray(np.uint8(img_boundary*255))
        return this_exp, readable_exp, txt_list, im


class LimeMultimodalExplainer(object):

    def __init__(self, image, text, kernel_width=.25, kernel=None,
                 feature_selection='auto', class_names=None):
        self.image = image
        self.text = text
        self.random_state = check_random_state(None)
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.feature_selection = feature_selection
        self.class_names = class_names
        self.base = LimeBase(kernel_fn, verbose=False,
                                       random_state=self.random_state)

    def explain_instance(self, classifier_fn, n_samples, top_labels=2):

        # get data, labels, distances to fit the linear model
        data, labels, distances, n_txt_features, n_img_features,\
            segments, domain_mapper, n_detection_features, detection_label = self.data_labels(n_samples, classifier_fn)
        num_features = data.shape[1]

        if self.class_names is None:
            self.class_names = [str(x) for x in range(labels[0].shape[0])]

        ret_exp = MultiModalExplanation(self.image, segments, domain_mapper=domain_mapper,
                                        n_txt_features=n_txt_features,
                                        n_img_features=n_img_features,
                                        n_detection_features=n_detection_features,
                                        detection_label=detection_label,
                                        class_names=self.class_names,
                                        random_state=self.random_state
                                        )
        ret_exp.predict_proba = labels[0]

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.unsorted_weights[label],
             ret_exp.local_exp[label],
             ret_exp.local_exp_txt[label],
             ret_exp.local_exp_img[label],
             ret_exp.local_exp_det[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features, n_txt_features, n_img_features, n_detection_features,
                feature_selection=None)

            # split local explanation into text and image features

        return ret_exp

    def data_labels(self, num_samples, classifier_fn, detection=False):
        '''
        Steps of this function:
            1. generate perturbed text features and image features
            2. in a loop, 1) using these features to make instances of perturbed (text, image) pairs,
                          2) make predictions on these pairs, store labels into 'labels'
            3. concatenate text and image features, store into 'data',
                also append the original input and prediction of it
            4. calculate distances

            TODO: add object detection: first run on original image, create feature components,
                    then run on perturbed images to get corresponding value

        :param num_samples:
        :param classifier_fn:
        :param object_detection:
        :return:
        '''

        ''' 1. make text features '''
        indexed_string = IndexedString(self.text, bow=True, split_expression=r'\W+', mask_string=None)
        domain_mapper = TextDomainMapper(indexed_string)

        doc_size = indexed_string.num_words()
        sample = self.random_state.randint(1, doc_size + 1, num_samples)                        # num_samples - 1
        data_txt = np.ones((num_samples, doc_size))
        # data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_data_txt = []

        ''' 1. make image features '''
        random_seed = self.random_state.randint(0, high=1000)
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                max_dist=200, ratio=0.2,
                                                random_seed=random_seed)

        #segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=200, sigma=2, min_size=100)
        '''segmentation_fn = SegmentationAlgorithm('slic', n_segments=60, compactness=10, sigma=1,
                     start_label=1)'''

        segments = segmentation_fn(self.image)  # get segmentation
        n_img_features = np.unique(segments).shape[0]  # get num of superpixel features
        data_img = self.random_state.randint(0, 2, n_img_features * num_samples).reshape(
            (num_samples, n_img_features))
        data_img_rows = tqdm(data_img)
        imgs = []

        ''' 1. make object detection features 
        if detection:
            predictor, cfg = object_detection_predictor()
            ori_label = object_detection_obtain_label(predictor, cfg, self.image)
            num_object_detection = ori_label.shape[0]
            data_object_detection = np.zeros((num_samples,num_object_detection))'''
        
        # create fudged_image
        fudged_image = self.image.copy()
        for x in np.unique(segments):
            fudged_image[segments == x] = (
                np.mean(self.image[segments == x][:, 0]),
                np.mean(self.image[segments == x][:, 1]),
                np.mean(self.image[segments == x][:, 2]))

        # img_features[0, :] = 1  # the first sample is the full image                                # num_samples

        '''2. create data instances and make predictions'''
        labels = []
        for i, instance in enumerate(zip(sample, data_img_rows)):
            size_txt, row_img = instance

            # make text instance
            inactive = self.random_state.choice(features_range, size_txt,
                                                replace=False)
            data_txt[i, inactive] = 0
            inverse_data_txt.append(indexed_string.inverse_removing(inactive))

            # make image instance
            temp = copy.deepcopy(self.image)
            zeros = np.where(row_img == 0)[0]             # get segment numbers that are turned off in this instance
            mask = np.zeros(segments.shape).astype(bool)
            for zero in zeros:
                mask[segments == zero] = True
            temp[mask] = fudged_image[mask]

            '''if detection:
                label = object_detection_obtain_label(predictor, cfg, temp)
                label_diff = compare_labels(ori_label,label)
                data_object_detection[i] = label_diff'''
            imgs.append(temp)

            # make prediction and append result
            if len(imgs) == 10:
                preds = classifier_fn(imgs, inverse_data_txt)
                labels.extend(preds)
                imgs = []
                inverse_data_txt = []

        if len(imgs) > 0:
            preds = classifier_fn(imgs, inverse_data_txt)
            labels.extend(preds)

        '''3. concatenate and append features'''
        data = np.concatenate((data_txt, data_img), axis=1)

        # append the original input to the last
        orig_img_f = np.ones((n_img_features,))
        orig_txt_f = np.ones(doc_size)

        '''if detection:
            data = np.concatenate((data, data_object_detection),axis=1)
            orig_ot = np.ones(num_object_detection)
            data = np.vstack((data, np.concatenate((np.concatenate((orig_txt_f, orig_img_f)),orig_ot))))
        else:'''
        data = np.vstack((data, np.ones((data.shape[1])))) ###
            
        labels.extend(classifier_fn([self.image], [self.text]))


        '''4. compute distance# distances[:, :(doc_size-1)] *= 100
            use platt scaling t get relative importance of text and image modalities
        '''

        labels = np.array(labels, dtype=float)

        # Modify MMF source code to zero out image / text attributes
        #dummy_label_image = np.array(classifier_fn([self.image], [self.text], zero_text=True))  # zero out text
        #dummy_label_text = np.array(classifier_fn([self.image], [self.text], zero_image=True))  # zero out image

        # perform calibration
        labels_for_calib = np.array(labels[:, 0] < 0.5, dtype=float)
        calibrated = CalibratedClassifierCV(cv=3)
        calibrated.fit(data[:,:doc_size + n_img_features], labels_for_calib)

        calib_data = np.ones((3, doc_size + n_img_features), dtype=float)
        calib_data[0][:doc_size] = 0        # zero out text
        calib_data[1][doc_size:] = 0        # zero out image
        calibrated_labels = calibrated.predict_proba(calib_data)
        
        delta_txt = abs(calibrated_labels[-1][0] - calibrated_labels[0][0])
        delta_img = abs(calibrated_labels[-1][0] - calibrated_labels[1][0])

        ratio_txt_img = max(min(10, delta_txt/delta_img), 0.1)

        # calculate distances
        distances_img = sklearn.metrics.pairwise_distances(
            data[:, doc_size:],
            data[-1, doc_size:].reshape(1, -1),
            metric='cosine'
        ).ravel()

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[-1], metric='cosine').ravel()

        distances_txt = distance_fn(sp.sparse.csr_matrix(data[:, :doc_size]))

        distances = 1/(1 + ratio_txt_img) * distances_img + (1 - 1/(1 + ratio_txt_img)) * distances_txt

        # As required by lime_base, make the first element of data, labels, distances the original data point
        data[0] = data[-1]
        labels[0] = labels[-1]
        distances[0] = distances[-1]

        '''if not detection:'''
        num_object_detection = 0
        ori_label = None

        return data, labels, distances, doc_size, n_img_features, segments, domain_mapper, num_object_detection, ori_label
