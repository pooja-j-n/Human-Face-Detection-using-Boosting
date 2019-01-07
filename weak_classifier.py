from abc import ABC, abstractmethod
import datetime
from copy import deepcopy
import numpy as np
from operator import itemgetter
import pdb
from joblib import Parallel, delayed
import logging as log


class Weak_Classifier(ABC):
    # initialize a harr filter with the positive and negative rects
    # rects are in the form of [x1, y1, x2, y2] 0-index
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        self.id = id
        self.plus_rects = plus_rects
        self.minus_rects = minus_rects
        self.num_bins = num_bins
        self.activations = None
        self.threshold = None

    # take in one integrated image and return the value after applying the image
    # integrated_image is a 2D np array
    # return value is the number BEFORE polarity is applied
    def apply_filter2image(self, integrated_image):
        pos = 0
        for rect in self.plus_rects:
            rect = [int(n) for n in rect]
            pos += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        neg = 0
        for rect in self.minus_rects:
            rect = [int(n) for n in rect]
            neg += integrated_image[rect[3], rect[2]] \
                   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1]) \
                   - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]]) \
                   - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        return pos - neg

    # take in a list of integrated images and calculate values for each image
    # integrated images are passed in as a 3-D np-array
    # calculate activations for all images BEFORE polarity is applied
    # only need to be called once
    def apply_filter(self, integrated_images):
        values = []
        for idx in range(integrated_images.shape[0]):
            values.append(self.apply_filter2image(integrated_images[idx, ...]))
        if (self.id + 1) % 100 == 0:
            print('Weak Classifier No. %d has finished applying' % (self.id + 1))
        return values

    # using this function to compute the error of
    # applying this weak classifier to the dataset given current weights
    # return the error and potentially other identifier of this weak classifier
    # detailed implementation is up you and depends
    # your implementation of Boosting_Classifier.train()
    @abstractmethod
    def calc_error(self, weights, labels):
        pass

    @abstractmethod
    def predict_image(self, integrated_image):
        pass


class Ada_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.polarity = None
        self.threshold = None

    def calc_error(self, weights, labels):
        ######################
        ######## TODO ########
        ######################
        thresholds = []
        errors = []
        polarities = []
        min_activation = min(self.activations)
        max_activation = max(self.activations)
        scale = 50
        thresholds = np.linspace(min_activation, max_activation, scale)

        for i in range(len(thresholds)):
            temp_polarity = 1
            temp_error = (np.sum(((temp_polarity * (np.sign(self.activations - thresholds[i]))) != labels) * weights))
            if temp_error > 0.5:
                temp_error = 1 - temp_error
                temp_polarity = -1
            errors.append(temp_error)
            polarities.append(temp_polarity)

        ind_final_error, final_error = min(enumerate(errors), key=itemgetter(1))
        threshold = thresholds[ind_final_error]
        polarity = polarities[ind_final_error]

        return final_error,threshold,polarity

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        return self.polarity * np.sign(value - self.threshold)


class Real_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.thresholds = None  # this is different from threshold in ada_weak_classifier, think about it
        self.bin_pqs = None
        self.train_assignment = None

    def calc_error(self, weights, labels, t):
        ######################
        ######## TODO ########
        ######################

        #pdb.set_trace()
        self.bin_pqs = np.zeros(shape=(2,self.num_bins))
        size_data = len(weights)
        #pdb.set_trace()

        sorted_activations = np.array([np.sort(self.activations), np.argsort(self.activations)])

        self.thresholds = np.linspace(sorted_activations[0][0],sorted_activations[0][size_data-1],self.num_bins+1)[1:]

        lower_bound_index = 0
        upper_bound_index = None
        indices = None
        epsilon = 4e-4
        for bin_idx in (range(self.num_bins)):

            if bin_idx==self.num_bins-1:
                upper_bound_index = size_data-1
            else:
                upper_bound_index = np.argmax(sorted_activations[0] > self.thresholds[bin_idx])

            indices = np.int_(sorted_activations[1][lower_bound_index:upper_bound_index])

            if(np.sum(weights[indices] * (labels[indices] == 1))):
                self.bin_pqs[0][bin_idx] = np.sum(weights[indices] * (labels[indices] == 1))
            else:
                self.bin_pqs[0][bin_idx] = epsilon

            if(np.sum(weights[indices] * (labels[indices] == -1))):
                self.bin_pqs[1][bin_idx] = np.sum(weights[indices] * (labels[indices] == -1))
            else:
                self.bin_pqs[1][bin_idx] = epsilon




            #pdb.set_trace()

            lower_bound_index = upper_bound_index

        #pdb.set_trace()

        #Z = 2 * np.sum(np.sqrt(self.bin_pqs[0] * self.bin_pqs[1]))

        #pdb.set_trace()

        #return Z

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        bin_idx = np.sum(self.thresholds < value)
        return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])


def main():
    plus_rects = [(1, 2, 3, 4)]
    minus_rects = [(4, 5, 6, 7)]
    num_bins = 50
    ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
    real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)


if __name__ == '__main__':
    main()
