from hw3_utils import abstract_classifier_factory
from knn_classifier import knn_classifier
from aux_functions import *
import numpy as np
import random


class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels)

        subtract = []
        divisor = []

        for i in range(len(data[0])):
            min_val = min(data[:, i])
            max_val = max(data[:, i])
            subtract.append(min_val)
            divisor.append(max_val-min_val)
        
        norm_data = [normalize(datum, subtract, divisor) for datum in data]
        classifier = knn_classifier(self.k, norm_data, labels, subtract, divisor)
        return classifier
