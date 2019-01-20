from hw3_utils import abstract_classifier_factory
from knn_classifier import knn_classifier
from aux_functions import *
import numpy as np


class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):

        data = np.array(data)
        data = data.astype(np.float)
        labels = np.array(labels)

        min_list = []
        max_list = []

        for i in range(len(data[0])):
            min_val = min(data[:, i])
            max_val = max(data[:, i])
            min_list.append(min_val)
            max_list.append(max_val)
            data[:, i] = normalize(data[:, i], min_val, max_val)

        return knn_classifier(self.k, data, labels, min_list, max_list)
