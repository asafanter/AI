from hw3_utils import abstract_classifier_factory
from knn_classifier import knn_classifier
from aux_functions import *


class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):

        for e in data:
            normalize(e)

        return knn_classifier(self.k, data, labels)
