from hw3_utils import abstract_classifier
from aux_functions import *
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict


class knn_classifier(abstract_classifier):

    def __init__(self, k, data, labels, subtract, divisor):

        self.k = k
        self.data = data
        self.labels = labels
        self.subtracts = subtract
        self.divisors = divisor

    def classify(self, features):

        normalized_features = normalize(features, self.subtracts, self.divisors)

        dist = [euclidean_distance(normalized_features, x) for x in self.data]
        dic = {dist[i]: self.labels[i] for i in range(len(self.labels))}
        sorted_dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))

        values = (list(sorted_dic.values()))[1:self.k]

        return sum(values) > self.k / 2
