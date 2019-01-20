from hw3_utils import abstract_classifier
from aux_functions import *
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict


class knn_classifier(abstract_classifier):

    def __init__(self, k, data, labels, min_list, max_list):

        self.k = k
        self.data = data
        self.labels = labels
        self.min_list = min_list
        self.max_list = max_list

    def classify(self, features):

        normalized_features = self.normalize(features)

        dist = [euclidean_distance(normalized_features, x) for x in self.data]
        dic = {dist[i]: self.labels[i] for i in range(len(self.labels))}
        sorted_dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))

        values = (list(sorted_dic.values()))[1:self.k]

        return sum(values) > self.k / 2

    def normalize(self, features):

        normalized_features = []

        for i in range(len(features)):
            normalized_features.append((features[i] - self.min_list[i]) / (self.max_list[i] - self.min_list[i]))

        return normalized_features
