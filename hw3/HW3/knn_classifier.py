from hw3_utils import abstract_classifier
from aux_functions import *
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict


class knn_classifier(abstract_classifier):

    def __init__(self, k, data, labels):

        self.k = k
        self.data = [normalize(datum) for datum in data]
        self.labels = labels

    def classify(self, features):

        normalize(features)

        dist = [euclidean_distance(features, x) for x in self.data]
        dic = {dist[i]: self.labels[i] for i in range(len(self.labels))}
        sorted_dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))

        values = (list(sorted_dic.values()))[1:self.k]

        return sum(values) > self.k / 2
