from hw3_utils import abstract_classifier_factory
from knn_classifier import knn_classifier
from aux_functions import *
import numpy as np
import random


class competition_factory(abstract_classifier_factory):

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
        
        dl_pairs = []
        for i in range(len(data)):
            dl_pairs.append((data[i], labels[i]))

        # Initialize classifier with a random data/label pair
        random.shuffle(dl_pairs)
        used_data = [normalize(dl_pairs[0][0], subtract, divisor)]
        used_labels = [dl_pairs.pop(0)[1]]
        classifier = knn_classifier(self.k, used_data, used_labels, subtract, divisor)
        changed = True
        while changed:
            externals = []
            print("loop {}/{}".format(len(used_labels), len(dl_pairs)))
            changed = False
            random.shuffle(dl_pairs) # Not 100% sure why this is important, but its in the lecture slides ¯\_(ツ)_/¯
            for d, l in dl_pairs:
                if classifier.classify(d) != l:
                    # If the classifier failed, we add the new pair to the internals
                    # and indicate that a change has occured
                    changed = True
                    used_data.append(normalize(d, subtract, divisor))
                    used_labels.append(l)
                    classifier = knn_classifier(self.k, used_data, used_labels, subtract, divisor)
                else:
                    # Otherwise we keep it for the next round, in case
                    # it's classification changes for the worse
                    externals.append((d, l))
            dl_pairs = externals
        
        # When the set is stable, we return that classifier. It is the one with
        # (hopefully) the minimal number of elements that is consistent with the
        # original dataset
        return classifier
