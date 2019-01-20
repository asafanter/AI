from sklearn.linear_model import Perceptron
from hw3_utils import abstract_classifier_factory, abstract_classifier


class PerceptronClassifierFactory(abstract_classifier_factory):

    def train(self, data, labels):

        classifier = Perceptron(tol=1e-3, random_state=0)
        classifier = classifier.fit(data, labels)

        return PerceptronClassifier(classifier)


class PerceptronClassifier(abstract_classifier):

    def __init__(self, classifier):

        self.classifier = classifier

    def classify(self, features):
        return self.classifier.predict([features])
