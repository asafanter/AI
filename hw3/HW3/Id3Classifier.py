from sklearn import tree
from hw3_utils import abstract_classifier_factory, abstract_classifier


class Id3ClassifierFactory(abstract_classifier_factory):

    def train(self, data, labels):

        classifier = tree.DecisionTreeClassifier()
        classifier = classifier.fit(data, labels)

        return Id3Classifier(classifier)


class Id3Classifier(abstract_classifier):

    def __init__(self, classifier):

        self.classifier = classifier

    def classify(self, features):
        return self.classifier.predict([features])
