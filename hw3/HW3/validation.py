import numpy
from aux_functions import *
import pickle

def split_crosscheck_groups(dataset, num_folds):

    groups = []
    group_labels = []
    for i in range(num_folds):
        groups.append([])
        group_labels.append([])

    data = dataset[0]
    labels = dataset[1]

    true_data = numpy.array([data[i] for i in range(len(labels)) if labels[i]])
    false_data = numpy.array([data[i] for i in range(len(labels)) if not labels[i]])

    num_trues, num_falses = countBool(labels)

    minority = false_data
    majority = true_data
    majority_tag = True

    if num_falses > num_trues:
        minority = true_data
        majority = false_data
        majority_tag = False

    toGroups(groups, majority, num_folds, group_labels, majority_tag)
    toGroups(groups, minority, num_folds, group_labels, not majority_tag)

    toFiles(groups, group_labels)


def toGroups(groups, arr, num_of_groups, groups_labels, tag):

    size = int(len(arr) / num_of_groups)

    for i in range(num_of_groups):
        indices = numpy.random.choice(len(arr), size, replace=False)

        chosen = arr[indices]

        tmp = numpy.delete(arr, indices, axis=0)
        arr = tmp

        for j in range(len(chosen)):
            groups[i].append(chosen[j])
            groups_labels[i].append(tag)

    for i in range(len(arr)):
        groups[i].append(arr[i])
        groups_labels[i].append(tag)


def toFiles(groups, group_lables):

    for i in range(len(groups)):
        output_file = open("ecg_fold_" + str(i) + ".data", "wb")

        pickle.dump(groups[i], output_file)
        pickle.dump(group_lables[i], output_file)

        output_file.close()


def load_k_fold_data(i):

    input_file = open("ecg_fold_" + str(i) + ".data", "rb")

    group = pickle.load(input_file)
    labels = pickle.load(input_file)

    return group, labels


def evaluate(classifier_factory, k):
    data = []
    labels = []
    for i in range(k):
        d, l = load_k_fold_data(i)
        data.append(d)
        labels.append(l)
    
    acc, err = 0, 0
    for i in range(k):
        train_data = [datum for group in data[:i] + data[i+1:] for datum in group]
        train_labels = [label for group in labels[:i] + labels[i+1:] for label in group]
        classifier = classifier_factory.train(train_data, train_labels)
        for j in range(len(data[i])):
            result = classifier.classify(data[i][j])
            if result == labels[i][j]:
                acc += 1
            else:
                err += 1
    N = sum([len(group) for group in data])
    return acc/N, err/N

def get_best(classifier_factory, k):
    """ Returns the best classifier made by classifier_factory on k folds of data.
        That is, returns the classifier whose train partition was most accurate over its validation partition """
    data = []
    labels = []
    for i in range(k):
        d, l = load_k_fold_data(i)
        data.append(d)
        labels.append(l)
    
    best_validation = 0
    best_acc = 0
    best_classifier = None
    for i in range(k):
        # Build classifier
        acc = 0
        train_data = [datum for group in data[:i] + data[i+1:] for datum in group]
        train_labels = [label for group in labels[:i] + labels[i+1:] for label in group]
        classifier = classifier_factory.train(train_data, train_labels)
        # Test classifier
        print("validating with fold {}".format(i))
        for j in range(len(data[i])):
            result = classifier.classify(data[i][j])
            if result == labels[i][j]:
                acc += 1
        # Update best classifier
        if acc/len(data[i]) > best_acc:
            best_acc = acc/len(data[i])
            best_classifier = classifier
            best_validation = i
    
    # Return best classifier
    return {'classifier':best_classifier, 'acc': best_acc, 'validation': best_validation}
