import numpy
from aux_functions import *

def split_crosscheck_groups(dataset, num_folds):

    groups = []
    for i in range(num_folds):
        groups.append([])

    data = dataset[0]
    labels = dataset[1]

    true_data = numpy.array([data[i] for i in range(len(labels)) if labels[i]])
    false_data = numpy.array([data[i] for i in range(len(labels)) if not labels[i]])

    num_trues, num_falses = countBool(labels)

    minority = false_data
    majority = true_data

    if num_falses > num_trues:
        minority = true_data
        majority = false_data

    toGroups(groups, majority, num_folds)
    toGroups(groups, minority, num_folds)
    # majority, minority = divideRemains(groups, minority, majority, num_folds)

    print(len(groups[0]))
    print(len(groups[1]))
    print(len(groups[2]))
    print(len(groups[3]))


def toGroups(groups, arr, num_of_groups):

    size = int(len(arr) / num_of_groups)

    for i in range(num_of_groups):
        indices = numpy.random.choice(len(arr), size, replace=False)

        chosen = arr[indices]

        tmp = numpy.delete(arr, indices, axis=0)
        arr = tmp

        for j in range(len(chosen)):
            groups[i].append(chosen[j])

    for i in range(len(arr)):
        groups[i].append(arr[i])


