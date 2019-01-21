import numpy as np

def euclidean_distance(p1, p2):

    if len(p1) == 0 or len(p2) == 0:
        return 0

    return (sum([(x - y) ** 2 for x, y in zip(p1, p2)])) ** 0.5


def normalize(vec, subtracts, divisors):
    return (np.array(vec) - np.array(subtracts))/np.array(divisors)

def countBool(arr):
    num_trues = sum(arr)
    num_falses = len(arr) - num_trues

    return num_trues, num_falses
