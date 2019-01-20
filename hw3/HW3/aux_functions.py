import numpy as np

def euclidean_distance(p1, p2):

    if len(p1) == 0 or len(p2) == 0:
        return 0

    return (sum([(x - y) ** 2 for x, y in zip(p1, p2)])) ** 0.5


def normalize(vec, min_val, max_val):

    tot_range = max_val - min_val

    res = vec
    res = res.astype(np.float)

    for i in range(len(res)):

        numerator = res[i] - min_val

        if numerator == 0.:
            res[i] = 0.
            continue
        else:
            res[i] = numerator / tot_range

    return res


def countBool(arr):
    num_trues = sum(arr)
    num_falses = len(arr) - num_trues

    return num_trues, num_falses
