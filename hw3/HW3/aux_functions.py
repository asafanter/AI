def euclidean_distance(p1, p2):

    if len(p1) == 0 or len(p2) == 0:
        return 0

    return (sum([(x - y) ** 2 for x, y in zip(p1, p2)])) ** 0.5


def normalize(vec):

    max_val = max(vec)
    min_val = min(vec)
    tot_range = max_val - min_val

    for i in range(len(vec)):

        numerator = vec[i] - min_val

        if numerator == 0:
            vec[i] = 0
            break
        else:
            vec[i] = numerator / tot_range


def countBool(arr):
    num_trues = sum(arr)
    num_falses = len(arr) - num_trues

    return num_trues, num_falses
