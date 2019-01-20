from knn_classifier import knn_classifier
from knn_factory import knn_factory
from hw3_utils import *
import numpy
from validation import *
from sklearn.model_selection import StratifiedKFold
from PerceptronClassifier import PerceptronClassifierFactory
from Id3Classifier import Id3ClassifierFactory

if __name__ == '__main__':
    data, labels, test = load_data()
    #
    data1 = np.array([[0, 0],
                      [1, 1],
                      [2, 2],
                      [3, 3],
                      [4, 4],
                      [5, 5],
                      [6, 6],
                      [7, 7],
                      [8, 8],
                      [9, 9],
                      [10, 10],
                      [11, 11],
                      [12, 12],
                      [13, 13],
                      [14, 14],
                      [15, 15],
                      [16, 16],
                      [17, 17],
                      [18, 18],
                      [19, 19],
                      [20, 20],
                      [21, 21],
                      [22, 22],
                      [23, 23],
                      [24, 24],
                      [25, 25],
                      [26, 26],
                      [27, 27]])

    labels1 = [True, True, True, True, True, True, True,
              True, True, True, True, True, True, True,
              True, True, True, True, True, True, True,
              True, False, False, False, False, False, False]

    # split_crosscheck_groups([data, labels], 2)

    # with open('experiments6.csv', 'w') as fp:
    #     for k in [1,3,5,7,13]:
    #         acc, err = evaluate(knn_factory(k), 2)
    #         fp.write("{}, {}, {}\n".format(k, acc, err))

    with open('experiments12.csv', 'w') as fp:
            acc, err = evaluate(Id3ClassifierFactory(), 2)
            fp.write("{}, {}, {}\n".format(1, acc, err))
            acc, err = evaluate(PerceptronClassifierFactory(), 2)
            fp.write("{}, {}, {}\n".format(2, acc, err))



    # acc, err = evaluate(knn_factory(3), 2)
    # print("{} {}".format(acc, err))











