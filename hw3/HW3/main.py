from knn_classifier import knn_classifier
from competition_factory import competition_factory
from knn_factory import knn_factory
from hw3_utils import *
import numpy
from validation import *
from sklearn.model_selection import StratifiedKFold
from PerceptronClassifier import PerceptronClassifierFactory
from Id3Classifier import Id3ClassifierFactory
from timeit import default_timer

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

    # with open('experiments12.csv', 'w') as fp:
    #         acc, err = evaluate(Id3ClassifierFactory(), 2)
    #         fp.write("{}, {}, {}\n".format(1, acc, err))
    #         acc, err = evaluate(PerceptronClassifierFactory(), 2)
    #         fp.write("{}, {}, {}\n".format(2, acc, err))

    ## Competition classifier
    best_classifier = {'acc':0}
    for num_folds in range(2, 10):
        print("Starting loop for {} folds".format(num_folds))
        split_crosscheck_groups([data, labels], num_folds)

        tick = default_timer()
        best_results = get_best(knn_factory(1), num_folds)
        print("{} time units".format(tick - default_timer()))
        print("checked 1-NN. Acc = {}".format(best_results['acc']))
        results = get_best(knn_factory(3), num_folds)
        print("checked 3-NN. Acc = {}".format(results['acc']))
        num_neighbors = 3
        while best_results['acc'] <= results['acc']:
            best_results = results
            num_neighbors += 2
            results = get_best(knn_factory(num_neighbors), num_folds)
            print("checked {}-NN. Acc = {}".format(num_neighbors, results['acc']))
        
        if best_results['acc'] > best_classifier['acc']:
            best_classifier = best_results
            # Current value is the one that broke the loop by being worse
            best_classifier['num_neighbors'] = num_neighbors - 2
            best_classifier['num_folds'] = num_folds
    
    # TODO: Analysis of best classifier (best_classifier['classifier']) and
    #       parameters, or just running the test data through it.

    print(best_classifier['acc'], best_classifier['num_neighbors'], best_classifier['num_folds'], best_classifier['validation'])





    # acc, err = evaluate(knn_factory(3), 2)
    # print("{} {}".format(acc, err))











