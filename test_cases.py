# EECS 445 - Winter 2021
# Project 1 - test_cases.py

import pandas as pd
import numpy as np

from project1 import *

def test_dictionary():
    """
    Test case for extract_dictionary()
    """
    print('TESTING EXTRACT DICTIONARY')
    X_train = pd.DataFrame({'reviewText': ['BEST book ever! It\'s great'],
                            'summary': ['NA'],
                            'unix_review_time': [0],
                            'helpful': [1],
                            'unhelpful': [0],
                            'rating': [4],
                            'label': [1]})

    expected_dictionary = {'best': 0, 'book': 1, 'ever': 2, 'it': 3, 's': 4, 'great': 5}
    dictionary = extract_dictionary(X_train)

    print('EXPECTED OUTPUT:\t' + str(expected_dictionary))
    print('STUDENT OUTPUT: \t' + str(dictionary))  
    assert dictionary == expected_dictionary, 'OUTPUTS DIFFER - test_feature_matrix()'
    print('SUCCESS')

    return dictionary


def test_feature_matrix(dictionary):
    """
    Test case for generate_feature_matrix()
    """
    print('TESTING GENERATE FEATURE MATRIX')
    X_test = pd.DataFrame({ 'reviewText': ['Markley has the best books! The one about Tendie Fridays was one of the best I have EVER read'],
                            'summary': ['NA'],
                            'unix_review_time': [0],
                            'helpful': [999],
                            'unhelpful': [0],
                            'rating': [5],
                            'label': [1]})

    expected_feature_matrix = np.array([[1., 0., 1., 0., 0., 0.]])
    feature_matrix = generate_feature_matrix(X_test, dictionary)

    print('EXPECTED OUTPUT:\t' + str(expected_feature_matrix))
    print('STUDENT OUTPUT: \t' + str(feature_matrix))
    assert np.array_equal(feature_matrix, expected_feature_matrix), 'OUTPUTS DIFFER - test_feature_matrix()'
    print('SUCCESS')


def test_select_param_linear():
    """
    Test case for select_param_linear()
    """
    print('TESTING SELECT PARAM LINEAR ON SMALL DATASET')
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(10)
    C_range = np.logspace(-5, 2, 8) # These are not the same values as used in 3.2!
    expected_best_C = 0.1

    print('EXPECTED SENSITIVITY SCORES:')
    print('c: 0.000010 score: 0.1000\n'
            'c: 0.000100 score: 0.1000\n'
            'c: 0.001000 score: 0.1000\n'
            'c: 0.010000 score: 0.1000\n'
            'c: 0.100000 score: 0.6000\n'
            'c: 1.000000 score: 0.6000\n'
            'c: 10.000000 score: 0.6000\n'
            'c: 100.000000 score: 0.6000\n')

    print('\nRunning select_param_linear()...')
    best_C = select_param_linear(X_train, Y_train, 5, 'sensitivity', C_range, 'l2')

    print('EXPECTED OUTPUT:\t' + str(expected_best_C))
    print('STUDENT OUTPUT: \t' + str(best_C))
    assert expected_best_C == best_C, 'OUTPUTS DIFFER - test_feature_matrix()'
    print('SUCCESS')


def test_select_param_quadratic():
    """
    Test case for select_param_quadratic()
    """
    print('TESTING SELECT PARAM QUADRATIC ON SMALL DATASET')
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(20)
    # Grid search in {1e-1, 1e-2, 1e-3, 1e-4}^2
    param_range = [[C, r] for C in np.logspace(-3, -1, 3) for r in np.logspace(-3, -1, 3)]
    expected_best_C = 0.001
    expected_best_r = 0.01

    print('EXPECTED AUROC SCORES:')
    print('c: 0.001000 r: 0.001000 score: 0.6625\n'
            'c: 0.001000 r: 0.010000 score: 0.7000\n'
            'c: 0.001000 r: 0.100000 score: 0.7000\n'
            'c: 0.010000 r: 0.001000 score: 0.6625\n'
            'c: 0.010000 r: 0.010000 score: 0.7000\n'
            'c: 0.010000 r: 0.100000 score: 0.7000\n'
            'c: 0.100000 r: 0.001000 score: 0.6625\n'
            'c: 0.100000 r: 0.010000 score: 0.7000\n'
            'c: 0.100000 r: 0.100000 score: 0.7000')

    print('\nRunning select_param_quadratic()...')
    best_C, best_r = select_param_quadratic(X_train, Y_train, 5, 'auroc', param_range)

    print('EXPECTED OUTPUT:\tc: ' + str(expected_best_C) + ' r: ' + str(expected_best_r))
    print('STUDENT OUTPUT: \tc: ' + str(best_C) + ' r: ' + str(best_r))
    assert expected_best_C == best_C and expected_best_r == best_r, 'OUTPUTS DIFFER - test_feature_matrix()'
    print('SUCCESS')


def main():
    dictionary = test_dictionary()
    print('----------------')
    test_feature_matrix(dictionary)
    print('----------------')
    test_select_param_linear()
    print('----------------')
    test_select_param_quadratic()



if __name__ == '__main__':
    main()
