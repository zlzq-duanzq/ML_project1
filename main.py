# EECS 445 - Winter 2021
# Project 1 - project1.py

from numpy.core.function_base import logspace
import pandas as pd
import numpy as np
import itertools
import string
import random

from sklearn.metrics.pairwise import kernel_metrics

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *

import spacy
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

def extract_dictionary(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    # TODO: Implement this function
    word_num = 0
    for index, row in df.iterrows():
        text = row['reviewText']
        for p in string.punctuation:
            text = text.replace(p, " ")
        text = text.lower()
        word_list = text.split()
        for word in word_list:
            if isinstance(word_dict.get(word), type(None)):
                word_dict[word] = word_num
                word_num += 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has the ratings and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    for y in range(df.shape[0]):
        text = df.at[y,'reviewText']
        for p in string.punctuation:
            text = text.replace(p, " ")
        text = text.lower()
        word_list = text.split()
        for word in word_list:
            index = word_dict.get(word)
            if isinstance(index, int):
                feature_matrix[y,index] = 1
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    TP = np.float64(0); FP = np.float64(0) 
    FN = np.float64(0); TN = np.float64(0)
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            elif y_pred[i] == -1:
                FN += 1
        elif y_true[i] == -1:
            if y_pred[i] == 1:
                FP += 1
            elif y_pred[i] == -1:
                TN += 1

    if metric == "accuracy":
        return (TP + TN) / (TP + TN + FP + FN)
    elif metric == "precision":
        return TP / (TP + FP)
    elif metric == "sensitivity":
        return TP / (TP + FN)
    elif metric == "specificity":
        return TN / (TN + FP)
    elif metric == "f1-score":
        return 2 * TP / (2 * TP + FP + FN)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful

    #Put the performance of the model on each fold in the scores array
    scores = []
    skf = StratifiedKFold(n_splits=k)
    for train_indices, test_indices in skf.split(X, y):
        clf.fit(X[train_indices], y[train_indices])
        if metric == "auroc":
            y_pred = clf.decision_function(X[test_indices])
            #score = metrics.roc_auc_score(y[test_indices], clf.decision_function(X[test_indices]))
        else:
            y_pred = clf.predict(X[test_indices])
        score = performance(y[test_indices], y_pred, metric)
        scores.append(score)
    #And return the average performance across all fold splits.
    return np.array(scores).mean()

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.

    #return LinearSVC(penalty=penalty, C=c, class_weight=class_weight)
    if degree == 1:
        return SVC(kernel="linear", C=c, class_weight=class_weight)
    return SVC(kernel="poly", C=c, degree=degree, coef0=r, class_weight=class_weight)

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_C_val=0.0
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM
    best_score = 0
    for c in C_range:
        if penalty == "l1":
            clf = LinearSVC(penalty="l1", dual=False, C=c, class_weight="balanced")
        else:
            clf = SVC(kernel="linear", C=c, class_weight="balanced")

        score = cv_performance(clf, X, y, k, metric)
        #print(f"c: {c}, score: {score}")
        if score > best_score:
            best_C_val = c
            best_score = score
    print(f"C: {best_C_val}, Performance: {best_score}")
    return best_C_val


def plot_weight(X,y,penalty,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        if penalty == "l1":
            clf = LinearSVC(penalty=penalty, dual=False, C=c, class_weight="balanced")
        else:
            clf = SVC(kernel="linear", C=c, class_weight="balanced")
        clf.fit(X, y)
        val = np.linalg.norm(clf.coef_.reshape((-1,)), ord=0)
        norm0.append(val)

    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def train_perceptron(X_train, Y_train):
    """
    Takes in an input training data X and labels y and 
    returns a valid decision boundary theta, b found through
    the Perceptron algorithm. If a valid decision boundary 
    can't be found, this function fails to terminate.

    # NOTE: if you use the first 500 points of the dataset 
    # we have provided, this functions should converge
    """

    k = 0
    theta = np.zeros(X_train.shape[1])
    b = 0
    mclf = True
    while mclf:
        mclf = False
        for i in range(len(X_train)):
            if Y_train[i] * (np.dot(theta, X_train[i]) + b) <= 0:
                theta = theta + 0.1 * (Y_train[i] -  np.dot(theta, X_train[i])) * X_train[i]
                b += Y_train[i]
                mclf = True
                k += 1
    return theta, b


def get_random():
    n = random.random()
    n = n * 6 - 3
    n = 10 ** n
    return n

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            param_range: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter values for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance as a pair (C,r)
    """
    best_C_val,best_r_val = 0.0, 0.0
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_score = 0

    for pair in param_range:
        c = pair[0]
        r = pair[1]
    #for i in range(25):
    #    c = get_random()
    #    r = get_random()
        clf = SVC(kernel="poly", degree=2, C=c, coef0=r, class_weight="balanced", gamma='auto')
        score = cv_performance(clf, X, y, k, metric)
        print(c, r, score)
        if score > best_score:
            best_C_val = c
            best_r_val = r
            best_score = score
    print(f"C: {best_C_val}, r: {best_r_val}, Performance: {best_score}")
    return best_C_val, best_r_val

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    all_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]

    # TODO: Questions 2, 3, 4

    Q2(X_train, dictionary_binary)
    Q3_1_d(X_train, Y_train, all_metrics)
    Q3_1_e(X_train, Y_train, X_test, Y_test, all_metrics)
    Q3_1_g(X_train, Y_train)
    Q3_1_h(X_train, Y_train, dictionary_binary)
    Q3_1_i(X_train, Y_train, dictionary_binary)
    Q3_2_b(X_train, Y_train)
    Q3_4_a(X_train, Y_train)
    Q3_4_b(X_train, Y_train)
    Q3_5(X_train, Y_train, X_test, Y_test)
    Q4_1(X_train, Y_train, X_test, Y_test, all_metrics)
    Q4_2(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics)
    Q4_3_a(IMB_features, IMB_labels)
    Q4_3_b(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics)
    Q4_4(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    print("Question 5")
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    Challenge(multiclass_features, multiclass_labels, multiclass_dictionary, heldout_features)

def Q2(X_train, dictionary_binary):
    print(f"2.1 The number of unique words: {X_train.shape[1]}")
    print(f"2.2 The average number of non-zero features per rating: {np.average(np.sum(X_train, axis=1))}")
    print(f"2.3 The word appearing in the most number of reviews: {list(dictionary_binary)[np.argmax(np.sum(X_train, axis=0))]}")

def Q3_1_d(X_train, Y_train, all_metrics):
    print("3.1(d)")
    for str in all_metrics:
        print(f"Performance Measures: {str}")
        select_param_linear(X_train, Y_train, k=5, metric=str, C_range=np.logspace(-3, 3, 7), penalty='l2')

def Performance_helper(clf, X_train, Y_train, X_test, Y_test, all_metrics):
    clf.fit(X_train, Y_train)
    for i in range(6):
        print(f"Performance Measures: {all_metrics[i]}")
        if all_metrics[i] == "auroc":
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        score = performance(Y_test, y_pred, metric=all_metrics[i])
        print(f"Performance: {score}")

def Q3_1_e(X_train, Y_train, X_test, Y_test, all_metrics):
    print("3.1(e)")
    clf = SVC(kernel="linear", C=0.01, class_weight="balanced")
    Performance_helper(clf, X_train, Y_train, X_test, Y_test, all_metrics)

def Q3_1_g(X_train, Y_train):
    print("3.1(g)")
    plot_weight(X_train, Y_train, penalty='l2', C_range=np.logspace(-3, 3, 7))

def Q3_1_h(X_train, Y_train, dictionary_binary):
    print("3.1(h)")
    clf = SVC(kernel="linear", C=0.1, class_weight="balanced")
    clf.fit(X_train, Y_train)
    flat = clf.coef_.flatten()
    flat.sort()
    postive_words = []
    for i in range(1,11):
        index1, index2 = np.where(clf.coef_ == flat[-i])
        word = list(dictionary_binary)[index2[0]]
        print(f"positive {i-1}: {word}, {flat[-i]}")
        postive_words.append(word)
    for i in range(10):
        index1, index2 = np.where(clf.coef_ == flat[i])
        word = list(dictionary_binary)[index2[0]]
        print(f"negative {i}: {word}, {flat[i]}")

def Q3_1_i(X_train, Y_train, dictionary_binary):
    print("3.1(i)")
    postive_words = ["enjoyed", "loved", "great", "hot", "life", "highly", "enjoyable", "liked", "excellent", "look"]
    index = find_review(postive_words, dictionary_binary, X_train, Y_train)
    print(index)

def find_review(postive_words, dictionary_binary, X_train, Y_train):
    postive_words_index = []
    for word in postive_words:
        postive_words_index.append(dictionary_binary[word])
    for i in range(len(Y_train)):
        if Y_train[i] == -1:
            sum = 0
            for index in postive_words_index:
                if X_train[i, index] == 1:
                    sum += 1
                if sum == 3:
                    return i
    return -1

def Q3_2_b(X_train, Y_train):
    print("3.2(b)")
    param_range = [[C, r] for C in np.logspace(-3, 3, 7) for r in np.logspace(-3, 3, 7)]
    select_param_quadratic(X_train, Y_train, k=5, metric="auroc", param_range=param_range)

def Q3_4_a(X_train, Y_train):
    print("3.4(a)")
    select_param_linear(X_train, Y_train, k=5, metric="auroc", C_range=np.logspace(-3, 0, 4), penalty='l1')

def Q3_4_b(X_train, Y_train):
    print("3.4(b)")
    plot_weight(X_train, Y_train, penalty='l1', C_range=np.logspace(-3, 0, 4))


def Q3_5(X_train, Y_train, X_test, Y_test):
    print("3.5")
    theta, b = train_perceptron(X_train, Y_train)
    y_pred = np.sign(X_test.dot(theta) + b)
    val = performance(Y_test, y_pred, metric="accuracy")
    print(f"Accuracy of train_perceptron(): {val}")
    
def Q4_1(X_train, Y_train, X_test, Y_test, all_metrics):
    print("4.1")
    clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:10})
    Performance_helper(clf, X_train, Y_train, X_test, Y_test, all_metrics)

def Q4_2(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics):
    print("4.2")
    clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:1})
    Performance_helper(clf, IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics)

def Q4_3_a(IMB_features, IMB_labels):
    print("4.3(a)")
    best_score = 0.0
    best_weight = 0
    #for ratio in logspace(-3, 3, 7):
    for ratio in range(1,15):
        clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:ratio})
        score = cv_performance(clf, IMB_features, IMB_labels, k=5, metric="accuracy")
        print(f"score: {score} at class weight=1:{ratio}")
        if score > best_score:
            best_score = score
            best_weight = ratio
    print(f"best_score: {best_score}, and its class weight: {best_weight}")

def Q4_3_b(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics):
    print("4.3(b)")
    clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:4})
    Performance_helper(clf, IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, all_metrics)

def Q4_4(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels):
    print("4.4")
    clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:3})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr, tpr, thresholds = roc_curve(IMB_test_labels, y_pred, pos_label = 1)
    roc_auc = auc(fpr, tpr)

    clf = SVC(kernel="linear", C=0.1, class_weight={-1:1, 1:1})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr1, tpr1, thresholds1 = roc_curve(IMB_test_labels, y_pred, pos_label = 1)
    roc_auc1 = auc(fpr1, tpr1)

    plt.plot(fpr, tpr, lw=2, label='ROC curve Wn = 1, Wp = 3 (area = %0.2f)' % roc_auc)
    plt.plot(fpr1, tpr1, lw=2, label='ROC curve Wn = 1, Wp = 1 (area = %0.2f)' % roc_auc1)
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def Challenge(multiclass_features, multiclass_labels, multiclass_dictionary, heldout_features):

    print("linear, df_shape='ovo'")
    multiclass_select_param_linear(multiclass_features, multiclass_labels, k=5, C_range=np.logspace(-3, 3, 7), df_shape='ovo')
    print("linear, df_shape='ovr'")
    multiclass_select_param_linear(multiclass_features, multiclass_labels, k=5, C_range=np.logspace(-3, 3, 7), df_shape='ovr')

    param_range = [[C, r] for C in np.logspace(-3, 3, 7) for r in np.logspace(-3, 3, 7)]
    print("quadratic, df_shape='ovo'")
    multiclass_select_param_quadratic(multiclass_features, multiclass_labels, k=5, param_range=param_range, df_shape='ovo')
    print("quadratic, df_shape='ovr'")
    multiclass_select_param_quadratic(multiclass_features, multiclass_labels, k=5, param_range=param_range, df_shape='ovr')

    clf = SVC(kernel="linear", C=0.1, class_weight="balanced")
    score = multiclass_cv_performance(clf, multiclass_features, multiclass_labels, df_shape='ovo')
    print(f"c=0.1, 'ovo', score: {score}")
    clf = SVC(kernel="linear", C=0.01, class_weight="balanced")
    score = multiclass_cv_performance(clf, multiclass_features, multiclass_labels, df_shape='ovr')
    print(f"c=0.01, 'ovr', score: {score}")
    print(multiclass_features.shape)
    clf = LinearSVC(C=0.01, class_weight="balanced", dual=False)
    score = multiclass_cv_performance(clf, multiclass_features, multiclass_labels, df_shape='ovo')
    print(f"c=0.01, 'ovo', score: {score}")
    clf = LinearSVC(C=0.01, class_weight="balanced", dual=False)
    score = multiclass_cv_performance(clf, multiclass_features, multiclass_labels, df_shape='ovr')
    print(f"c=0.01, 'ovr', score: {score}")
    
    y_pred = OneVsRestClassifier(SVC(kernel="linear", C=0.01, class_weight="balanced")).fit(multiclass_features, multiclass_labels).predict(heldout_features)
    generate_challenge_labels(y_pred, "duanzq")

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_wordnet_pos(pos_tag):
    tag = pos_tag[0].upper()
    dict_tag = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return dict_tag.get(tag, wordnet.NOUN)

def convert_text(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    text_list = []
    stop = stopwords.words('english')
    for word in text.split():
        if word not in stop:
                text_list.append(word)
    lemmatizer = WordNetLemmatizer()
    #pos_tags = pos_tag(text_list)
    text_list = [lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tag(text_list)]
    text_list = [word for word in text_list if len(word) > 1]
    return text_list

def extract_dictionary2(df):
    word_dict = {}
    word_num = 0
    for index, row in df.iterrows():
        text = row['reviewText']
        text_list = convert_text(text) 
        for word in text_list:
            if isinstance(word_dict.get(word), type(None)):
                word_dict[word] = word_num
                word_num += 1
    print(len(word_dict))
    return word_dict


def generate_feature_matrix2(df, word_dict):
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for y in range(df.shape[0]):
        text = df.at[y,'reviewText']
        text_list = convert_text(text) 
        for word in text_list:
            index = word_dict.get(word)
            if isinstance(index, int):
                feature_matrix[y,index] = 1
                #feature_matrix[y,index] += 1
    #feature_matrix /= feature_matrix.sum(axis=1)[:,np.newaxis]
    #feature_matrix /= np.linalg.norm(feature_matrix, axis=1)[:,np.newaxis]
    return feature_matrix

def generate_feature_matrix3(df, word_dict):
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for y in range(df.shape[0]):
        review_text = df.at[y,'reviewText']
        text_list = convert_text(review_text)
        for word in text_list:
            index = word_dict.get(word)
            if isinstance(index, int):
                feature_matrix[y,index] += 1
        summary_text = df.at[y,'summary']
        text_list = convert_text(summary_text)
        for word in text_list:
            index = word_dict.get(word)
            if isinstance(index, int):
                feature_matrix[y,index] += 4
    return feature_matrix

def generate_feature_matrix4(df, word_dict):
    review = df['reviewText']
    sid = SentimentIntensityAnalyzer()
    sentiments = review.apply(lambda x: sid.polarity_scores(x))

    sentiments_array = np.zeros((len(sentiments),4))
    for i in range(len(sentiments)):
        sentiments_array[i][0] = sentiments[i]['compound']
        sentiments_array[i][1] = sentiments[i]['neg']
        sentiments_array[i][2] = sentiments[i]['neu']
        sentiments_array[i][3] = sentiments[i]['pos']

    feature_matrix = generate_feature_matrix3(df, word_dict)
    multiclass_features = np.concatenate((sentiments_array * 30, feature_matrix),axis=1)
    return multiclass_features


def multiclass_accuracy(y_true, y_pred):
    count = 0
    n = len(y_true)
    for i in range(n):
        if y_true[i] == y_pred[i]:
            count += 1
    return count / n

def multiclass_cv_performance(clf, X, y, k=5, df_shape='ovo'):
    scores = []
    skf = StratifiedKFold(n_splits=k)
    if df_shape == 'ovo':
        ovx_clf = OneVsOneClassifier(clf)
    else:
        ovx_clf = OneVsRestClassifier(clf)
        
    for train_indices, test_indices in skf.split(X, y):
        ovx_clf.fit(X[train_indices], y[train_indices])
        y_pred = ovx_clf.predict(X[test_indices])
        score = multiclass_accuracy(y[test_indices], y_pred)
        scores.append(score)
    return np.array(scores).mean()

def multiclass_select_param_linear(X, y, k=5, C_range = [], df_shape='ovo'):
    best_C_val = 0.0
    best_score = 0.0
    for c in C_range:
        clf = LinearSVC(C=c, class_weight="balanced", dual=False)
        #clf = SVC(kernel="linear", C=c, class_weight="balanced")
        score = multiclass_cv_performance(clf, X, y, k, df_shape)
        print(f"c: {c}, score: {score}")
        if score > best_score:
            best_C_val = c
            best_score = score
    print(f"best C: {best_C_val}, Performance: {best_score}")
    return best_C_val

def multiclass_select_param_quadratic(X, y, k=5, param_range=[], df_shape='ovo'):
    best_C_val, best_r_val = 0.0, 0.0
    best_score = 0.0
    for pair in param_range:
        c = pair[0]
        r = pair[1]
        clf = SVC(kernel="poly", degree=2, C=c, coef0=r, class_weight="balanced", gamma='auto')
        score = multiclass_cv_performance(clf, X, y, k, df_shape)
        print(f"c: {c}, r: {r}, score: {score}")
        if score > best_score:
            best_C_val = c
            best_r_val = r
            best_score = score
    print(f"best C: {best_C_val}, r: {best_r_val}, Performance: {best_score}")
    return best_C_val, best_r_val


if __name__ == '__main__':
    main()
