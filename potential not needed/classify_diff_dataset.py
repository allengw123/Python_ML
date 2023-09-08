import datetime
import json
import glob
import math
from re import A
import mne
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np

from eeg_utils import *
from env_handler import *
from scipy import signal, stats
from sklearn import svm, linear_model, discriminant_analysis, ensemble, tree, naive_bayes, neighbors, model_selection, metrics
from enum import Enum, auto

class Y_Value(Enum):
    HOLD = 0
    MOVE_L = 1
    MOVE_R = 2

class Class_Method(Enum):
    SVM = auto()
    LR = auto()
    LDA = auto()
    DT = auto()
    RF = auto()
    NB = auto()
    KNN = auto()

class Feat_Method(Enum):
    PSD = auto()
    PSDBIN = auto()
    DE = auto()

###
# Classification Methods
###

# Support Vector Machine
# https://scikit-learn.org/stable/modules/svm.html


def train_svm(X: np.array, y: np.array):
    """Create and train an SVM model.

    Keyword arguments:
    X:  The input/feature data. Each row is a sample and each column is a feature.
    y:  The output data. Each element i is a result corresponding to the ith row in X.
        Note: Should only have one dimension
    """
    debug = False

    gamma = [2**i for i in range(-12, 25, 1)]
    C = [2**i for i in range(0, 21, 1)]
    tuned_parameters = {
        'kernel': ['rbf'],
        'gamma': gamma,
        'C': C
    }

    classifier = model_selection.GridSearchCV(svm.SVC(), tuned_parameters)
    classifier.fit(X, y)

    # Training Contour:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(gamma), len(C)))
        for i, j in zip(tested_params, scores):
            scores_array[gamma.index(i['gamma']), C.index(i['C'])] = j
        print('C min/max:', min(C), max(C), 'gamma min/max:', min(gamma), max(gamma))
        plt.figure()
        plt.contour(np.log2(C), np.log2(gamma), scores_array)
        plt.title('Support Vector Machine Accuracy Optimization')
        plt.xlabel('C (2^x)')
        plt.ylabel('gamma (2^y)')
        plt.colorbar()
        plt.show()

    return classifier, classifier.best_params_

# Linear Regression (LR)
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

def train_lr(X: np.array, y: np.array):
    """Create and train an LR model.

    Keyword arguments:
    X:  The input/feature data. Each row is a sample and each column is a feature.
    y:  The output data. Each element i is a result corresponding to the ith row in X.
        Note: Should only have one dimension
    """
    debug = False

    C = [2**i for i in range(15, 35, 1)]
    tuned_parameters = {
        'C': C
    }

    classifier = model_selection.GridSearchCV(linear_model.LogisticRegression(max_iter=1024), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros(len(C))
        for i, j in zip(tested_params, scores):
            scores_array[C.index(i['C'])] = j
        print('C min/max:', min(C), max(C))
        plt.figure()
        plt.plot(np.log2(C), scores_array)
        plt.title('Logistic Regression Accuracy Optimization')
        plt.xlabel('C (2^x)')
        plt.ylabel('acc')
        plt.show()

    return classifier, classifier.best_params_

# Linear Discriminant Analysis (LDA)
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
# It was empirically determined that LDA's default svd solver was sufficient for quality results.

def train_lda(X: np.array, y: np.array):
    """
    """

    solver = 'svd' # 'svd', 'lsqr', 'eigen'
    classifier = discriminant_analysis.LinearDiscriminantAnalysis(solver=solver)
    classifier.fit(X, y)

    return classifier, classifier.get_params()

# Decision Trees (DT)
# https://scikit-learn.org/stable/modules/tree.html
# min_weight_fraction_leaf empirically driven to be 0
# min_samples_split needs to be somewhere between 2 and 10.
# max_depth needs to be somewhere between 2 and 10

def train_dt(X, y):
    """
    """

    debug = False

    min_samples_split = [i for i in range(2, 11, 1)]
    max_depth = [i for i in range(2, 30, 1)]
    tuned_parameters = {
        'min_samples_split': min_samples_split,
        'max_depth': max_depth
    }

    classifier = model_selection.GridSearchCV(tree.DecisionTreeClassifier(min_weight_fraction_leaf=0), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros((len(min_samples_split), len(max_depth)))
        for i, j in zip(tested_params, scores):
            scores_array[min_samples_split.index(i['min_samples_split']), max_depth.index(i['max_depth'])] = j

        print(f'max_depth: {min(max_depth)}:{max(max_depth)}, min_samples_split: {min(min_samples_split)}:{max(min_samples_split)}')
        plt.figure()
        plt.contour(max_depth, min_samples_split, scores_array)
        plt.title('Decision Tree Accuracy Optimization')
        plt.xlabel('max_depth')
        plt.ylabel('min_samples_split')
        plt.colorbar()
        plt.show()

    return classifier, classifier.best_params_

# Random Forest (RF)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# min_samples_leaf empirically determined to be 1
# max_features was more consistently high accuracy as 'auto' rather than 'sqrt'
# max_depth was more consistently high accuracy as max_depth

def train_rf(X, y):
    """
    """
    debug = False

    min_samples_split = [i for i in range(2, 5)]
    n_estimators = [i for i in range(25, 105, 5)]

    tuned_parameters = {
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }

    classifier = model_selection.GridSearchCV(ensemble.RandomForestClassifier(min_samples_leaf=1, max_features='auto', max_depth=30), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(min_samples_split)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(i['n_estimators']), min_samples_split.index(i['min_samples_split'])] = j

        print(f'min_samples_split: {min(min_samples_split[1:])}:{max(min_samples_split[1:])}, n_estimators: {min(n_estimators)}:{max(n_estimators)}')
        plt.figure()
        plt.contour(min_samples_split, n_estimators, scores_array)
        plt.title('Random Forest Accuracy Optimization')
        plt.xlabel('min_samples_split')
        plt.ylabel('n_estimators')
        plt.colorbar()
        plt.show()

    return classifier, classifier.best_params_

# Naive Bayes (NB)
# https://scikit-learn.org/stable/modules/naive_bayes.html
# var_smoothing was empirically tested to have 1e-10 to be the best

def train_nb(X, y):
    """
    """

    classifier = naive_bayes.GaussianNB(var_smoothing=1e-10)
    classifier.fit(X, y)

    return classifier, classifier.get_params()

# K-Nearest Neighbors (KNN)
# https://scikit-learn.org/stable/modules/neighbors.html

def train_knn(X, y):
    """
    """

    debug = False

    n_neighbors = [i for i in range(3, 10)]
    tuned_parameters = {
        'n_neighbors': n_neighbors
    }

    classifier = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros(len(n_neighbors))
        for i, j in zip(tested_params, scores):
            scores_array[n_neighbors.index(i['n_neighbors'])] = j

        print(f'n_neighbors: {min(n_neighbors)}:{max(n_neighbors)}')
        plt.figure()
        plt.plot(n_neighbors, scores_array)
        plt.title('K-Nearest Neighbors Accuracy Optimization')
        plt.xlabel('n_neighbors')
        plt.ylabel('acc')
        plt.show()

    return classifier, classifier.best_params_

###
# MAIN
###

def main():
    mne.set_log_level(verbose='WARNING')

    # Load in data
    all_x = [] # each entry is a 2D array where rows are samples and columns are channels
    all_y = [] # each entry is a 1D array [P(hold) P(reach)]

    data_folder = pathlib.Path('C:/Users/maura/Downloads/eeg_db')
    for subj_num in [3]:
        for task_num in [3, 7, 11]:
            edf = mne.io.read_raw_edf(data_folder / f'S{subj_num:03}' / f'S{subj_num:03}R{task_num:02}.edf', preload=True).filter(5, 50)
            data = edf.get_data(['C3..', 'C4..'])
            fs = edf.info['sfreq']
            for anno in edf.annotations: # 'onset', 'duration', 'description'
                start = int(anno['onset'] * fs)
                length = int(4 * fs)
                anno_data = data[:, start:start+length]
                anno_data = (anno_data - np.mean(anno_data)) / np.std(anno_data)
                psd_bin = []
                for ch in anno_data:
                    psd = signal.welch(ch, fs=edf.info['sfreq'], nfft=edf.info['sfreq'], nperseg=edf.info['sfreq'])
                    psd_delta = np.sum(psd[1][1:4])
                    psd_theta = np.sum(psd[1][4:8])
                    psd_alpha = np.sum(psd[1][8:13])
                    psd_beta  = np.sum(psd[1][13:31])
                    psd_gamma = np.sum(psd[1][31:51])
                    psd_bin.extend([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])
                psd_bin = np.array(psd_bin)

                if anno['description'] == 'T0':
                    all_y.append(Y_Value.HOLD.value)
                elif anno['description'] == 'T1':
                    all_y.append(Y_Value.MOVE_L.value)
                elif anno['description'] == 'T2':
                    all_y.append(Y_Value.MOVE_R.value)

                all_x.append(psd_bin)


    assert len(all_x) == len(all_y)
    num_items = len(all_x)

    ###
    # Separate into training and validation
    ###

    num_train = math.floor(num_items * 0.7) # 70/30
    num_val = num_items - num_train
    print(f"Number of training data: {num_train}; Number of validation data: {num_val}")
    val_indices = random.sample(list(range(num_items)), num_val)
    train_x, train_y, val_x, val_y = [], [], [], []
    for i in range(num_items):
        if i in val_indices:
            val_x.append(all_x[i])
            val_y.append(all_y[i])
        else:
            train_x.append(all_x[i])
            train_y.append(all_y[i])

    # At this point, train_x and val_x should contain the correct features
    # In order to generate the correct features, use output_features.py
    for i in Class_Method:
        train_start = datetime.datetime.now()
        if i == Class_Method.SVM:
            model, params = train_svm(train_x, train_y)
        elif i == Class_Method.LR:
            model, params = train_lr(train_x, train_y)
        elif i == Class_Method.LDA:
            model, params = train_lda(train_x, train_y)
        elif i == Class_Method.DT:
            model, params = train_dt(train_x, train_y)
        elif i == Class_Method.RF:
            model, params = train_rf(train_x, train_y)
        elif i == Class_Method.NB:
            model, params = train_nb(train_x, train_y)
        elif i == Class_Method.KNN:
            model, params = train_knn(train_x, train_y)
        train_stop = datetime.datetime.now()

        val_start = datetime.datetime.now()
        score = model.score(val_x, val_y)
        val_stop = datetime.datetime.now()

        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)

        print(f"Model {i} scored \t{score*100:.2f}%")
        print(f"Confusion Matrix:")
        for row in conf_matx:
            print(f"\t{row}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print(f"Training time:\t\t{train_stop - train_start}")
        print(f"Validation time:\t{val_stop - val_start}")
        print()
        print()

if __name__ == '__main__':
    main()