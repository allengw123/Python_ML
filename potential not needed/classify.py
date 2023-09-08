import datetime
import json
import glob
import itertools
import math
import pathlib
import pyperclip
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from eeg_utils import *
from env_handler import *
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import svm, linear_model, discriminant_analysis, ensemble, tree, naive_bayes, neighbors, model_selection, metrics
from enum import Enum, auto

from sklearn.tree import export_graphviz

import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def copy_arr(arr):
    pyperclip.copy('\t'.join(map(str, arr)))

def out_dot(model, subj_num, trial_num):
    export_graphviz(
        model,
        out_file=f'output/{subj_num}_{trial_num}_dt.dot',
        feature_names=[
            'C3 Delta', 'C3 Theta', 'C3 Alpha', 'C3 Beta', 'C3 Gamma',
            'C4 Delta', 'C4 Theta', 'C4 Alpha', 'C4 Beta', 'C4 Gamma'
        ],
        class_names=['hold', 'reach'],
        filled=True,
        precision=7
    )

class Y_Value(Enum):
    HOLD = auto()
    REACH = auto()

class Class_Method(Enum):
    SVM = auto()
    LR = auto()
    LDA = auto()
    DT = auto()
    RF = auto()
    NB = auto()
    KNN = auto()
    ADA = auto()

class Feat_Method(Enum):
    PSD = auto()
    PSDBIN = auto()
    DE = auto()
    COH = auto()

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

    classifier = model_selection.GridSearchCV(svm.SVC(probability=True), tuned_parameters)
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

    debug = False

    solver = ['svd', 'lsqr', 'eigen']
    tuned_parameters = {
        'solver': solver
    }

    classifier = model_selection.GridSearchCV(discriminant_analysis.LinearDiscriminantAnalysis(), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros(len(solver))
        for i, j in zip(tested_params, scores):
            scores_array[solver.index(i['solver'])] = j
        plt.figure()
        plt.bar([0, 1, 2], scores_array)
        plt.title('Linear Discriminant Analysis Accuracy Optimization')
        plt.xlabel('Solver')
        plt.xticks([0, 1, 2], solver)
        plt.ylabel('acc')
        plt.show()

    return classifier.best_estimator_, classifier.best_params_


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

    return classifier.best_estimator_, classifier.best_params_

# Random Forest (RF)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# min_samples_leaf empirically determined to be 1
# max_features was more consistently high accuracy as 'auto' rather than 'sqrt'
# max_depth was more consistently high accuracy as 30

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

    return classifier.best_estimator_, classifier.best_params_

# Naive Bayes (NB)
# https://scikit-learn.org/stable/modules/naive_bayes.html
# var_smoothing was empirically tested to have 1e-10 to be the best

def train_nb(X, y):
    """
    """

    debug = False

    var_smoothing = [10**i for i in range(-15, 0, 1)]
    tuned_parameters = {
        'var_smoothing': var_smoothing
    }

    classifier = model_selection.GridSearchCV(naive_bayes.GaussianNB(), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros(len(var_smoothing))
        for i, j in zip(tested_params, scores):
            scores_array[var_smoothing.index(i['var_smoothing'])] = j
        print('var_smoothing min/max:', min(var_smoothing), max(var_smoothing))
        plt.figure()
        plt.plot(np.log10(var_smoothing), scores_array)
        plt.title('Naive Bayes Accuracy Optimization')
        plt.xlabel('var_smoothing (10^x)')
        plt.ylabel('acc')
        plt.show()

    return classifier, classifier.best_params_

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


# Adaboost (Ada)

def train_ada(X, y):
    """
    """

    debug = False

    n_estimators = [i for i in range(200, 401, 25)]
    estimator_names = ['svm', 'nb', 'dt']
    base_estimator = [
            svm.SVC(probability=True, kernel='rbf', C=2**(12.5), gamma=2**15),
            naive_bayes.GaussianNB(),
            tree.DecisionTreeClassifier(max_depth=1)
    ]

    tuned_parameters = {
        'n_estimators': n_estimators,
        'base_estimator': base_estimator
    }

    classifier = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), tuned_parameters)
    classifier.fit(X, y)

    # Training Graph
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(estimator_names)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(i['n_estimators']), base_estimator.index(i['base_estimator'])] = j

        plt.figure()
        plt.xticks([i for i in range(len(base_estimator))], estimator_names)
        plt.contour([i for i in range(len(base_estimator))], n_estimators, scores_array)
        plt.title('AdaBoost Accuracy Optimization')
        plt.xlabel('base_estimator')
        plt.ylabel('n_estimators')
        plt.colorbar()
        plt.show()

    return classifier.best_estimator_, classifier.best_params_


###
# MAIN
###

def main():
    verbose = True
    patient_name = 'pro00087153_0040'
    output_folder = pathlib.Path('output') / patient_name
    with (output_folder / 'file_metadata.json').open('r') as fp:
        metadata = json.load(fp)

    sample_rate = metadata['eeg_sample_rate']
    spec_sample_rate = metadata['spec_sample_rate']
    num_seconds = metadata['seconds']

    specified_electrodes = ['C3', 'C4', 'C3C4']
    specified_features = [Feat_Method.PSDBIN, Feat_Method.PSDBIN, Feat_Method.COH]

    all_x = []
    all_y = []

    if specified_features == Feat_Method.PSD:
        feature_str = 'psd'
    elif specified_features == Feat_Method.PSDBIN:
        feature_str = 'psdbin'
    elif specified_features == Feat_Method.DE:
        feature_str = 'de'
    elif specified_features == Feat_Method.COH:
        feature_str = 'coh'

    print(f"Using feature selection method: {specified_features} with {specified_electrodes}")
    print(f"\ton patient {patient_name}")

    for file in glob.glob(str(output_folder / f'**/**/*_{specified_electrodes[0]}_{feature_str}.csv')):
        _, _, trial, event, filename = pathlib.Path(file).parts

        event_id = filename.split('_')[0]

        if event == 'hold':
            all_y.append(Y_Value.HOLD.value)
        elif event == 'reach':
            all_y.append(Y_Value.REACH.value)
        else:
            continue

        x = []
        for electrode in specified_electrodes:
            x.append(np.loadtxt(output_folder / trial / event / f'{event_id}_{electrode}_{feature_str}.csv', delimiter=','))
        all_x.append(np.hstack(x))

    assert(len(all_x) == len(all_y))
    num_items = len(all_x)

    ##### TODO: Turn this part into for loop and collect accuracies

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
    all_models = []
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
        elif i == Class_Method.ADA:
            model, params = train_ada(train_x, train_y)
        train_stop = datetime.datetime.now()

        if i not in [Class_Method.ADA]:
            all_models.append({'type': i, 'model': model, 'score': model.score(train_x, train_y)})

        #val_start = datetime.datetime.now()
        score = model.score(val_x, val_y)
        #val_stop = datetime.datetime.now()

        val_start = time.time()
        val_res_y = model.predict(val_x)
        val_stop = time.time()
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)

        print(f"Model {i} scored \t{score*100:.2f}% in {train_stop - train_start} training time")
        if verbose:
            print(f"Confusion Matrix:")
            for row in conf_matx:
                print(f"\t{row}")
            print(f"Parameters: {json.dumps(params, indent=2)}")
            print(f"Training time:\t\t{train_stop - train_start}")
            print(f"Validation time:\t{val_stop - val_start}")
            print()
        print()

    if len(all_models) > 0:
        train_start = datetime.datetime.now()
        stack_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=[x['score'] for x in all_models], fit_base_estimators=False)
        stack_model.fit(train_x, train_y)
        train_stop = datetime.datetime.now()

        me_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=[3, 1, 2, 1, 3, 2, 2], fit_base_estimators=False)
        me_model.fit(train_x, train_y)

        global_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=[.72662, .59793, .71133, .72925, .79051, .67535, .72198], fit_base_estimators=False)
        global_model.fit(train_x, train_y)

        uniform_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', fit_base_estimators=False)
        uniform_model.fit(train_x, train_y)

        hard_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='hard', fit_base_estimators=False)
        hard_model.fit(train_x, train_y)

        score = stack_model.score(val_x, val_y)
        val_start = time.time()
        val_res_y = stack_model.predict(val_x)
        val_stop = time.time()
        print(f"Model Ensemble scored \t\t{score * 100:.2f}% in {train_stop - train_start} training time")
        print(f"\t\tWeights used:\t{[x['score'] for x in all_models]}.")
        print(f"\t\tOrder:\t\t{[x['type'].name for x in all_models]}")
        print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        score = me_model.score(val_x, val_y)
        val_start = time.time()
        val_res_y = me_model.predict(val_x)
        val_stop = time.time()
        print(f"Model ME scored \t\t{score * 100:.2f}%")
        print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        score = global_model.score(val_x, val_y)
        val_start = time.time()
        val_res_y = global_model.predict(val_x)
        val_stop = time.time()
        print(f"Model GLOBAL scored \t\t{score * 100:.2f}%")
        print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        score = uniform_model.score(val_x, val_y)
        val_start = time.time()
        val_res_y = uniform_model.predict(val_x)
        val_stop = time.time()
        print(f"Model UNIFORM scored \t\t{score * 100:.2f}%")
        print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        score = hard_model.score(val_x, val_y)
        val_start = time.time()
        val_res_y = hard_model.predict(val_x)
        val_stop = time.time()
        print(f"Model HARD scored \t\t{score * 100:.2f}%")
        print(f"\t\tValidation time: {val_stop - val_start}")
        print()

    return model

if __name__ == '__main__':
    model = main()