#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 07:58:12 2023

@author: rowlandlabbci
"""

import pyperclip
from enum import Enum, auto
import re
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from eeg_utils import *
from sklearn import svm, linear_model, discriminant_analysis, ensemble, tree, naive_bayes, neighbors, model_selection, metrics
from sklearn.tree import export_graphviz


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

    classifier = model_selection.GridSearchCV(
        svm.SVC(probability=True, cache_size=7000), tuned_parameters, n_jobs=-1)
    classifier.fit(X, y)

    # Training Contour:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(gamma), len(C)))
        for i, j in zip(tested_params, scores):
            scores_array[gamma.index(i['gamma']), C.index(i['C'])] = j
        print('C min/max:', min(C), max(C),
              'gamma min/max:', min(gamma), max(gamma))
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

    classifier = model_selection.GridSearchCV(
        linear_model.LogisticRegression(max_iter=1024), tuned_parameters, n_jobs=-1)
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

    classifier = model_selection.GridSearchCV(
        discriminant_analysis.LinearDiscriminantAnalysis(), tuned_parameters, n_jobs=-1)
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

    classifier = model_selection.GridSearchCV(
        tree.DecisionTreeClassifier(min_weight_fraction_leaf=0), tuned_parameters, n_jobs=-1)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        print(tested_params)
        print(scores)

        scores_array = np.zeros((len(min_samples_split), len(max_depth)))
        for i, j in zip(tested_params, scores):
            scores_array[min_samples_split.index(
                i['min_samples_split']), max_depth.index(i['max_depth'])] = j

        print(
            f'max_depth: {min(max_depth)}:{max(max_depth)}, min_samples_split: {min(min_samples_split)}:{max(min_samples_split)}')
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

    # changed max_features to 'sqrt' due to deprecation warning - Rishi
    classifier = model_selection.GridSearchCV(ensemble.RandomForestClassifier(
        min_samples_leaf=1, max_features='sqrt', max_depth=30), tuned_parameters, n_jobs=-1)
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(min_samples_split)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(
                i['n_estimators']), min_samples_split.index(i['min_samples_split'])] = j

        print(
            f'min_samples_split: {min(min_samples_split[1:])}:{max(min_samples_split[1:])}, n_estimators: {min(n_estimators)}:{max(n_estimators)}')
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

    classifier = model_selection.GridSearchCV(
        naive_bayes.GaussianNB(), tuned_parameters, n_jobs=-1)
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

    classifier = model_selection.GridSearchCV(
        neighbors.KNeighborsClassifier(), tuned_parameters, n_jobs=-1)
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
    # estimator_names = ['svm', 'nb', 'dt']
    estimator_names = ['nb', 'dt']
    estimator = [
        # svm.SVC(probability=True, kernel='rbf', C=2**(12.5), gamma=2**15),
        naive_bayes.GaussianNB(),
        tree.DecisionTreeClassifier(max_depth=1)
    ]

    tuned_parameters = {
        'n_estimators': n_estimators,
        'estimator': estimator
    }

    classifier = model_selection.GridSearchCV(
        ensemble.AdaBoostClassifier(), tuned_parameters, n_jobs=-1, error_score='raise')
    classifier.fit(X, y)

    # Training Graph
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(estimator_names)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(
                i['n_estimators']), estimator.index(i['estimator'])] = j

        plt.figure()
        plt.xticks([i for i in range(len(estimator))], estimator_names)
        plt.contour([i for i in range(len(estimator))],
                    n_estimators, scores_array)
        plt.title('AdaBoost Accuracy Optimization')
        plt.xlabel('estimator')
        plt.ylabel('n_estimators')
        plt.colorbar()
        plt.show()

    return classifier.best_estimator_, classifier.best_params_


# XGBoost (XG)

def train_xg(X, y):
    debug = False

    n_estimators = [i for i in range(50, 401, 25)]
    max_depth = [i for i in range(5, 31, 5)]

    # trying different tree methods for faster performance - Rishi
    tree_method = ["hist"]
    tuned_parameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'tree_method': tree_method
    }

    # changes

    # le = LabelEncoder()
    # y = le.fit_transform(y)

    classifier = model_selection.GridSearchCV(
        xgb.XGBClassifier(eval_metric='logloss'), tuned_parameters, n_jobs=-1, error_score='raise')
    classifier.fit(X, y)

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(max_depth)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(
                i['n_estimators']), max_depth.index(i['max_depth'])] = j

        print(
            f'max_depth: {min(max_depth[1:])}:{max(max_depth[1:])}, n_estimators: {min(n_estimators)}:{max(n_estimators)}')
        plt.figure()
        plt.contour(max_depth, n_estimators, scores_array)
        plt.title('Random Forest Accuracy Optimization')
        plt.xlabel('max_depth')
        plt.ylabel('n_estimators')
        plt.colorbar()
        plt.show()

    return classifier.best_estimator_, classifier.best_params_

##### Addtional functions ###


def process_subject(subject,
                    patient_dir,
                    training_trial_idx,
                    phases,
                    specified_features,
                    specified_electrodes,
                    all_patients,
                    val_size):

    train_x, train_y, val_x, val_y = [], [], [], []
    wk_trial_folder = sorted(glob.glob(os.path.join(
        patient_dir, subject, '*')))[training_trial_idx]

    for phase in phases:
        subject_features_csv = glob.glob(os.path.join(
            wk_trial_folder, phase, '*' + specified_features+'.csv'))

        for electrode in specified_electrodes:
            electrode_csv = [x for x in subject_features_csv if electrode in x]
            feature_val = []
            for csv in electrode_csv:
                feature_data = np.loadtxt(csv, delimiter=',')
                # delta only
                feature_data = feature_data[0]
                if len(feature_data.shape) == 1:
                    feature_data = np.append(
                        feature_data, specified_electrodes.index(electrode))
                    # feature_data = np.append(
                    # feature_data, getattr(Feat_Method, phase).value)
                    feature_val.append(feature_data)
                elif len(feature_data.shape) == 2:
                    electrode_data = [
                        specified_electrodes.index(electrode)] * len(feature_data)
                    # phase_data = [
                    # getattr(Feat_Method, phase).value] * len(feature_data)
                    feature_data = [np.concatenate((arr, np.array([x])), axis=0) for arr, x in zip(
                        feature_data, electrode_data)]
                    feature_val.extend(feature_data)

            if len(feature_val) == 0:
                break
            '''disease_label = [
                key for key, values in all_patients.items() if subject in values][0]
            feature_label = [
                getattr(Disease_State, disease_label).value] * len(feature_val)'''
            '''stim_list = ["pro00087153_0022", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026", "pro00087153_0029",
                         "pro00087153_0030", "pro00087153_0003", "pro00087153_0004", "pro00087153_0005", "pro00087153_0042", "pro00087153_0043"]
            stim_label = "STIM" if subject in stim_list else "SHAM"'''

            # phase labels
            if(phase == "hold"):
                phase_label = 1
            elif (phase == 'reach'):
                phase_label = 2

            feature_label = [phase_label] * len(feature_val)

            sbj_x_train, sbj_x_val, sbj_y_train, sbj_y_val = train_test_split(
                feature_val, feature_label, test_size=val_size, random_state=42)

            train_x.extend(sbj_x_train)
            train_y.extend(sbj_y_train)
            val_x.extend(sbj_x_val)
            val_y.extend(sbj_y_val)
    return train_x, train_y, val_x, val_y


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


class Disease_State(Enum):
    HEALTHY = auto()
    STROKE = auto()


class Stimulation_State(Enum):
    STIM = auto()
    SHAM = auto()


class Disease_State_Group(Enum):
    HC_Sham = 1
    HC_Stim = 1
    CS_Sham = 2
    CS_Stim = 2


class Class_Method(Enum):
    # SVM = auto()
    LR = auto()
    LDA = auto()
    DT = auto()
    RF = auto()
    NB = auto()
    KNN = auto()
    ADA = auto()
    XG = auto()


class Feat_Method(Enum):
    psd = auto()
    psdbin = auto()
    de = auto()
    coh = auto()
    cohbin = auto()
    reach = auto()
    hold = auto()
    prep = auto()


class EEG_Ch_val(Enum):

    Fp1 = auto()
    F7 = auto()
    T3 = auto()
    T5 = auto()
    O1 = auto()
    F3 = auto()
    C3 = auto()
    P3 = auto()
    A1 = auto()
    Fz = auto()
    Cz = auto()
    Fp2 = auto()
    F8 = auto()
    T4 = auto()
    T6 = auto()
    O2 = auto()
    F4 = auto()
    C4 = auto()
    P4 = auto()
    A2 = auto()
    Fpz = auto()
    Pz = auto()
