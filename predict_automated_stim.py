#import beepy
import datetime
import json
import glob
import itertools
import math
import pathlib
import pyperclip
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from eeg_utils import *
from env_handler import *
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import svm, linear_model, discriminant_analysis, ensemble, tree, naive_bayes, neighbors, model_selection, metrics
from enum import Enum, auto

from sklearn.tree import export_graphviz

import warnings
from sklearn.exceptions import FitFailedWarning

#import changes
from sklearn.preprocessing import LabelEncoder

import pickle
import csv
import os

warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

#changes
LOG_FILE_DIRECTORY = ""

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
    PREP = auto()

class Disease_State(Enum):
    HEALTHY = auto()
    STROKE = auto()

class Stim_State(Enum):
    STIM = auto()
    SHAM = auto()

class Class_Method(Enum):
    #SVM = auto()
    LR = auto()
    LDA = auto()
    DT = auto()
    RF = auto()
    NB = auto()
    KNN = auto()
    ADA = auto()
    XG = auto()

class Feat_Method(Enum):
    PSD = auto()
    PSDBIN = auto()
    DE = auto()
    COH = auto()
    COHBIN = auto()

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

    classifier = model_selection.GridSearchCV(svm.SVC(probability=True, cache_size=7000), tuned_parameters)
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

    # changed max_features to 'sqrt' due to deprecation warning - Rishi
    classifier = model_selection.GridSearchCV(ensemble.RandomForestClassifier(min_samples_leaf=1, max_features='sqrt', max_depth=30), tuned_parameters)
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
    #estimator_names = ['svm', 'nb', 'dt']
    estimator_names = ['nb', 'dt']
    base_estimator = [
            #svm.SVC(probability=True, kernel='rbf', C=2**(12.5), gamma=2**15),
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


# XGBoost (XG)

def train_xg(X, y):
    debug = False

    n_estimators = [i for i in range(50, 401, 25)]
    max_depth = [i for i in range(5, 31, 5)]

    #trying different tree methods for faster performance - Rishi
    tree_method = ["hist"]
    tuned_parameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'tree_method': tree_method
    }

    # changes
    
    #le = LabelEncoder()
    #y = le.fit_transform(y)

    classifier = model_selection.GridSearchCV(xgb.XGBClassifier(), tuned_parameters)
    classifier.fit(X, y, eval_metric='logloss')

    # Training Graph:
    if debug:
        tested_params = classifier.cv_results_['params']
        scores = classifier.cv_results_['mean_test_score']

        scores_array = np.zeros((len(n_estimators), len(max_depth)))
        for i, j in zip(tested_params, scores):
            scores_array[n_estimators.index(i['n_estimators']), max_depth.index(i['max_depth'])] = j

        print(f'max_depth: {min(max_depth[1:])}:{max(max_depth[1:])}, n_estimators: {min(n_estimators)}:{max(n_estimators)}')
        plt.figure()
        plt.contour(max_depth, n_estimators, scores_array)
        plt.title('Random Forest Accuracy Optimization')
        plt.xlabel('max_depth')
        plt.ylabel('n_estimators')
        plt.colorbar()
        plt.show()

    return classifier.best_estimator_, classifier.best_params_


###
# MAIN
###

def main():
    verbose = False

    #state = "intra5" # pre | intra5 | intra15 | post
    #new_output_folder_path = "output/contralateral_C3/post"

    #create loop through patients here
    HC_Sham_pts = ["HC_Sham", ["pro00087153_0020", "pro00087153_0023", "pro00087153_0027", "pro00087153_0028", "pro00087153_0036"]]
    HC_Stim_pts = ["HC_Stim", ["pro00087153_0022", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026", "pro00087153_0029", "pro00087153_0030"]]
    CS_Sham_pts = ["CS_Sham", ["pro00087153_0013", "pro00087153_0015", "pro00087153_0017", "pro00087153_0018", "pro00087153_0021"]]
    CS_Stim_pts = ["CS_Stim", ["pro00087153_0003", "pro00087153_0004", "pro00087153_0005", "pro00087153_0042", "pro00087153_0043"]]

    C3_contra_pts = ["pro00087153_0003", "pro00087153_0004", "pro00087153_0043", "pro00087153_0015", "pro00087153_0024",
     "pro00087153_0025", "pro00087153_0026", "pro00087153_0029", "pro00087153_0020", "pro00087153_0027", "pro00087153_0028",
     "pro00087153_0036"]
    
    C4_contra_pts = ["pro00087153_0005", "pro00087153_0042", "pro00087153_0013", "pro00087153_0017", "pro00087153_0018", "pro00087153_0021",
    "pro00087153_0022", "pro00087153_0030", "pro00087153_0023"]

    #laterality = "ipsi"

    #specified_electrodes = ['C4']
    #specified_features = [Feat_Method.PSDBIN]

    specified_electrodes = ['C3', 'C4', 'T3', 'T4', 'Fp1', 'F7', 'T3', 'T5', 'O1',
                            'F3', 'P3', 'A1', 'Fz', 'Cz', 'Fp2', 'F8', 'T4', 'T6',
                            'O2', 'F4', 'P4', 'A2', 'Fpz', 'Pz']

    specified_features = [Feat_Method.PSDBIN]

    #groups = [HC_Sham_pts, HC_Stim_pts, CS_Sham_pts, CS_Stim_pts]
    all_hc_pts = ["HC", ["pro00087153_0020", "pro00087153_0023", "pro00087153_0027", "pro00087153_0028", "pro00087153_0036", "pro00087153_0022", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026", "pro00087153_0029", "pro00087153_0030"]]
    all_cs_pts = ["CS", ["pro00087153_0013", "pro00087153_0015", "pro00087153_0017", "pro00087153_0018", "pro00087153_0021", "pro00087153_0003", "pro00087153_0004", "pro00087153_0005", "pro00087153_0042", "pro00087153_0043"]]

    groups = [all_cs_pts]

    for group in groups:
        pts = group[1]
        all_x = []
        all_y = []
        for state in ["intra5", "intra15", "post"]:
            for patient_name in pts:
                '''
                if(patient_name in C4_contra_pts):
                    #patient is C4 contra, skip
                    continue
                '''
                #identify pts disease state
                '''
                if((patient_name in HC_Sham_pts[1]) | (patient_name in HC_Stim_pts[1])):
                    disease = Disease_State.HEALTHY
                else:
                    disease = Disease_State.STROKE
                '''

                #identify stim state
                if((patient_name in CS_Stim_pts[1]) | (patient_name in HC_Stim_pts[1])):
                    stim = Stim_State.STIM
                else:
                    stim = Stim_State.SHAM

                output_folder = pathlib.Path("output") / patient_name
                with (output_folder / 'file_metadata.json').open('r') as fp:
                    metadata = json.load(fp)

                sample_rate = metadata['eeg_sample_rate']
                spec_sample_rate = metadata['spec_sample_rate']
                num_seconds = metadata['seconds']

                #specified_features = [Feat_Method.PSD, Feat_Method.PSD, Feat_Method.COH]
                #specified_electrodes = ['C3C4']
                #specified_features = [Feat_Method.COHBIN] # WARNING, CODE SELECTED TO ONLY USE BETA (3, 4th)

                print(f"Using feature selection method: {specified_features} with {specified_electrodes}")
                #print(f"\ton patient {patient_name}")
                print("Using state: ", state)

                

                trials = sorted([x for x in metadata if x.startswith('TRIAL')]) # get in numeric order
                if state == 'pre':
                    trials = trials[0:1] # 0th
                elif state == 'intra5':
                    trials = trials[1:2] # 1st
                elif state == 'intra15':
                    trials = trials[2:3] # 2nd
                elif state == 'post':
                    trials = trials[3:4] # 3rd, only care about post5
                else:
                    print('Error with state:', state)
                    exit(-1)

                print('Trials used:', trials)

                for trial in trials:
                    #for action in [Y_Value.HOLD, Y_Value.REACH]:
                    for action in [Y_Value.HOLD, Y_Value.REACH]:
                        for reach in range(len(metadata[trial][action.name.lower()])):
                            x = []
                            for electrode, feature in zip(specified_electrodes, specified_features):
                                #x.append(np.loadtxt(output_folder / trial / action.name.lower() / f'{reach}_{electrode}_{feature.name.lower()}.csv', delimiter=','))
                                file_data = np.loadtxt(output_folder / trial / action.name.lower() / f'{reach}_{electrode}_{feature.name.lower()}.csv', delimiter=',')
                                if len(file_data.shape) == 1: # 3:4 specifies BETA
                                    x.append(file_data[0:5]) #all
                                elif len(file_data.shape) == 2:
                                    x.append(file_data[:, 0:5])
                            x = [row for row in np.hstack(x)]
                            if type(x[0]) == np.float64:
                                x = [np.array(x)]
                            
                            all_x.extend(x)
                            all_y.extend([stim.value] * len(x))

            print(len(all_x), len(all_y))
            assert(len(all_x) == len(all_y))
            scores = []
            kappa_scores = []


            results = "./" + group[0] + "_" + state + "/"
            os.system("mkdir %s" % results)
            results = pathlib.Path(results)
            fields = ["algorithm", "accuracy"]
            results_file_name = group[0] + "_" + state + ".csv"
            with (results / results_file_name).open('a') as fp:
                csvwriter = csv.writer(fp)
                csvwriter.writerow(fields)

            fields = ["algorithm", "kappa"]
            results_kappa_file_name = group[0] + "_" + state + "_kappa.csv"
            with (results / results_kappa_file_name).open('a') as fp:
                csvwriter = csv.writer(fp)
                csvwriter.writerow(fields)

            # For each abstractable training algorithm
            for i in Class_Method:
                current_model = ""
                if i == Class_Method.LR:
                    current_model = "LR"
                elif i == Class_Method.LDA:
                    current_model = "LDA"
                elif i == Class_Method.DT:
                    current_model = "DT"
                elif i == Class_Method.RF:
                    current_model = "RF"
                elif i == Class_Method.NB:
                    current_model = "NB"
                elif i == Class_Method.KNN:
                    current_model = "KNN"
                elif i == Class_Method.ADA:
                    current_model = "ADA"
                elif i == Class_Method.XG:
                    current_model = "XG"
                train_stop = datetime.datetime.now()

                models_dir = "./" + group[0] + "_models/"
                for model_n in range(0, 10):
                    #load model
                    filename = models_dir + current_model + str(model_n) + ".sav"
                    loaded_model = pickle.load(open(filename, 'rb'))
                    score = loaded_model.score(all_x, all_y)
                    scores.append(score)

                    #kappa
                    pred_y = loaded_model.predict(all_x)
                    kappa = metrics.cohen_kappa_score(pred_y, all_y)
                    kappa_scores.append(kappa)

                results = pathlib.Path(results)
                results_file_name = group[0] + "_" + state + ".csv"
                with (results / results_file_name).open('a') as fp:
                    csvwriter = csv.writer(fp)
                    for x in scores:
                        row = [current_model, str(x)]
                        csvwriter.writerow(row)
                
                results_kappa_file_name = group[0] + "_" + state + "_kappa.csv"
                with (results / results_kappa_file_name).open('a') as fp:
                    csvwriter = csv.writer(fp)
                    for x in kappa_scores:
                        row = [current_model, str(x)]
                        csvwriter.writerow(row)

               
            #score ensembles
            for ensemble in ["global", "hard", "me", "uni"]:
                scores = []
                kappa_scores = []
                current_model = ensemble
                models_dir = "./" + group[0] + "_models/"
                for model_n in range(0, 10):
                    #load model
                    filename = models_dir + current_model + str(model_n) + ".sav"
                    loaded_model = pickle.load(open(filename, 'rb'))
                    score = loaded_model.score(all_x, all_y)
                    scores.append(score)

                    #kappa
                    pred_y = loaded_model.predict(all_x)
                    kappa = metrics.cohen_kappa_score(pred_y, all_y)
                    kappa_scores.append(kappa)

                results = pathlib.Path(results)
                results_file_name = group[0] + "_" + state + ".csv"
                with (results / results_file_name).open('a') as fp:
                    csvwriter = csv.writer(fp)
                    for x in scores:
                        row = [current_model, str(x)]
                        csvwriter.writerow(row)
                
                results_kappa_file_name = group[0] + "_" + state + "_kappa.csv"
                with (results / results_kappa_file_name).open('a') as fp:
                    csvwriter = csv.writer(fp)
                    for x in kappa_scores:
                        row = [current_model, str(x)]
                        csvwriter.writerow(row)
                

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Failed!!')
        print(e.with_traceback())

    #beepy.beep(7)