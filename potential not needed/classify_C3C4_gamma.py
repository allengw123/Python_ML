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
    if len(sys.argv) < 2:
        # change 7-1-2022 by Rishi
        #patient_name = "pro00087153_0015"
        patient_name = "pro00087153_0036"
    else:
        patient_name = sys.argv[1]
        state = sys.argv[2]

    #state = "intra5" # pre | intra5 | intra15 | post
    #new_output_folder_path = "output/contralateral_C3/post"
    output_folder = pathlib.Path("output") / patient_name
    with (output_folder / 'file_metadata.json').open('r') as fp:
        metadata = json.load(fp)

    sample_rate = metadata['eeg_sample_rate']
    spec_sample_rate = metadata['spec_sample_rate']
    num_seconds = metadata['seconds']

    #specified_electrodes = ['C3', 'C4', 'C3C4']
    specified_electrodes = ['C3C4']
    #specified_electrodes = ['T3', 'T4', 'T3T4']
    #specified_features = [Feat_Method.PSDBIN, Feat_Method.PSDBIN, Feat_Method.COHBIN]
    specified_features = [Feat_Method.COHBIN]

    #specified_features = [Feat_Method.PSD, Feat_Method.PSD, Feat_Method.COH]
    #specified_electrodes = ['C3C4']
    #specified_features = [Feat_Method.COHBIN] # WARNING, CODE SELECTED TO ONLY USE BETA (3, 4th)

    print(f"Using feature selection method: {specified_features} with {specified_electrodes}")
    print(f"\ton patient {patient_name}")
    print("Using state: ", state)

    all_x = []
    all_y = []

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
        #for action in [Y_Value.HOLD, Y_Value.REACH, Y_Value.PREP]:
        for action in [Y_Value.HOLD, Y_Value.PREP, Y_Value.REACH]:
            for reach in range(len(metadata[trial][action.name.lower()])):
                x = []
                for electrode, feature in zip(specified_electrodes, specified_features):
                    #x.append(np.loadtxt(output_folder / trial / action.name.lower() / f'{reach}_{electrode}_{feature.name.lower()}.csv', delimiter=','))
                    file_data = np.loadtxt(output_folder / trial / action.name.lower() / f'{reach}_{electrode}_{feature.name.lower()}.csv', delimiter=',')
                    if len(file_data.shape) == 1: # 3:4 specifies BETA
                        x.append(file_data[4:5])
                    elif len(file_data.shape) == 2:
                        x.append(file_data[:, 4:5])
                x = [row for row in np.hstack(x)]
                if type(x[0]) == np.float64:
                    x = [np.array(x)]
                all_x.extend(x)
                all_y.extend([action.value] * len(x))

    print(len(all_x), len(all_y))
    assert(len(all_x) == len(all_y))

    num_items = len(all_x)
    percent_train = 0.7
    num_train = math.floor(num_items * percent_train) # 70/30
    num_val = num_items - num_train
    print(f"Number of training data: {num_train}; Number of validation data: {num_val}")

    # Create/overwrite file for saving accuracies
    model_names = [x.name for x in Class_Method] + ["train", "me", "global", "uniform", "hard"]
    with (output_folder / 'model_accuracies.csv').open('w') as fp:
        fp.write(','.join(model_names) + '\n')
    with (output_folder / 'model_accuracies_train.csv').open('w') as fp:
        fp.write(','.join(model_names) + '\n')
    #overwrite hyperparameter file
    with (output_folder / "hyperparameters.txt").open('w') as hp:
        hp.write(patient_name + "\n")
        hp.write("State: " + state + "\n")
        hp.write("Sample rate: " + str(sample_rate) + "\n")
        hp.write("Spec sample rate: " + str(spec_sample_rate) + "\n")
        hp.write("Number of seconds: " + str(num_seconds) + "\n")
        hp.write("Specified electrodes: ")
        for electrode in specified_electrodes:
            hp.write(electrode)
        hp.write("\n" + "Specified features: ")
        for feature in specified_features:
            if feature == Feat_Method.PSD:
                hp.write("PSD ")
            elif feature == Feat_Method.PSDBIN:
                hp.write("PSDBIN ")
            elif feature == Feat_Method.DE:
                hp.write("DE ")
            elif feature == Feat_Method.COH:
                hp.write("COH ")
            elif feature == Feat_Method.COHBIN:
                hp.write("COHBIN ")
        hp.write("\n")
    # Perform model training 10 times and collect results
    # Features generated by output_features.py
    for train_trial in range(10):
        print(f"Iteration {train_trial + 1}")
        # Separate into training and validation
        val_indices = random.sample(list(range(num_items)), num_val)
        train_x, train_y, val_x, val_y = [], [], [], []
        for i in range(num_items):
            if i in val_indices:
                val_x.append(all_x[i])
                val_y.append(all_y[i])
            else:
                train_x.append(all_x[i])
                train_y.append(all_y[i])

        all_models = []
        scores = []
        train_scores = []

        # For each abstractable training algorithm
        for i in Class_Method:
            current_model = ""
            train_start = datetime.datetime.now()
            #if i == Class_Method.SVM:
            #    model, params = train_svm(train_x, train_y)
            #elif i == Class_Method.LR:
            if i == Class_Method.LR:
                current_model = "LR"
                model, params = train_lr(train_x, train_y)
            elif i == Class_Method.LDA:
                current_model = "LDA"
                model, params = train_lda(train_x, train_y)
            elif i == Class_Method.DT:
                current_model = "DT"
                model, params = train_dt(train_x, train_y)
            elif i == Class_Method.RF:
                current_model = "RF"
                model, params = train_rf(train_x, train_y)
            elif i == Class_Method.NB:
                current_model = "NB"
                model, params = train_nb(train_x, train_y)
            elif i == Class_Method.KNN:
                current_model = "KNN"
                model, params = train_knn(train_x, train_y)
            elif i == Class_Method.ADA:
                current_model = "ADA"
                model, params = train_ada(train_x, train_y)
            elif i == Class_Method.XG:
                current_model = "XG"
                model, params = train_xg(train_x, train_y)
            train_stop = datetime.datetime.now()

            score = model.score(train_x, train_y)
            train_scores.append(score)
            score = model.score(val_x, val_y)

            #output hyperparameters
            with (output_folder / "hyperparameters.txt").open('a') as hp:
                model_string = "Model " + current_model + " trial-" + str(train_trial+1) + ": \n"
                hp.write(model_string)
                for p in params:
                    hp.write("\t")
                    param_string = p + ": " + str(params[p]) + "\n"
                    hp.write(param_string)
                hp.write("\n")

            scores.append(score)
            #if i not in [Class_Method.ADA, Class_Method.XG, Class_Method.SVM]:
            if i not in [Class_Method.ADA, Class_Method.XG]:
                all_models.append({'type': i, 'model': model, 'score': model.score(train_x, train_y)})

            val_start = datetime.datetime.now()
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"{i.name}\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model {i} scored \t{score*100:.2f}% in {train_stop - train_start} training time")
            if verbose:
                print("Confusion Matrix:")
                for row in conf_matx:
                    print(f"\t{row}")
                print(f"Parameters: {json.dumps(params, indent=2)}")
                print(f"Training time:\t\t{train_stop - train_start}")
                print(f"Validation time:\t{val_stop - val_start}")
                print()
            print()

        if len(all_models) == 6:
            print(f"Voting Ensemble Order: {[x['type'].name for x in all_models]}")

            # Ensemble weighted by training accuracies
            train_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=[x['score'] for x in all_models], fit_base_estimators=False)
            train_model.fit(train_x, train_y)
            score = train_model.score(train_x, train_y)
            train_scores.append(score)

            val_start = datetime.datetime.now()
            score = train_model.score(val_x, val_y)
            scores.append(score)
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"train\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model TRAIN scored \t\t{score * 100:.2f}%")
            if verbose:
                print(f"\t\tWeights used:\t{[x['score'] for x in all_models]}.")
                print(f"\t\tValidation time: {val_stop - val_start}")
            print()

            # Ensemble weighted by personal opinion
            #weights = [3, 1, 2, 1, 3, 2, 2]
            weights = [1, 2, 1, 3, 2, 2]
            me_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=weights, fit_base_estimators=False)
            me_model.fit(train_x, train_y)
            score = me_model.score(train_x, train_y)
            train_scores.append(score)

            val_start = datetime.datetime.now()
            score = me_model.score(val_x, val_y)
            scores.append(score)
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"me\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model ME scored \t\t{score * 100:.2f}%")
            if verbose:
                print(f"\t\tWeights used:\t{weights}.")
                print(f"\t\tValidation time: {val_stop - val_start}")
            print()

            # Ensemble weighted by empirical global accuracies
            #weights = [.72662, .59793, .71133, .72925, .79051, .67535, .72198]
            weights = [.59793, .71133, .72925, .79051, .67535, .72198]
            global_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=weights, fit_base_estimators=False)
            global_model.fit(train_x, train_y)
            score = global_model.score(train_x, train_y)
            train_scores.append(score)

            val_start = datetime.datetime.now()
            score = global_model.score(val_x, val_y)
            scores.append(score)
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"global\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model GLOBAL scored \t\t{score * 100:.2f}%")
            if verbose:
                print(f"\t\tWeights used:\t{weights}.")
                print(f"\t\tValidation time: {val_stop - val_start}")
            print()

            # Ensemble with uniform weights
            uniform_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', fit_base_estimators=False)
            uniform_model.fit(train_x, train_y)
            score = uniform_model.score(train_x, train_y)
            train_scores.append(score)

            val_start = datetime.datetime.now()
            score = uniform_model.score(val_x, val_y)
            scores.append(score)
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"uniform\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model UNI scored \t\t{score * 100:.2f}%")
            if verbose:
                print(f"\t\tValidation time: {val_stop - val_start}")
            print()

            # Ensemble with hard voting
            hard_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='hard', fit_base_estimators=False)
            hard_model.fit(train_x, train_y)
            score = hard_model.score(train_x, train_y)
            train_scores.append(score)

            val_start = datetime.datetime.now()
            score = hard_model.score(val_x, val_y)
            scores.append(score)
            val_res_y = model.predict(val_x)
            conf_matx = metrics.confusion_matrix(val_y, val_res_y)
            val_stop = datetime.datetime.now()

            with (output_folder / f'model_conf_matrix_{train_trial + 1}.txt').open('a') as fp:
                fp.write(f"hard\n")
                for row in conf_matx:
                    fp.write(f"\t{row}\n")
                fp.write("\n")

            print(f"Model HARD scored \t\t{score * 100: .2f}%")
            if verbose:
                print(f"\t\tValidation time: {val_stop - val_start}")
            print()

        with (output_folder / f'model_accuracies.csv').open('a') as fp:
            fp.write(','.join(str(x) for x in scores) + '\n')
        with (output_folder / f'model_accuracies_train.csv').open('a') as fp:
            fp.write(','.join(str(x) for x in train_scores) + '\n')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Failed!!')
        print(e.with_traceback())

    #beepy.beep(7)