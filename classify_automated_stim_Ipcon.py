import datetime
import json
import math
import pathlib
import random
import sys
import pickle
import os
import numpy as np
import warnings
import re
import multiprocessing

from model_training_functions import *

from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.exceptions import FitFailedWarning


# Define input parameters
specified_electrodes = EEG_Channel_ipcon_list()
# electrodes = ['C3_Ip','C4_Ip','C3_Con','C4_Con']
verbose = False
training_trial_idx = 0
model_iterations = 10
val_size = 0.3
phases = ['hold', 'reach']
patient_dir = "/home/changal/Documents/bmi_allen_rishi/patient_features_ipcon"
model_output = "/home/changal/Documents/bmi_allen_rishi/Stim_classification_group_CS"
specified_features = 'psdbin'


# %% Perform model training 10 times and collect results
# Features generated by output_features.py

os.makedirs(model_output, exist_ok=True)

# Ignore Warnings
warnings.filterwarnings('ignore', category=FitFailedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Create log file
LOG_FILE_DIRECTORY = ""

# Obtain Patients
patient_dict = retreive_patient_info()
'''
all_patients = {"HEALTHY":
                patient_dict["HC_Sham"],
                "STROKE": patient_dict["CS_Sham"]}
'''
disease_type_patients = {"HEALTHY":
                         patient_dict["HC_Stim"] + patient_dict["HC_Sham"],
                         "STROKE": patient_dict["CS_Sham"] + patient_dict["CS_Stim"]}

stim_type_patients = {"STIM":
                      patient_dict["HC_Stim"] + patient_dict["CS_Stim"],
                      "SHAM": patient_dict["HC_Sham"] + patient_dict["CS_Sham"]}

all_patients = []
subject_list = disease_type_patients["STROKE"]


# Perform model training 10 times and collect results
# Features generated by output_features.py

for train_trial in range(model_iterations):

    print(f"Iteration {train_trial + 1}")

    print('Loading subject features')

    # for subject in subject_list:
    #     process_subject(subject,
    #                     patient_dir,
    #                     training_trial_idx,
    #                     phases,
    #                     specified_features,
    #                     specified_electrodes,
    #                     all_patients,
    #                     val_size
    #                     )
    # Create a pool of worker processes
    with multiprocessing.Pool() as p:
        results = p.starmap(process_subject, [(subject,
                                              patient_dir,
                                              training_trial_idx,
                                              phases,
                                              specified_features,
                                              specified_electrodes,
                                              all_patients,
                                              val_size
                                               )
                                              for subject in subject_list])

    print('Features loaded')

    train_x, train_y, val_x, val_y = zip(*results)

    # Combine the results
    train_x = [item for sublist in train_x for item in sublist]
    train_y = [item for sublist in train_y for item in sublist]

    for i in range(0, len(train_y)):
        if train_y[i] == 1:
            train_y[i] = 0
        elif train_y[i] == 2:
            train_y[i] = 1

    val_x = [item for sublist in val_x for item in sublist]
    val_y = [item for sublist in val_y for item in sublist]

    for i in range(0, len(val_y)):
        if val_y[i] == 1:
            val_y[i] = 0
        elif val_y[i] == 2:
            val_y[i] = 1

    print("VAL Y \n", val_y, "\n")

    all_models = []
    scores = []
    train_scores = []
    kappa_scores = []

    # For each abstractable training algorithm
    for i in Class_Method:
        current_model = ""
        train_start = datetime.datetime.now()
        # if i == Class_Method.SVM:
        #     current_model = 'SVM'
        #     model, params = train_svm(train_x, train_y)
        if i == Class_Method.LR:
            current_model = "LR"
            print('Training ' + current_model)
            model, params = train_lr(train_x, train_y)
        elif i == Class_Method.LDA:
            current_model = "LDA"
            print('Training ' + current_model)
            model, params = train_lda(train_x, train_y)
        elif i == Class_Method.DT:
            current_model = "DT"
            print('Training ' + current_model)
            model, params = train_dt(train_x, train_y)
        elif i == Class_Method.RF:
            current_model = "RF"
            print('Training ' + current_model)
            model, params = train_rf(train_x, train_y)
        elif i == Class_Method.NB:
            current_model = "NB"
            print('Training ' + current_model)
            model, params = train_nb(train_x, train_y)
        elif i == Class_Method.KNN:
            current_model = "KNN"
            print('Training ' + current_model)
            model, params = train_knn(train_x, train_y)
        elif i == Class_Method.ADA:
            current_model = "ADA"
            print('Training ' + current_model)
            model, params = train_ada(train_x, train_y)
        elif i == Class_Method.XG:
            current_model = "XG"
            print('Training ' + current_model)
            model, params = train_xg(train_x, train_y)

        train_stop = datetime.datetime.now()

        score = model.score(train_x, train_y)
        train_scores.append(score)
        score = model.score(val_x, val_y)

        # Model directory
        models_dir = os.path.join(
            model_output, "disease_prediction", specified_features)
        os.makedirs(models_dir, exist_ok=True)

        # save model
        filename = os.path.join(
            models_dir, current_model + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        # Output hyperparameters
        with open(os.path.join(models_dir, f'hyperparameters_{train_trial}.txt'), 'a') as hp:
            model_string = "Model " + current_model + \
                " trial-" + str(train_trial + 1) + ": \n"
            hp.write(model_string)
            for p in params:
                hp.write("\t")
                param_string = p + ": " + str(params[p]) + "\n"
                hp.write(param_string)
            hp.write("\n")

        scores.append(score)
        pred_y = model.predict(val_x)
        kappa = metrics.cohen_kappa_score(pred_y, val_y)
        kappa_scores.append(kappa)

        # if i not in [Class_Method.ADA, Class_Method.XG, Class_Method.SVM]:
        if i not in [Class_Method.ADA, Class_Method.XG]:
            all_models.append(
                {'type': i, 'model': model, 'score': model.score(train_x, train_y)})

        val_start = datetime.datetime.now()
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        val_stop = datetime.datetime.now()

        # Output confusion matrix
        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write(f"{i.name}\n")
            for row in conf_matx:
                fp.write(f"\t{row}\n")
            fp.write("\n")

        print(
            f"Model {i} scored \t{score*100:.2f}% in {train_stop - train_start} training time")
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
        print(
            f"Voting Ensemble Order: {[x['type'].name for x in all_models]}")

        # Ensemble weighted by training accuracies
        train_model = EnsembleVoteClassifier([x['model'] for x in all_models], voting='soft', weights=[
            x['score'] for x in all_models], fit_base_estimators=False)
        train_model.fit(train_x, train_y)
        score = train_model.score(train_x, train_y)
        train_scores.append(score)

        val_start = datetime.datetime.now()
        score = train_model.score(val_x, val_y)
        scores.append(score)
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        kappa = metrics.cohen_kappa_score(val_y, val_res_y)
        kappa_scores.append(kappa)
        val_stop = datetime.datetime.now()

        filename = os.path.join(
            models_dir, 'train' + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write("train\n")
            for row in conf_matx:
                fp.write("\t".join(map(str, row)) + "\n")
            fp.write("\n")

        print(f"Model TRAIN scored \t\t{score * 100:.2f}%")
        if verbose:
            print(
                f"\t\tWeights used:\t{[x['score'] for x in all_models]}.")
            print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        # Ensemble weighted by personal opinion
        # weights = [3, 1, 2, 1, 3, 2, 2]
        weights = [1, 2, 1, 3, 2, 2]
        me_model = EnsembleVoteClassifier(
            [x['model'] for x in all_models], voting='soft', weights=weights, fit_base_estimators=False)
        me_model.fit(train_x, train_y)
        score = me_model.score(train_x, train_y)
        train_scores.append(score)

        val_start = datetime.datetime.now()
        score = me_model.score(val_x, val_y)
        scores.append(score)
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        kappa = metrics.cohen_kappa_score(val_y, val_res_y)
        kappa_scores.append(kappa)
        val_stop = datetime.datetime.now()

        filename = os.path.join(
            models_dir, 'me' + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write("me\n")
            for row in conf_matx:
                fp.write("\t".join(map(str, row)) + "\n")
            fp.write("\n")

        print(f"Model ME scored \t\t{score * 100:.2f}%")
        if verbose:
            print(f"\t\tWeights used:\t{weights}.")
            print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        # Ensemble weighted by empirical global accuracies
        # weights = [.72662, .59793, .71133, .72925, .79051, .67535, .72198]
        weights = [.59793, .71133, .72925, .79051, .67535, .72198]
        global_model = EnsembleVoteClassifier(
            [x['model'] for x in all_models], voting='soft', weights=weights, fit_base_estimators=False)
        global_model.fit(train_x, train_y)
        score = global_model.score(train_x, train_y)
        train_scores.append(score)

        val_start = datetime.datetime.now()
        score = global_model.score(val_x, val_y)
        scores.append(score)
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        kappa = metrics.cohen_kappa_score(val_y, val_res_y)
        kappa_scores.append(kappa)
        val_stop = datetime.datetime.now()

        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write(f"global\n")
            for row in conf_matx:
                fp.write(f"\t{row}\n")
                fp.write("\n")

        filename = os.path.join(
            models_dir, 'global' + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        print(f"Model GLOBAL scored \t\t{score * 100:.2f}%")
        if verbose:
            print(f"\t\tWeights used:\t{weights}.")
            print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        # Ensemble with uniform weights
        uniform_model = EnsembleVoteClassifier(
            [x['model'] for x in all_models], voting='soft', fit_base_estimators=False)
        uniform_model.fit(train_x, train_y)
        score = uniform_model.score(train_x, train_y)
        train_scores.append(score)

        val_start = datetime.datetime.now()
        score = uniform_model.score(val_x, val_y)
        scores.append(score)
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        kappa = metrics.cohen_kappa_score(val_y, val_res_y)
        kappa_scores.append(kappa)
        val_stop = datetime.datetime.now()

        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write("uniform\n")
            for row in conf_matx:
                fp.write("\t".join(map(str, row)) + "\n")
            fp.write("\n")

        filename = os.path.join(
            models_dir, 'uni' + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        print(f"Model UNI scored \t\t{score * 100:.2f}%")
        if verbose:
            print(f"\t\tValidation time: {val_stop - val_start}")
        print()

        # Ensemble with hard voting
        hard_model = EnsembleVoteClassifier(
            [x['model'] for x in all_models], voting='hard', fit_base_estimators=False)
        hard_model.fit(train_x, train_y)
        score = hard_model.score(train_x, train_y)
        train_scores.append(score)

        val_start = datetime.datetime.now()
        score = hard_model.score(val_x, val_y)
        scores.append(score)
        val_res_y = model.predict(val_x)
        conf_matx = metrics.confusion_matrix(val_y, val_res_y)
        kappa = metrics.cohen_kappa_score(val_y, val_res_y)
        kappa_scores.append(kappa)
        val_stop = datetime.datetime.now()

        filename = os.path.join(
            models_dir, 'hard' + str(train_trial) + ".sav")
        pickle.dump(model, open(filename, 'wb'))

        with open(os.path.join(models_dir, f'model_conf_matrix_{train_trial}.txt'), 'a') as fp:
            fp.write(f"hard\n")
            for row in conf_matx:
                fp.write("\t".join(map(str, row)) + "\n")
            fp.write("\n")

        print(f"Model HARD scored \t\t{score * 100: .2f}%")
        if verbose:
            print(f"\t\tValidation time: {val_stop - val_start}")
            print()

    models_dir = pathlib.Path(models_dir)
    results_csv = "pre_results.csv"
    with open(models_dir / results_csv, 'a') as fp:
        writer = csv.writer(fp)
        fields = ["LR", "LDA", "DT", "RF", "NB", "KNN",
                  "ADA", "XG", "train", "global", "hard", "uni"]
        writer.writerow(fields)
    with (models_dir / results_csv).open('a') as fp:
        fp.write(','.join(str(x) for x in scores) + '\n')
    with (models_dir / f'model_accuracies_train.csv').open('a') as fp:
        fp.write(','.join(str(x) for x in train_scores) + '\n')
    with (models_dir / f'kappa.csv').open('a') as fp:
        fp.write(','.join(str(x) for x in kappa_scores) + '\n')