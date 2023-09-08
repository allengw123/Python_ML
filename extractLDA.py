#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 21:30:26 2023

@author: changal
"""

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
import csv

from model_training_functions import *

from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.exceptions import FitFailedWarning

classification_dir = "/home/changal/Documents/bmi_allen_rishi/HVR_classification_"
groups = ["CS_Sham", "CS_Stim", "HC_Sham", "HC_Stim"]

output_file_dir = pathlib.Path(
    "/home/changal/Documents/bmi_allen_rishi/HVR_files/")


def write_to_output(group, band, weight):
    with open(os.path.join(output_file_dir, "LDAweights.csv"), 'a') as fp:
        writer = csv.writer(fp)
        row = [group, band, weight]
        writer.writerow(row)

# create csv file


with open(os.path.join(output_file_dir, "LDAweights.csv"), 'w') as fp:
    writer = csv.writer(fp)
    fields = ["group", "band", "weight"]
    writer.writerow(fields)

for group in groups:
    model_dir = classification_dir + group + "/disease_prediction/psdbin/"
    for LDA_n in range(0, 10):
        model_file = os.path.join(model_dir + "LDA" + str(LDA_n) + ".sav")
        loaded_model = pickle.load(open(model_file, 'rb'))
        weights = loaded_model.coef_[0]
        print(group + ":" + "LDA" + str(LDA_n) + "\n")
        for i in range(0, 5):
            if(i == 0):
                print("Delta: ", weights[i])
                write_to_output(group, "delta", weights[i])
                print("\n")
            elif(i == 1):
                print("Theta: ", weights[i])
                write_to_output(group, "theta", weights[i])
                print("\n")
            elif(i == 2):
                print("Alpha: ", weights[i])
                write_to_output(group, "alpha", weights[i])
                print("\n")
            elif(i == 3):
                print("Beta: ", weights[i])
                write_to_output(group, "beta", weights[i])
                print("\n")
            elif(i == 4):
                print("Gamma: ", weights[i])
                write_to_output(group, "gamma", weights[i])
                print("\n")
