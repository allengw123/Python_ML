from cgi import FieldStorage
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
import os
import csv

model_dir = "./HC_Sham_models/"
models = ["LDA"]
features = ["delta", "theta", "alpha", "beta", "gamma"]

output_file = model_dir + "weights.csv"

fields = ["algorithm"]
fields.extend(features)

with (pathlib.Path(output_file)).open('a') as fp:
    csvwriter = csv.writer(fp)
    csvwriter.writerow(fields)

for model in models:
    print("Model: ", model)

    if(model not in ["LR", "LDA"]):
        continue
    else:
        for n in range(0, 10):
            filename = model_dir + model + str(n) + ".sav"
            print("Filename: ", filename)
            loaded_model = pickle.load(open(filename, 'rb'))
            weights = loaded_model.coef_[0]
            
            row = [model]
            with (pathlib.Path(output_file)).open('a') as fp:
                csvwriter = csv.writer(fp)
                for x in weights:
                    row.append(x)
                csvwriter.writerow(row)

    

    

    

