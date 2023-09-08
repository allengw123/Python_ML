from eeg_utils import *

import multiprocessing
import os



patient_dir = "//home/changal/Documents/StrokeEEG_repo"  # Patient Dir
output_dir = "/home/changal/Documents/bmi_allen_rishi/patient_features_ipcon"  # Output Dir

sample_rate_scale = 1  # sample_rate / sample_rate_scale == new sample rate for eeg, spec
num_seconds = 8  # number of seconds of data to save in spec and eeg
epoch_seconds = 1  # size of each epoch

verbose = False

inputs = [patient_dir,
          output_dir,
          sample_rate_scale,
          num_seconds,
          epoch_seconds,
          verbose]

patients = stroke_patient_list()

createFeatures('pro00087153_0017',
                patient_dir,
                output_dir,
                sample_rate_scale,
                num_seconds,
                epoch_seconds,
                verbose)

# for subject_name in patients:
#     createFeatures(subject_name,
#                     patient_dir,
#                     output_dir,
#                     sample_rate_scale,
#                     num_seconds,
#                     epoch_seconds,
#                     verbose)

# 25
# 17
# 30
# with multiprocessing.Pool() as p:
#     results = p.starmap(createFeatures, [(subject_name,
#                                           patient_dir,
#                                           output_dir,
#                                           sample_rate_scale,
#                                           num_seconds,
#                                           epoch_seconds,
#                                           verbose)
#                                           for subject_name in patients])
