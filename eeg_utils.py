import csv
import glob
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import pathlib
import pyfftw
import time
import xml.etree.ElementTree as ET
import os
import itertools
import json
import math
import mne
import sys


from scipy import signal
from enum import Enum, auto
from collections import defaultdict

from enum import Enum
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats, signal


def createFeatures(subject_name, patient_dir, output_dir, sample_rate_scale,
                   num_seconds, epoch_seconds, verbose):

    status = []

    # Add Paths
    subject_folder = os.path.join(patient_dir, subject_name)
    edf_file_path = os.path.join(patient_dir, subject_name, 'edf',
                                 f'{subject_name}.edf')

    # Make paths
    os.makedirs(output_dir, exist_ok=True)
    output_folder = Path(output_dir) / subject_name

    # Check if subject completed already
    if os.path.exists(os.path.join(output_dir, subject_name, 'file_metadata.json')):
        print('Subject ' + subject_name
              + ' processed... skipping')
        return

    # Find handedness
    move_hand = [key for key, values in retreive_patient_info().items(
    ) if subject_name in values and key in ['Left', 'Right']][0]

    # Load EEG and header info
    edf = mne.io.read_raw_edf(edf_file_path)
    print('reading ' + edf_file_path)
    num_ch = edf.info['nchan']
    sample_rate = int(edf.info['sfreq'])
    spec_sample_rate = sample_rate // sample_rate_scale

    # Find VR files
    num_vr_trials = len([pathlib.Path(x)
                         for x in glob.glob(str(subject_folder) + '/vr/*')])

    # Find VR signals associated with each trial
    if subject_name[len(subject_name)-2:] in ["03", "04", "05"]:
        vr_trial_samples = get_trial_samples(
            edf, num_vr_trials, True)  # 3/4/5 uses diff params
    else:
        vr_trial_samples = get_trial_samples(edf, num_vr_trials, False)
    if len(vr_trial_samples) == 0:
        status = 'VR signal not detected '+subject_name
    elif len(vr_trial_samples) != num_vr_trials:
        status = 'VR signal doesnt match VR trials'+subject_name
        return status

    # Find VR events
    all_hold_events, all_prep_events, all_reach_events, saved_trial_numbers, all_events, trial_names = parse_trial_timestamps(
        subject_folder, vr_trial_samples, sample_rate, verbose)
    index_events = [all_hold_events, all_prep_events, all_reach_events]

    # Normalize EEG
    normalize = NormOptions.whole
    eeg_data = {}
    eeg_channels = EEG_Channel_list()
    for ch in eeg_channels:
        eeg_data[ch] = edf.get_data([ch])[0]
        if normalize == NormOptions.whole:
            eeg_data[ch] = normalize_signal(eeg_data[ch])

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_metadata = {
        'spec_sample_rate': spec_sample_rate,
        'eeg_sample_rate': sample_rate,
        'channels': eeg_channels,
        ' seconds': num_seconds,
        'normalization': normalize.name
    }

    # Extract features for each epoch [/ name / hold / id_{ch}_{feature}]
    num_eeg_samples = sample_rate * num_seconds
    num_spec_samples = spec_sample_rate * num_seconds
    for i, name in enumerate(trial_names):
        output_metadata[name] = {}

        # for each position (hold / prep / reach)
        for pos_index, pos_name in enumerate(['hold', 'prep', 'reach']):
            output_metadata[name][pos_name] = {}
            (output_folder / name / pos_name).mkdir(parents=True, exist_ok=True)

            # for each individual reach
            for id, (start, stop) in enumerate(index_events[pos_index][i]):
                base_sample = math.floor(
                    start - ((num_eeg_samples - (stop - start)) / 2))
                # Save metadata
                output_metadata[name][pos_name][id] = {
                    'base_sample': base_sample,
                    'end_sample': base_sample + num_eeg_samples,
                    'start_event': start,
                    'stop_event': stop,
                    'start_event_spec': math.floor((start - base_sample) / sample_rate_scale),
                    'stop_event_spec': math.ceil((stop - base_sample) / sample_rate_scale)
                }

                # output trimmed eeg and spec
                for ch in eeg_channels:

                    # generate trimmed eeg and normalize
                    trim_eeg = eeg_data[ch][base_sample: base_sample +
                                            num_eeg_samples: sample_rate_scale]
                    if normalize == NormOptions.epoch:
                        trim_eeg = normalize_signal(trim_eeg)
                    with (output_folder / name / pos_name / f'{id}_{ch}_eeg.csv').open('w') as fp:
                        if verbose:
                            print(fp.name)
                        fp.write('\n'.join(str(x) for x in trim_eeg))

                    # generate spectrogram
                    # spec = generate_spectrogram(eeg_data[ch], start, sample_rate, num_seconds, sample_rate_scale)
                    # with (output_folder / name / pos_name / f'{id}_{ch}_spec.csv').open('w') as fp:
                    #    print(fp.name)
                    #    fp.write('\n'.join(','.join(str(col) for col in row) for row in spec))

                    # generate differential entropy over entire epoch
                    de_array = generate_de(
                        trim_eeg, sample_rate // sample_rate_scale)
                    with (output_folder / name / pos_name / f'{id}_{ch}_de.csv').open('w') as fp:
                        if verbose:
                            print(fp.name)
                        fp.write('\n'.join(str(x) for x in de_array))

                # Feature Extraction for each epoch
                feat_output = defaultdict(list)
                for epoch_num, epoch in enumerate(range(start, stop, sample_rate)):
                    # Calculate PSD
                    for ch in eeg_channels:
                        # non-binned psd
                        _, psd = signal.welch(
                            eeg_data[ch][epoch: epoch + epoch_seconds *
                                         sample_rate], fs=sample_rate, nfft=math.floor(sample_rate*1.1))

                        ip_con_cn_name = ip_con_switch(ch, move_hand)
                        feat_output[f'{ip_con_cn_name}_psd'].append(psd)

                        # binned psd
                        psd_delta = np.sum(
                            psd[eeg_bands.DELTA[0]: eeg_bands.DELTA[1] + 1])
                        psd_theta = np.sum(
                            psd[eeg_bands.THETA[0]: eeg_bands.THETA[1] + 1])
                        psd_alpha = np.sum(
                            psd[eeg_bands.ALPHA[0]: eeg_bands.ALPHA[1] + 1])
                        psd_beta = np.sum(
                            psd[eeg_bands.BETA[0]: eeg_bands.BETA[1] + 1])
                        psd_gamma = np.sum(
                            psd[eeg_bands.GAMMA[0]: eeg_bands.GAMMA[1] + 1])
                        feat_output[f'{ip_con_cn_name}_psdbin'].append(
                            [psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])

                    # Calculate Coherence
                    for ch1, ch2 in itertools.combinations(eeg_channels, 2):
                        if ch1 == ch2:
                            continue

                        ch1_ipcon = ip_con_switch(ch1, move_hand)
                        ch2_ipcon = ip_con_switch(ch2, move_hand)

                        # non-binned coherence
                        _, coh = signal.coherence(eeg_data[ch1][epoch: epoch + epoch_seconds*sample_rate], eeg_data[ch2]
                                                  [epoch: epoch + 2*sample_rate], fs=sample_rate, nfft=math.floor(sample_rate*1.1))

                        # if period of blank or steady state signal, possible that coherence will return nan due to autospectral density = 0
                        for bad_index in np.where(np.isnan(coh))[0]:
                            fix = []
                            if bad_index - 1 > 0:
                                fix.append(coh[bad_index - 1])
                            if bad_index + 1 < coh.shape[0]:
                                fix.append(coh[bad_index + 1])
                            coh[bad_index] = np.average(fix)

                        feat_output[f'{ch1_ipcon}{ch2_ipcon}_coh'].append(coh)

                        # binned coherence
                        coh_delta = np.sum(
                            coh[eeg_bands.DELTA[0]: eeg_bands.DELTA[1] + 1])
                        coh_theta = np.sum(
                            coh[eeg_bands.THETA[0]: eeg_bands.THETA[1] + 1])
                        coh_alpha = np.sum(
                            coh[eeg_bands.ALPHA[0]: eeg_bands.ALPHA[1] + 1])
                        coh_beta = np.sum(
                            coh[eeg_bands.BETA[0]: eeg_bands.BETA[1] + 1])
                        coh_gamma = np.sum(
                            coh[eeg_bands.GAMMA[0]: eeg_bands.GAMMA[1] + 1])

                        feat_output[f'{ch1_ipcon}{ch2_ipcon}_cohbin'].append(
                            [coh_delta, coh_theta, coh_alpha, coh_beta, coh_gamma])

                # Create CSV file output for each feature
                for feat_key in feat_output:
                    tmp_arr = np.vstack(feat_output[feat_key])
                    with (output_folder / name / pos_name / f'{id}_{feat_key}.csv').open('w') as fp:
                        if verbose:
                            print(fp.name)
                        fp.write('\n'.join(','.join(str(col)
                                 for col in row) for row in tmp_arr))

    # output the json file
    # Make sure all int64 values are int
    with (output_folder / 'file_metadata.json').open('w') as fp:
        output_metadata = convert_int64_to_int(output_metadata)
        json.dump(output_metadata, fp, indent=2)

    print(f"Done! Look into {output_folder} for all outputted data!")
    status = subject_name + ' Completed!'


def get_trial_samples(edf: mne.io.Raw, num_vr_trials, alt):

    if alt:
        vr_sig = edf.copy().pick(['DC2'])
        vr_sig = (np.abs(vr_sig.load_data().get_data(['DC2'])) > 0.05)[
            0]  # needed for 0003/0004/0005
    else:
        vr_sig = edf.copy().pick(['DC1'])
        vr_sig = (np.abs(vr_sig.load_data().get_data(['DC1'])) > 2.5)[0]

    # Find indices where there's a switch from True to False or from False to True
    switch_to_false_indices = np.where(
        np.logical_and(vr_sig[:-1], ~vr_sig[1:]))[0] + 1
    switch_to_true_indices = np.where(
        np.logical_and(~vr_sig[:-1], vr_sig[1:]))[0] + 1

    # Create a list of tuples containing the indices
    switch_indices_list = list(
        zip(switch_to_true_indices, switch_to_false_indices))

    # Calculate the differences between tuple elements and keep track of original indices
    differences_with_indices = [(calculate_difference(item), i)
                                for i, item in enumerate(switch_indices_list)]

    # Sort the list of tuples based on differences in descending order
    sorted_indices_with_indices = sorted(
        differences_with_indices, reverse=True)

    # Get the indices of the tuples to keep
    indices_to_keep = [item[1]
                       for item in sorted_indices_with_indices[:num_vr_trials]]
    sorted_indices_to_keep = sorted(indices_to_keep)

    # Keep only the tuples with the largest differences while maintaining original order
    result_list = [switch_indices_list[i] for i in sorted_indices_to_keep]

    return result_list


def parse_trial_timestamps(subject_folder, vr_trial_samples, sample_rate, verbose):
    all_hold_events = []
    all_prep_events = []
    all_reach_events = []
    saved_trial_numbers = []
    all_events = []
    trial_names = []

    vr_paths = [pathlib.Path(x)
                for x in glob.glob(str(subject_folder) + '/vr/*')]
    if len(vr_paths) == 0:
        status = 'ERROR: VR FOLDERS NOT FOUND', subject_folder
        return status

    for vr_idx, path in enumerate(vr_paths):

        vr_trial_num = int(path.name[-3:])

        trial_xml = ET.parse(path/'trial information.xml')
        if trial_xml.find('quality').text != 'accepted' or trial_xml.find('studyCondition').text != 'Single reach per cue, random time, no visual cue':
            if verbose:
                print("Invalid trial, continuing.")
            continue
        else:
            if verbose:
                print("Working on Trial", vr_trial_num)

        base = vr_trial_samples[vr_idx][0]

        # Parse through Events.csv to get each trial
        with open(path/'Events.csv', 'r') as fp:
            csv_reader = csv.reader(fp)
            raw_events = [i for i in csv_reader][1:]

        # atStartPosition -> cueEvent (HOLD)
        # cueEvent -> targetUp (PREP)
        # targetUp -> targetHit (REACH)
        hold_events, prep_events, reach_events, events = [], [], [], []
        hold_start, prep_start, reach_start = -1, -1, -1
        for i in raw_events:
            if i[event_index.EVENT] == "atStartPosition":
                hold_start = base + \
                    math.ceil(float(i[event_index.TIME]) * sample_rate)
                events.append((hold_start - base, event_ids.HOLD_START))
            elif i[event_index.EVENT] == "cueEvent":
                prep_start = base + \
                    math.ceil(float(i[event_index.TIME]) * sample_rate)
                events.append((prep_start - base, event_ids.PREP_START))
                hold_events.append((hold_start, prep_start))
                hold_start = -1
            elif i[event_index.EVENT] == "targetUp":
                reach_start = base + \
                    math.ceil(float(i[event_index.TIME]) * sample_rate)
                events.append((reach_start - base, event_ids.REACH_START))
                prep_events.append((prep_start, reach_start))
                prep_start = -1
            elif i[event_index.EVENT] == "targetHit":
                reach_stop = base + \
                    math.ceil(float(i[event_index.TIME]) * sample_rate)
                events.append((reach_stop - base, event_ids.REACH_STOP))
                reach_events.append((reach_start, reach_stop))
                reach_start = -1
            elif i[event_index.EVENT] == "outsideStartPosition":
                if hold_start != -1:
                    # Remove the hold_start event
                    events.pop()
                    hold_start = -1
                if prep_start != -1:
                    # Remove the prep_start event
                    events.pop()
                    events.pop()
                    prep_start = -1
                    hold_events.pop()
                else:
                    continue
            else:
                continue

        # Skip flexion trials
        if not len(reach_events) > 3:
            continue

        assert len(
            reach_events) == 12, 'LESS THAN 12 REACH EVENTS DETECTED: ' + str(path)

        trial_names.append(os.path.basename(path))
        all_hold_events.append(hold_events)
        all_prep_events.append(prep_events)
        all_reach_events.append(reach_events)
        saved_trial_numbers.append(vr_trial_num)
        all_events.append(events)

    return all_hold_events, all_prep_events, all_reach_events, saved_trial_numbers, all_events, trial_names


class event_ids:
    HOLD_START = 0
    PREP_START = 1
    REACH_START = 2
    REACH_STOP = 3
    INVALID_STATE = 4


class event_index:
    TIME = 0
    EVENT = 2
    INFO = 3


class eeg_bands:
    DELTA = [1, 3]
    THETA = [4, 7]
    ALPHA = [8, 12]
    BETA = [13, 30]
    GAMMA = [31, 50]


def output_trim_eeg(edf_file: Path, drop_channels: list[str], notch_freqs: list[float], band_pass: list[float]):
    raw_edf = mne.io.read_raw_edf(edf_file)
    raw_edf.drop_channels(drop_channels)
    raw_edf.drop_channels(
        [ch for ch in raw_edf.ch_names if ch.startswith('X')])
    raw_edf.load_data()
    if notch_freqs:
        raw_edf.notch_filter(notch_freqs)
    if band_pass and len(band_pass) == 2:
        raw_edf.filter(band_pass[0], band_pass[1])
    raw_edf.save(edf_file.parent /
                 ('.'.join(edf_file.name.split('.')[:-1]) + '_trim_raw.fif'))


def calc_pca_contrib(pca_input: np.ndarray, num_features: int):
    pca_fn = PCA()
    pca_fn.fit(pca_input)

    num_eig = pca_fn.components_.shape[0]
    C = np.zeros([num_eig, num_features])
    for i in range(num_eig):
        abs_coeff = np.abs(pca_fn.components_[i, :])
        C[i, :] = 100 * abs_coeff / np.sum(abs_coeff)

    pca_out = np.zeros(num_features)
    exp_var_sum = np.sum(pca_fn.explained_variance_)
    for i in range(num_features):
        pca_out[i] = np.dot(C[:, i], pca_fn.explained_variance_) / exp_var_sum

    return pca_out


def convert_to_heatmap(input: np.ndarray, rows: int, cols: int):
    heatmap_input = np.zeros([rows, cols])
    for i in range(cols):
        for j in range(rows):
            heatmap_input[j, i] = input[rows * i + j]

    return heatmap_input


def generate_spectrogram(arr, start, sample_rate, length, scale):
    print(start, sample_rate, length, scale)
    spec = np.zeros([sample_rate * length // scale, 51])  # 0-50 Hz
    pyfftw_buffer = pyfftw.empty_aligned(sample_rate, dtype="float64")
    for i in range(0, sample_rate * length, scale):
        pyfftw_buffer[:] = arr[start + i - sample_rate: start + i]
        res = pyfftw.interfaces.numpy_fft.rfft(
            pyfftw_buffer, n=sample_rate, norm=None)[:51]  # 0-50 Hz
        spec[i // scale, :] = res
    return spec


def generate_de(arr, sample_rate):
    bp_delta_sos = signal.butter(
        2, [eeg_bands.DELTA[0], eeg_bands.DELTA[1]], btype='bandpass', output='sos', fs=sample_rate)
    bp_delta_eeg = signal.sosfilt(bp_delta_sos, arr)
    de_delta = stats.differential_entropy(bp_delta_eeg)

    bp_theta_sos = signal.butter(
        2, [eeg_bands.THETA[0], eeg_bands.THETA[1]], btype='bandpass', output='sos', fs=sample_rate)
    bp_theta_eeg = signal.sosfilt(bp_theta_sos, arr)
    de_theta = stats.differential_entropy(bp_theta_eeg)

    bp_alpha_sos = signal.butter(
        2, [eeg_bands.ALPHA[0], eeg_bands.ALPHA[1]], btype='bandpass', output='sos', fs=sample_rate)
    bp_alpha_eeg = signal.sosfilt(bp_alpha_sos, arr)
    de_alpha = stats.differential_entropy(bp_alpha_eeg)

    bp_beta_sos = signal.butter(
        2, [eeg_bands.BETA[0], eeg_bands.BETA[1]], btype='bandpass', output='sos', fs=sample_rate)
    bp_beta_eeg = signal.sosfilt(bp_beta_sos, arr)
    de_beta = stats.differential_entropy(bp_beta_eeg)

    bp_gamma_sos = signal.butter(
        2, [eeg_bands.GAMMA[0], eeg_bands.GAMMA[1]], btype='bandpass', output='sos', fs=sample_rate)
    bp_gamma_eeg = signal.sosfilt(bp_gamma_sos, arr)
    de_gamma = stats.differential_entropy(bp_gamma_eeg)

    return [de_delta, de_theta, de_alpha, de_beta, de_gamma]


def normalize_signal(arr):
    """Performs Z-score normalization on array arr."""
    return (arr - np.average(arr)) / np.std(arr)


def calculate_difference(tuple_item):
    return abs(tuple_item[1] - tuple_item[0])


def convert_int64_to_int(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    else:
        return obj  # Return unchanged if not int64, list, or dict


def EEG_Channel_side_dictionary():
    eeg_channels = {
        "Left": ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'A1'],
        "Midline": ['Fz', 'Cz', 'Fpz', 'Pz'],
        "Right": ['Fp2', 'F8', 'T4', 'T6', 'O2', 'F4', 'C4', 'P4', 'A2']
    }

    return eeg_channels


def EEG_Channel_list():
    channel_list = [
        'Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'A1',
        'Fz', 'Cz', 'Fp2', 'F8', 'T4', 'T6', 'O2', 'F4', 'C4', 'P4', 'A2',
        'Fpz', 'Pz']
    return channel_list


def EEG_Channel_ipcon_list():

    all_eeg_channels = ['Fp1_Ip', 'F1_Ip', 'T1_Ip', 'T2_Ip', 'O1_Ip', 'F2_Ip', 'C1_Ip', 'P1_Ip', 'A1_Ip',
                        'Fz', 'Cz', 'Pz', 'Fpz',
                        'Fp1_Con', 'F1_Con', 'T1_Con', 'T2_Con', 'O1_Con', 'F2_Con', 'C1_Con', 'P1_Con', 'A1_Con']

    return all_eeg_channels


class EEG_Channel_ipcon_right(Enum):

    Fp1 = 'Fp1_Ip'
    F7 = 'F1_Ip'
    T3 = 'T1_Ip'
    T5 = 'T2_Ip'
    O1 = 'O1_Ip'
    F3 = 'F2_Ip'
    C3 = 'C1_Ip'
    P3 = 'P1_Ip'
    A1 = 'A1_Ip'

    Fz = 'Fz'
    Cz = 'Cz'
    Pz = 'Pz'
    Fpz = 'Fpz'

    Fp2 = 'Fp1_Con'
    F8 = 'F1_Con'
    T4 = 'T1_Con'
    T6 = 'T2_Con'
    O2 = 'O1_Con'
    F4 = 'F2_Con'
    C4 = 'C1_Con'
    P4 = 'P1_Con'
    A2 = 'A1_Con'


class EEG_Channel_ipcon_left(Enum):

    Fp1 = 'Fp1_Con'
    F7 = 'F1_Con'
    T3 = 'T1_Con'
    T5 = 'T2_Con'
    O1 = 'O1_Con'
    F3 = 'F2_Con'
    C3 = 'C1_Con'
    P3 = 'P1_Con'
    A1 = 'A1_Con'

    Fz = 'Fz'
    Cz = 'Cz'
    Pz = 'Pz'
    Fpz = 'Fpz'

    Fp2 = 'Fp1_Ip'
    F8 = 'F1_Ip'
    T4 = 'T1_Ip'
    T6 = 'T2_Ip'
    O2 = 'O1_Ip'
    F4 = 'F2_Ip'
    C4 = 'C1_Ip'
    P4 = 'P1_Ip'
    A2 = 'A1_Ip'


def ip_con_switch(cn, move_hand):

    if move_hand == "Right":
        ip_con_cn = getattr(EEG_Channel_ipcon_right, cn).value

    elif move_hand == "Left":
        ip_con_cn = getattr(EEG_Channel_ipcon_left, cn).value

    return ip_con_cn


class NormOptions(Enum):
    none = auto()
    whole = auto()
    epoch = auto()


def retreive_patient_info():

    output_dictionary = {"HC_Sham": ["pro00087153_0020", "pro00087153_0023", "pro00087153_0027", "pro00087153_0028", "pro00087153_0036"],
                         "HC_Stim": ["pro00087153_0022", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026", "pro00087153_0029", "pro00087153_0030"],
                         "CS_Sham": ["pro00087153_0013", "pro00087153_0015", "pro00087153_0017", "pro00087153_0018", "pro00087153_0021"],
                         "CS_Stim": ["pro00087153_0003", "pro00087153_0004", "pro00087153_0005", "pro00087153_0042", "pro00087153_0043"],
                         "Left": ["pro00087153_0020", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026",
                                  "pro00087153_0027", "pro00087153_0028", "pro00087153_0036", "pro00087153_0003",
                                  "pro00087153_0004", "pro00087153_0015", "pro00087153_0043"],
                         "Right": ["pro00087153_0022", "pro00087153_0023", "pro00087153_0029", "pro00087153_0030", "pro00087153_0005",
                                   "pro00087153_0013", "pro00087153_0018", "pro00087153_0017",
                                   "pro00087153_0018", "pro00087153_0021", "pro00087153_0042"]
                         }
    return output_dictionary


def stroke_patient_list():

    output_list = ["pro00087153_0020", "pro00087153_0023", "pro00087153_0027", "pro00087153_0028", "pro00087153_0036",
                   "pro00087153_0022", "pro00087153_0024", "pro00087153_0025", "pro00087153_0026", "pro00087153_0029", "pro00087153_0030",
                   "pro00087153_0013", "pro00087153_0015", "pro00087153_0017", "pro00087153_0018", "pro00087153_0021",
                   "pro00087153_0003", "pro00087153_0004", "pro00087153_0005", "pro00087153_0042", "pro00087153_0043"]
    return output_list
