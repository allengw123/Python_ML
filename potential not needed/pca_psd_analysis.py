import csv
import glob
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import time
import xml.etree.ElementTree as ET

from env_handler import get_patient_dir
from pathlib import Path
from sklearn.decomposition import PCA
from eeg_utils import *

TIME = 0
EVENT = 2
INFO = 3

DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2', 'Fp1', 'Fp2']
NOTCH_FREQS = [60, 120, 180]

# Stop value is exclusive, e.g. delta = [1, 2, 3]
DELTA = [1, 4]
THETA = [4, 9]
ALPHA = [8, 13]
BETA = [13, 30]
GAMMA = [30, 50]

def plot_pca_heatmap(heatmap_input, patient_name, plot_title, file_name, trial=None):
    output_path = Path('output') / patient_name / (str(trial) if trial else '') / 'pca' / file_name
    if not output_path.parent.is_dir():
        Path.mkdir(output_path.parent, parents=True)

    plt.figure()
    plt.imshow(heatmap_input, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Frequency Band')
    plt.xticks(ticks=[i for i in range(4)], labels=[r"$\theta$ (4-8 Hz)", r"$\alpha$ (8-12 Hz)", r"$\beta$ (13-29 Hz)", r"$low\gamma$ (30-50 Hz)"], rotation=45)
    plt.ylabel('Electrode')
    plt.yticks(ticks=[i for i in range(num_ch)], labels=trim_fif.ch_names)
    plt.title(f"PCA contribution (%) from {plot_title} Epochs")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

start_time = time.time()

def log(msg):
    print(f'[{time.time() - start_time:.3f} s]', msg)

mne.set_log_level(verbose='WARNING')

patient_dir = get_patient_dir()
if not patient_dir.exists():
    print(f'Folder {patient_dir} does not exist. Please check your configuration settings.')
    exit(-1)

# Subject in question

#subject_name = 'pro00087153_0034' # right arm stim
#subject_name = 'pro00087153_0036' # right arm sham
subject_name = 'pro00087153_0041' # right arm sham
subject_folder = patient_dir / subject_name
if not subject_folder.exists():
    print(f"Folder {subject_folder} does not exist. Please check your selected subject.")
    exit(-1)

log(f"Beginning to load in data from {subject_name}")

edf_file_path = subject_folder / f'{subject_name}.edf'
fif_file_path = subject_folder / f'{subject_name}_trim_raw.fif'

# Load data, apply AC filters
if not fif_file_path.is_file():
    log(f"Trimming base EEG file")
    output_trim_eeg(edf_file_path, DROP_CHANNELS, NOTCH_FREQS)
else:
    log(f"Using found trimmed EEG file")

full_edf = mne.io.read_raw_edf(edf_file_path)
trim_fif = mne.io.read_raw_fif(fif_file_path)

num_ch = trim_fif.info['nchan']
sample_rate = trim_fif.info['sfreq']
pwelch_res = 1
NFFT = int(sample_rate / pwelch_res)

log("Data successfully loaded")
log(f"\tUsing DC1 for VR input")
log(f"\tDropping the following channels: X*, " + ', '.join(DROP_CHANNELS))

# Grab data required for VR signal processing
vr_trial_samples = get_trial_samples(full_edf, 'DC1')

log("Start and stop times of trials determined.")

saved_trials = 0

# Go through ALL of the TRIALS that are ACCEPTED
for index, path in enumerate([Path(x) for x in glob.glob(str(subject_folder / 'vr' / '*'))]):
    log(f"Working on Trial {index + 1}")

    trial_xml = ET.parse(path/'trial information.xml')
    if trial_xml.find('quality').text != 'accepted' or trial_xml.find('studyCondition').text != 'Single reach per cue, random time, no visual cue':
        log("\tNon-reach trial, continuing.")
        continue
    else:
        saved_trials += 1

    trial_name = trial_xml.find('trialName').text

    # Parse through Events.csv to get each trial
    with open(path/'Events.csv', 'r') as fp:
        csv_reader = csv.reader(fp)
        raw_events = [i for i in csv_reader][1:]

    base = vr_trial_samples[index][0]
    hold_events = []
    prep_events = []
    reach_events = []
    event_target_loc = []

    # atStartPosition -> cueEvent (HOLD)
    # cueEvent -> targetUp (PREP)
    # targetUp -> targetHit (REACH)

    hold_start = -1
    prep_start = -1
    reach_start = -1
    for i in raw_events:
        if i[EVENT] == "atStartPosition":
            hold_start = base + math.ceil(float(i[TIME]) * sample_rate)
        elif i[EVENT] == "cueEvent":
            prep_start = base + math.ceil(float(i[TIME]) * sample_rate)
            hold_events.append((hold_start, prep_start))
            hold_start = -1
        elif i[EVENT] == "targetUp":
            reach_start = base + math.ceil(float(i[TIME]) * sample_rate)
            prep_events.append((prep_start, reach_start))
            prep_start = -1
        elif i[EVENT] == "targetHit":
            reach_events.append((reach_start, base + math.ceil(float(i[TIME]) * sample_rate)))
            reach_start = -1
        else:
            continue

    ##
    # Separate into epochs (12 trials: {hold, prep, reach}, 36 total epochs)
    ##

    hold_edf = []
    prep_edf = []
    reach_edf = []

    log("\tDetermining start and stop timestamps for hold, prep, and reach epochs")
    for hold, prep, reach in zip(hold_events, prep_events, reach_events):
        hold_edf.append(trim_fif.copy().crop(hold[0]/sample_rate, hold[1]/sample_rate))
        prep_edf.append(trim_fif.copy().crop(prep[0]/sample_rate, prep[1]/sample_rate))
        reach_edf.append(trim_fif.copy().crop(reach[0]/sample_rate, reach[1]/sample_rate))

    ##
    # Power spectrial densities on all epochs all electrodes via pwelch
    ##
    num_events = len(hold_edf)
    hold_psd = []
    prep_psd = []
    reach_psd = []

    log("\tPerforming PSD over all hold, prep, and reach epochs in the trial")
    for i in range(num_events):
        hold_psd.append(mne.time_frequency.psd_welch(hold_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))
        prep_psd.append(mne.time_frequency.psd_welch(prep_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))
        reach_psd.append(mne.time_frequency.psd_welch(reach_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))

    log(f"\tFinished saving PSD information from Trial {index + 1}")

    num_events = len(hold_psd)
    num_freq_bands = 4
    num_features = num_ch * num_freq_bands # freq bands in consideration
    hold_psd_sum = np.zeros([num_events, num_features])
    prep_psd_sum = np.zeros([num_events, num_features])
    reach_psd_sum = np.zeros([num_events, num_features])

    log(f"Finished saving out all PSD information from the trials, summing over frequency bands.")

    for i in range(num_events):
        hold_psd_sum[i, 0*num_ch:1*num_ch] = np.sum(hold_psd[i][0][:, THETA[0]:THETA[1]], axis=1)
        hold_psd_sum[i, 1*num_ch:2*num_ch] = np.sum(hold_psd[i][0][:, ALPHA[0]:ALPHA[1]], axis=1)
        hold_psd_sum[i, 2*num_ch:3*num_ch] = np.sum(hold_psd[i][0][:, BETA[0]:BETA[1]], axis=1)
        hold_psd_sum[i, 3*num_ch:4*num_ch] = np.sum(hold_psd[i][0][:, GAMMA[0]:GAMMA[1]], axis=1)

        prep_psd_sum[i, 0*num_ch:1*num_ch] = np.sum(prep_psd[i][0][:, THETA[0]:THETA[1]], axis=1)
        prep_psd_sum[i, 1*num_ch:2*num_ch] = np.sum(prep_psd[i][0][:, ALPHA[0]:ALPHA[1]], axis=1)
        prep_psd_sum[i, 2*num_ch:3*num_ch] = np.sum(prep_psd[i][0][:, BETA[0]:BETA[1]], axis=1)
        prep_psd_sum[i, 3*num_ch:4*num_ch] = np.sum(prep_psd[i][0][:, GAMMA[0]:GAMMA[1]], axis=1)

        reach_psd_sum[i, 0*num_ch:1*num_ch] = np.sum(reach_psd[i][0][:, THETA[0]:THETA[1]], axis=1)
        reach_psd_sum[i, 1*num_ch:2*num_ch] = np.sum(reach_psd[i][0][:, ALPHA[0]:ALPHA[1]], axis=1)
        reach_psd_sum[i, 2*num_ch:3*num_ch] = np.sum(reach_psd[i][0][:, BETA[0]:BETA[1]], axis=1)
        reach_psd_sum[i, 3*num_ch:4*num_ch] = np.sum(reach_psd[i][0][:, GAMMA[0]:GAMMA[1]], axis=1)

    log(f"\tFinished setting up input to PCA")

    log(f"\tRunning hold")
    pca_out = calc_pca_contrib(hold_psd_sum, num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Hold', 'psd_hold.pdf', index + 1)

    log(f"\tRunning prep")
    pca_out = calc_pca_contrib(prep_psd_sum, num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Prep', 'psd_prep.pdf', index + 1)

    log(f"\tRunning reach")
    pca_out = calc_pca_contrib(reach_psd_sum, num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Reach', 'psd_reach.pdf', index + 1)

    log(f"\tRunning hold-prep")
    pca_out = calc_pca_contrib(np.vstack([hold_psd_sum, prep_psd_sum]), num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Hold, Prep', 'psd_hold_prep.pdf', index + 1)

    log(f"\tRunning hold-reach")
    pca_out = calc_pca_contrib(np.vstack([hold_psd_sum, reach_psd_sum]), num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Hold, Reach', 'psd_hold_reach.pdf', index + 1)

    log(f"\tRunning prep-reach")
    pca_out = calc_pca_contrib(np.vstack([prep_psd_sum, reach_psd_sum]), num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Prep, Reach', 'psd_prep_reach.pdf', index + 1)

    log(f"\tRunning hold-prep-reach")
    pca_out = calc_pca_contrib(np.vstack([hold_psd_sum, prep_psd_sum, reach_psd_sum]), num_features)
    heatmap_input = convert_to_heatmap(pca_out, num_ch, num_freq_bands)
    plot_pca_heatmap(heatmap_input, subject_name, 'Hold, Prep, Reach', 'psd_hold_prep_reach.pdf', index + 1)

    log(f"Plotting PSD output for each electrode")
    hold_psd_avg = np.sum(hold_psd_sum, axis=0) / hold_psd_sum.shape[0]
    prep_psd_avg = np.sum(prep_psd_sum, axis=0) / prep_psd_sum.shape[0]
    reach_psd_avg = np.sum(reach_psd_sum, axis=0) / reach_psd_sum.shape[0]
    psd_labels = ['Hold', 'Prep', 'Reach']
    psd_label_index = 3*np.arange(len(psd_labels))
    plot_width = 0.5
    for i in range(len(trim_fif.ch_names)):
        output_path = Path('output') / subject_name / str(index + 1) / 'psd_plots' / f'{trim_fif.ch_names[i]}_psd.pdf'
        if not output_path.parent.is_dir():
            Path.mkdir(output_path.parent, parents=True)
        fig, ax = plt.subplots()
        theta = ax.bar(psd_label_index-.75, [hold_psd_avg[i], prep_psd_avg[i], reach_psd_avg[i]], plot_width, label=r'$\theta$ (4-8 Hz)')
        alpha = ax.bar(psd_label_index-.25, [hold_psd_avg[num_ch + i], prep_psd_avg[num_ch + i], reach_psd_avg[num_ch + i]], plot_width, label=r'$\alpha$ (8-12 Hz)')
        beta = ax.bar(psd_label_index+.25, [hold_psd_avg[2 * num_ch + i], prep_psd_avg[2 * num_ch + i], reach_psd_avg[2 * num_ch + i]], plot_width, label=r'$\beta$ (13-29 Hz)')
        gamma = ax.bar(psd_label_index+.75, [hold_psd_avg[3 * num_ch + i], prep_psd_avg[3 * num_ch + i], reach_psd_avg[3 * num_ch + i]], plot_width, label=r'$low\gamma$ (30-50 Hz)')
        ax.set_ylabel('PSD Power')
        ax.set_xlabel('Phase / Frequency Band')
        ax.set_xticks(psd_label_index)
        ax.set_xticklabels(psd_labels)
        ax.set_title(f'PSD Power by Phase in Trial {index + 1} on Electrode {trim_fif.ch_names[i]}')
        ax.legend()
        ax.bar_label(theta)
        ax.bar_label(alpha)
        ax.bar_label(beta)
        ax.bar_label(gamma)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    """
    log(f"Plotting spectrogram for each electrode")
    for i in range(len(trim_fif.ch_names)):
        for j in range(len(hold_edf)):
            output_path = Path('output') / subject_name / str(index + 1) / 'spectrograms' / f'hold_{j+1}_{trim_fif.ch_names[i]}_spectrogram.pdf'
            if not output_path.parent.is_dir():
                Path.mkdir(output_path.parent, parents=True)
            spec_f, spec_t, spec_Sxx = scipy.signal.spectrogram(hold_edf[j].get_data(trim_fif.ch_names[i]), sample_rate)
            plt.figure()
            plt.pcolormesh(spec_t, spec_f, spec_Sxx[0], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim([0, 50])
            plt.title(f'Spectrogram for Hold from Trial {index + 1} - Event {j + 1} on Electrode {trim_fif.ch_names[i]}')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        for j in range(len(prep_edf)):
            output_path = Path('output') / subject_name / str(index + 1) / 'spectrograms' / f'prep_{j+1}_{trim_fif.ch_names[i]}_spectrogram.pdf'
            if not output_path.parent.is_dir():
                Path.mkdir(output_path.parent, parents=True)
            spec_f, spec_t, spec_Sxx = scipy.signal.spectrogram(prep_edf[j].get_data(trim_fif.ch_names[i]), sample_rate)
            plt.figure()
            plt.pcolormesh(spec_t, spec_f, spec_Sxx[0], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim([0, 50])
            plt.title(f'Spectrogram for Prep from Trial {index + 1} - Event {j + 1} on Electrode {trim_fif.ch_names[i]}')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        for j in range(len(reach_edf)):
            output_path = Path('output') / subject_name / str(index + 1) / 'spectrograms' / f'reach_{j+1}_{trim_fif.ch_names[i]}_spectrogram.pdf'
            if not output_path.parent.is_dir():
                Path.mkdir(output_path.parent, parents=True)
            spec_f, spec_t, spec_Sxx = scipy.signal.spectrogram(reach_edf[j].get_data(trim_fif.ch_names[i]), sample_rate)
            plt.figure()
            plt.pcolormesh(spec_t, spec_f, spec_Sxx[0], shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim([0, 50])
            plt.title(f'Spectrogram for Reach from Trial {index + 1} - Event {j + 1} on Electrode {trim_fif.ch_names[i]}')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
    """
log(f"Done")