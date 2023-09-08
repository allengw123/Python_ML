import csv
import glob
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import time
import xml.etree.ElementTree as ET

from env_handler import get_patient_dir
from pathlib import Path
from sklearn.decomposition import PCA

TIME = 0
EVENT = 2
INFO = 3

DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2']

# Stop value is exclusive, e.g. delta = [1, 2, 3]
DELTA = [1, 4]
THETA = [4, 9]
ALPHA = [8, 13]
BETA = [13, 30]
GAMMA = [30, 50]

start_time = time.time()

def log(msg):
    print(f'[{time.time() - start_time:.3f} s]', msg)

mne.set_log_level(verbose='WARNING')

patient_dir = get_patient_dir()
if not patient_dir.exists():
    print(f'Folder {patient_dir} does not exist. Please check your configuration settings.')
    exit(-1)

subject_name = 'pro00087153_0041' # Subject in question
subject_folder = patient_dir / subject_name
if not subject_folder.exists():
    print(f"Folder {subject_folder} does not exist. Please check your selected subject.")

log(f"Beginning to load in data from {subject_name}")

# Load data, apply AC filters
raw_edf = mne.io.read_raw_edf(subject_folder / f'{subject_name}.edf')
vr_sig = raw_edf.copy().pick_channels(['DC1'])
raw_edf.drop_channels(DROP_CHANNELS)
raw_edf.drop_channels([ch for ch in raw_edf.ch_names if ch.startswith('X')]) # Remove all dead channels
#raw_edf.load_data()
#raw_edf.notch_filter([60, 120, 180])
sample_rate = raw_edf.info['sfreq']

pwelch_res = 1
NFFT = int(sample_rate / pwelch_res)

log("Data successfully loaded")
log(f"\tUsing DC1 for VR input")
log(f"\tDropping the following channels: X*, " + ', '.join(DROP_CHANNELS))

# Grab data required for VR signal processing
vr_sig = (np.abs(vr_sig.load_data().get_data(['DC1'])) > 2.5)[0]

vr_trial_samples = [] # (start sample, stop sample)

# Grab the starting and stopping places for each VR trial
lookingForTrue = True
start = -1
for i in range(len(vr_sig)):
    if lookingForTrue and vr_sig[i]:
        start = i
        lookingForTrue = False
    elif not lookingForTrue and not vr_sig[i]:
        vr_trial_samples.append((start, i))
        start = -1
        lookingForTrue = True
    else:
        continue

log("Start and stop times of trials determined.")

hold_psd = []
prep_psd = []
reach_psd = []

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
        hold_edf.append(raw_edf.copy().crop(hold[0]/sample_rate, hold[1]/sample_rate))
        prep_edf.append(raw_edf.copy().crop(prep[0]/sample_rate, prep[1]/sample_rate))
        reach_edf.append(raw_edf.copy().crop(reach[0]/sample_rate, reach[1]/sample_rate))

    ##
    # Power spectrial densities on all epochs all electrodes via pwelch
    ##

    log("\tPerforming PSD over all hold, prep, and reach epochs in the trial")
    for i in range(len(hold_edf)):
        hold_psd.append(mne.time_frequency.psd_welch(hold_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))
        prep_psd.append(mne.time_frequency.psd_welch(prep_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))
        reach_psd.append(mne.time_frequency.psd_welch(reach_edf[i].copy().load_data(), n_fft=NFFT, n_per_seg=NFFT))

    log(f"\tFinished saving PSD information from Trial {index + 1}")

log(f"Finished saving out all PSD information from the trials, summing over frequency bands.")

all_psd = hold_psd + prep_psd + reach_psd # combine everything
num_ch = raw_edf.info['nchan']
num_features = num_ch * 4 # 5 == # freq bands in consideration
pca_input = np.zeros([len(all_psd), num_features]) 
# sum everything together, throw into pca_input... 
for i, psd in enumerate(all_psd):
    #pca_input[i, 0*num_ch:1*num_ch] = np.sum(psd[0][:, DELTA[0]:DELTA[1]], axis=1)
    pca_input[i, 0*num_ch:1*num_ch] = np.sum(psd[0][:, THETA[0]:THETA[1]], axis=1)
    pca_input[i, 1*num_ch:2*num_ch] = np.sum(psd[0][:, ALPHA[0]:ALPHA[1]], axis=1)
    pca_input[i, 2*num_ch:3*num_ch] = np.sum(psd[0][:, BETA[0]:BETA[1]], axis=1)
    pca_input[i, 3*num_ch:4*num_ch] = np.sum(psd[0][:, GAMMA[0]:GAMMA[1]], axis=1)

log(f"Finished setting up input to PCA")

pca_fn = PCA()
pca_fn.fit(pca_input)

log(f"Finished calculating PCA")

C = np.zeros([num_features, num_features])
for i, eig_vec in enumerate(pca_fn.components_):
    abs_eig_vec = np.abs(eig_vec)
    C[i, :] = 100 * abs_eig_vec / np.sum(abs_eig_vec)

pca_out = np.zeros(num_features)
for i in range(num_features):
    pca_out[i] = np.dot(C[:, i], pca_fn.explained_variance_) / np.sum(pca_fn.explained_variance_)

heatmap_input = np.zeros([num_ch, 5])
for i in range(4):
    for j in range(num_ch):
        heatmap_input[j, i] = pca_out[num_ch * i + j]

log(f"Finished generating heatmap, now plotting.")

plt.figure()
plt.imshow(heatmap_input, cmap='seismic', interpolation='nearest')
plt.colorbar()
plt.xlabel('Frequency Band')
plt.xticks(ticks=[i for i in range(4)], labels=[r"$\theta$ (4-8 Hz)", r"$\alpha$ (8-12 Hz)", r"$\beta$ (13-29 Hz)", r"$low\gamma$ (30-50 Hz)"], rotation=45)
plt.ylabel('Electrode')
plt.yticks(ticks=[i for i in range(num_ch)], labels=raw_edf.ch_names)
plt.title("PCA contribution (%) from Hold, Prep, Reach Epochs")
plt.savefig('output/output.svg', bbox_inches='tight')