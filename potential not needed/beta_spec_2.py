import csv
import glob
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import pathlib
import xml.etree.ElementTree as ET

TIME = 0
EVENT = 2
INFO = 3

UP_RIGHT = 0
UP_LEFT = 1
DOWN_RIGHT = 2
DOWN_LEFT = 3

print('WARNING!!! NOT DESIGNED TO WORK ON GITHUB YET!!!')

mne.set_log_level(verbose='WARNING')

subject_name = 'pro00087153_0040'
subject_folder = f'Subject_Data/{subject_name}' # glob folder and iterate over all?

# Load data, apply AC filters
raw_edf = mne.io.read_raw_edf(subject_folder + f'/{subject_name}.edf')
raw_edf.load_data()
raw_edf.pick_channels(['C3', 'C4', 'DC1']) # keep channels for motor cortex & vr input
raw_edf.notch_filter([60, 120, 180])
sample_rate = raw_edf.info['sfreq']

pwelch_res = 1
NFFT = int(sample_rate / pwelch_res)

print("Successfully loaded data")

# Grab data required for VR signal processing
vr_sig = (np.abs(raw_edf.get_data(['DC1'])) > 2.5)[0]

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

print("Trials separated")

# Go through ALL of the TRIALS that are ACCEPTED
for index, path in enumerate([pathlib.Path(x) for x in glob.glob(subject_folder + '/vr/*')]):
    print("Working on Trial", index + 1)

    trial_xml = ET.parse(path/'trial information.xml')
    if trial_xml.find('quality').text != 'accepted' or trial_xml.find('studyCondition').text != 'Single reach per cue, random time, no visual cue':
        print("Invalid trial, continuing.")
        continue

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
            
            [x, y, z] = list(map(float, i[INFO].split('/')[-1][1:].split('|')))
            if x > 0 and y > 0:
                event_target_loc.append(UP_RIGHT)
            elif x > 0 and y < 0:
                event_target_loc.append(DOWN_RIGHT)
            elif x < 0 and y > 0:
                event_target_loc.append(UP_LEFT)
            elif x < 0 and y < 0:
                event_target_loc.append(DOWN_LEFT)
            else:
                event_target_loc.append(-1)

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

    for hold, prep, reach in zip(hold_events, prep_events, reach_events):
        hold_edf.append(raw_edf.copy().crop(hold[0]/sample_rate, hold[1]/sample_rate))
        prep_edf.append(raw_edf.copy().crop(prep[0]/sample_rate, prep[1]/sample_rate))
        reach_edf.append(raw_edf.copy().crop(reach[0]/sample_rate, reach[1]/sample_rate))

    ##
    # Power spectrial densities on all epochs all electrodes via pwelch
    ##

    hold_psd = []
    prep_psd = []
    reach_psd = []

    for i in range(len(hold_edf)):
        hold_psd.append(mne.time_frequency.psd_welch(hold_edf[i], fmin=8, fmax=40, n_fft=NFFT, n_per_seg=NFFT))
        prep_psd.append(mne.time_frequency.psd_welch(prep_edf[i], fmin=8, fmax=40, n_fft=NFFT, n_per_seg=NFFT))
        reach_psd.append(mne.time_frequency.psd_welch(reach_edf[i], fmin=8, fmax=40, n_fft=NFFT, n_per_seg=NFFT))

    ##
    # Average epoch's psd across like phase (hold w hold, prep w prep, reach w reach)
    ##

    C3_channel = raw_edf.ch_names.index('C3')
    C4_channel = raw_edf.ch_names.index('C4')

    up_left_prep_avg = np.zeros(hold_psd[0][0].shape)
    up_left_reach_avg = np.zeros(hold_psd[0][0].shape)
    up_left_cnt = 0
    up_right_prep_avg = np.zeros(hold_psd[0][0].shape)
    up_right_reach_avg = np.zeros(hold_psd[0][0].shape)
    up_right_cnt = 0
    down_left_prep_avg = np.zeros(hold_psd[0][0].shape)
    down_left_reach_avg = np.zeros(hold_psd[0][0].shape)
    down_left_cnt = 0
    down_right_prep_avg = np.zeros(hold_psd[0][0].shape)
    down_right_reach_avg = np.zeros(hold_psd[0][0].shape)
    down_right_cnt = 0

    prep_psd_avg = np.zeros(hold_psd[0][0].shape)
    reach_psd_avg = np.zeros(hold_psd[0][0].shape)

    for i in range(0, len(hold_edf)):
        prep_delta = prep_psd[i][0] - hold_psd[i][0]
        reach_delta = reach_psd[i][0] - hold_psd[i][0]

        if event_target_loc[i] == UP_LEFT:
            up_left_prep_avg += prep_delta
            up_left_reach_avg += reach_delta
            up_left_cnt += 1
        elif event_target_loc[i] == UP_RIGHT:
            up_right_prep_avg += prep_delta
            up_right_reach_avg += reach_delta
            up_right_cnt += 1
        elif event_target_loc[i] == DOWN_LEFT:
            down_left_prep_avg += prep_delta
            down_left_reach_avg += reach_delta
            down_left_cnt += 1
        elif event_target_loc[i] == DOWN_RIGHT:
            down_right_prep_avg += prep_delta
            down_right_reach_avg += reach_delta
            down_right_cnt += 1
        else:
            continue
        
        prep_psd_avg += prep_delta
        reach_psd_avg += reach_delta
    
    up_left_prep_avg /= up_left_cnt
    up_left_reach_avg /= up_left_cnt
    up_right_prep_avg /= up_right_cnt
    up_right_reach_avg /= up_right_cnt
    down_left_prep_avg /= down_left_cnt
    down_left_reach_avg /= down_left_cnt
    down_right_prep_avg /= down_right_cnt
    down_right_reach_avg /= down_right_cnt
    prep_psd_avg /= len(hold_edf)
    reach_psd_avg /= len(hold_edf)

    plt.title(f"{trial_name} - C3 (Prep)")
    plt.plot(hold_psd[0][1], up_left_prep_avg[C3_channel])
    plt.plot(hold_psd[0][1], up_right_prep_avg[C3_channel])
    plt.plot(hold_psd[0][1], down_left_prep_avg[C3_channel])
    plt.plot(hold_psd[0][1], down_right_prep_avg[C3_channel])
    plt.legend(['up-left', 'up-right', 'down-left', 'down-right'])
    plt.xlim(8, 40)
    plt.savefig(f"{trial_name}_c3_prep.png")
    plt.close()

    plt.title(f"{trial_name} - C4 (Prep)")
    plt.plot(hold_psd[0][1], up_left_prep_avg[C4_channel])
    plt.plot(hold_psd[0][1], up_right_prep_avg[C4_channel])
    plt.plot(hold_psd[0][1], down_left_prep_avg[C4_channel])
    plt.plot(hold_psd[0][1], down_right_prep_avg[C4_channel])
    plt.legend(['up-left', 'up-right', 'down-left', 'down-right'])
    plt.xlim(8, 40)
    plt.savefig(f"{trial_name}_c4_prep.png")
    plt.close()

    plt.title(f"{trial_name} - C3 (Reach)")
    plt.plot(hold_psd[0][1], up_left_reach_avg[C3_channel])
    plt.plot(hold_psd[0][1], up_right_reach_avg[C3_channel])
    plt.plot(hold_psd[0][1], down_left_reach_avg[C3_channel])
    plt.plot(hold_psd[0][1], down_right_reach_avg[C3_channel])
    plt.legend(['up-left', 'up-right', 'down-left', 'down-right'])
    plt.xlim(8, 40)
    plt.savefig(f"{trial_name}_c3_reach.png")
    plt.close()

    plt.title(f"{trial_name} - C4 (Reach)")
    plt.plot(hold_psd[0][1], up_left_reach_avg[C4_channel])
    plt.plot(hold_psd[0][1], up_right_reach_avg[C4_channel])
    plt.plot(hold_psd[0][1], down_left_reach_avg[C4_channel])
    plt.plot(hold_psd[0][1], down_right_reach_avg[C4_channel])
    plt.legend(['up-left', 'up-right', 'down-left', 'down-right'])
    plt.xlim(8, 40)
    plt.savefig(f"{trial_name}_c4_reach.png")
    plt.close()

    plt.title(f"{trial_name} - Beta Change Separated by Channel and Action")
    plt.plot(hold_psd[0][1], prep_psd_avg[C3_channel])
    plt.plot(hold_psd[0][1], prep_psd_avg[C4_channel])
    plt.plot(hold_psd[0][1], reach_psd_avg[C3_channel])
    plt.plot(hold_psd[0][1], reach_psd_avg[C4_channel])
    plt.legend(['C3 Prep', 'C4 Prep', 'C3 Reach', 'C4 Reach'])
    plt.xlim(8, 40)
    plt.savefig(f"{trial_name}_gen_beta.png")
    plt.close()

    print("Finished Trial", index + 1)