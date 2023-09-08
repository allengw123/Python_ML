import json
import math
import mne

from eeg_utils import *
from env_handler import *
from scipy import signal
from enum import Enum, auto

class NormOptions(Enum):
    none = auto()
    whole = auto()
    epoch = auto()

patient_dir = get_patient_dir()
subject_name = 'pro00087153_0040' # right arm sham
eeg_channels = ['C3', 'C4']
sample_rate_scale = 2 # sample_rate / sample_rate_scale == new sample rate for spectrogram
num_seconds = 8 # number of seconds of data to save in spec and eeg
normalize = NormOptions.whole

subject_folder = patient_dir / subject_name
if not subject_folder.exists():
    raise Exception(f"Folder {subject_folder} does not exist. Please check your configuration.")

edf_file_path = subject_folder / f'{subject_name}.edf'

mne.set_log_level(verbose='WARNING')

edf = mne.io.read_raw_edf(edf_file_path)
num_ch = edf.info['nchan']
sample_rate = int(edf.info['sfreq'])
spec_sample_rate = sample_rate // sample_rate_scale

vr_trial_samples = get_trial_samples(edf, 'DC1')

# all_x_events in form of [[(start, stop), (start, stop)], [], []] where start, stop are sample indices
all_hold_events, all_prep_events, all_reach_events, saved_trial_numbers, all_events, trial_names = parse_trial_timestamps(subject_folder, vr_trial_samples, sample_rate)

output_folder = Path('output') / subject_name
output_folder.mkdir(parents=True, exist_ok=True)
output_metadata = {
    'spec_sample_rate': spec_sample_rate,
    'eeg_sample_rate': sample_rate,
    'channels': eeg_channels,
    'seconds': num_seconds,
    'normalization': normalize.name}

num_eeg_samples = sample_rate * num_seconds
num_spec_samples = spec_sample_rate * num_seconds

eeg_data = {}
for ch in eeg_channels:
    eeg_data[ch] = edf.get_data([ch])[0]
    if normalize == NormOptions.whole:
        eeg_data[ch] = normalize_signal(eeg_data[ch])

# export as output_folder / name / hold / id_ch_{eeg,spec}.?
for i, name in enumerate(trial_names):
    output_metadata[name] = {}

    # handle all the hold events
    output_metadata[name]['hold'] = {}
    (output_folder / name / 'hold').mkdir(parents=True, exist_ok=True)
    for id, (start, stop) in enumerate(all_hold_events[i]):
        base_sample = math.floor(start - ((num_eeg_samples - (stop - start)) / 2))
        output_metadata[name]['hold'][id] = {
            'base_sample': base_sample,
            'end_sample': base_sample + num_eeg_samples,
            'start_event': start,
            'stop_event': stop,
            'start_event_spec': math.floor((start - base_sample) / sample_rate_scale),
            'stop_event_spec': math.ceil((stop - base_sample) / sample_rate_scale)
        }

        for ch in eeg_channels:
            trim_eeg = eeg_data[ch][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
            if normalize == NormOptions.epoch:
                trim_eeg = normalize_signal(trim_eeg)
            with (output_folder / name / 'hold' / f'{id}_{ch}_eeg.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in trim_eeg))
            spec = generate_spectrogram(eeg_data[ch], start, sample_rate, num_seconds, sample_rate_scale)
            with (output_folder / name / 'hold' / f'{id}_{ch}_spec.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(','.join(str(col) for col in row) for row in spec))
            psd = signal.welch(eeg_data[ch][start : stop], fs=sample_rate, nfft=sample_rate)
            with (output_folder / name / 'hold' / f'{id}_{ch}_psd.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd[1][:51]))
            psd_delta = np.sum(psd[1][1:4])
            psd_theta = np.sum(psd[1][4:8])
            psd_alpha = np.sum(psd[1][8:13])
            psd_beta  = np.sum(psd[1][13:31])
            psd_gamma = np.sum(psd[1][31:51])
            psd_bin = np.array([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])
            with (output_folder / name / 'hold' / f'{id}_{ch}_psdbin.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd_bin))
            de_array = generate_de(trim_eeg, sample_rate)
            with (output_folder / name / 'hold' / f'{id}_{ch}_de.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in de_array))
        c3_data = eeg_data['C3'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        c4_data = eeg_data['C4'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        _, coh = signal.coherence(c3_data, c4_data, fs=sample_rate, nfft=sample_rate)
        with (output_folder / name / 'hold' / f'{id}_C3C4_coh.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh))
        coh_delta = np.sum(coh[1:4])
        coh_theta = np.sum(coh[4:8])
        coh_alpha = np.sum(coh[8:13])
        coh_beta  = np.sum(coh[13:31])
        coh_gamma = np.sum(coh[31:51])
        coh_bin = np.array([coh_delta, coh_theta, coh_alpha, coh_beta, coh_gamma])
        with (output_folder / name / 'hold' / f'{id}_C3C4_cohbin.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh_bin))

    # handle all the prep events
    output_metadata[name]['prep'] = {}
    (output_folder / name / 'prep').mkdir(parents=True, exist_ok=True)
    for id, (start, stop) in enumerate(all_prep_events[i]):
        base_sample = math.floor(start - ((num_eeg_samples - (stop - start)) / 2))
        output_metadata[name]['prep'][id] = {
            'base_sample': base_sample,
            'end_sample': base_sample + num_eeg_samples,
            'start_event': start,
            'stop_event': stop,
            'start_event_spec': math.floor((start - base_sample) / sample_rate_scale),
            'stop_event_spec': math.ceil((stop - base_sample) / sample_rate_scale)
        }

        for ch in eeg_channels:
            trim_eeg = eeg_data[ch][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
            if normalize == NormOptions.epoch:
                trim_eeg = normalize_signal(trim_eeg)
            with (output_folder / name / 'prep' / f'{id}_{ch}_eeg.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in trim_eeg))
            spec = generate_spectrogram(eeg_data[ch], start, sample_rate, num_seconds, sample_rate_scale)
            with (output_folder / name / 'prep' / f'{id}_{ch}_spec.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(','.join(str(col) for col in row) for row in spec))
            psd = signal.welch(eeg_data[ch][start : stop], fs=sample_rate, nfft=sample_rate)
            with (output_folder / name / 'prep' / f'{id}_{ch}_psd.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd[1][:51]))
            psd_delta = np.sum(psd[1][1:4])
            psd_theta = np.sum(psd[1][4:8])
            psd_alpha = np.sum(psd[1][8:13])
            psd_beta  = np.sum(psd[1][13:31])
            psd_gamma = np.sum(psd[1][31:51])
            psd_bin = np.array([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])
            with (output_folder / name / 'prep' / f'{id}_{ch}_psdbin.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd_bin))
            de_array = generate_de(trim_eeg, sample_rate)
            with (output_folder / name / 'prep' / f'{id}_{ch}_de.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in de_array))
        c3_data = eeg_data['C3'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        c4_data = eeg_data['C4'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        _, coh = signal.coherence(c3_data, c4_data, fs=sample_rate, nfft=sample_rate)
        with (output_folder / name / 'prep' / f'{id}_C3C4_coh.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh))
        coh_delta = np.sum(coh[1:4])
        coh_theta = np.sum(coh[4:8])
        coh_alpha = np.sum(coh[8:13])
        coh_beta  = np.sum(coh[13:31])
        coh_gamma = np.sum(coh[31:51])
        coh_bin = np.array([coh_delta, coh_theta, coh_alpha, coh_beta, coh_gamma])
        with (output_folder / name / 'prep' / f'{id}_C3C4_cohbin.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh_bin))

    # handle all the reach events
    output_metadata[name]['reach'] = {}
    (output_folder / name / 'reach').mkdir(parents=True, exist_ok=True)
    for id, (start, stop) in enumerate(all_reach_events[i]):
        base_sample = math.floor(start - ((num_eeg_samples - (stop - start)) / 2))
        output_metadata[name]['reach'][id] = {
            'base_sample': base_sample,
            'end_sample': base_sample + num_eeg_samples,
            'start_event': start,
            'stop_event': stop,
            'start_event_spec': math.floor((start - base_sample) / sample_rate_scale),
            'stop_event_spec': math.ceil((stop - base_sample) / sample_rate_scale)
        }

        for ch in eeg_channels:
            trim_eeg = eeg_data[ch][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
            if normalize == NormOptions.epoch:
                trim_eeg = normalize_signal(trim_eeg)
            with (output_folder / name / 'reach' / f'{id}_{ch}_eeg.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in trim_eeg))
            spec = generate_spectrogram(eeg_data[ch], start, sample_rate, num_seconds, sample_rate_scale)
            with (output_folder / name / 'reach' / f'{id}_{ch}_spec.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(','.join(str(col) for col in row) for row in spec))
            psd = signal.welch(eeg_data[ch][start : stop], fs=sample_rate, nfft=sample_rate)
            with (output_folder / name / 'reach' / f'{id}_{ch}_psd.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd[1][:51]))
            psd_delta = np.sum(psd[1][1:4])
            psd_theta = np.sum(psd[1][4:8])
            psd_alpha = np.sum(psd[1][8:13])
            psd_beta  = np.sum(psd[1][13:31])
            psd_gamma = np.sum(psd[1][31:51])
            psd_bin = np.array([psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma])
            with (output_folder / name / 'reach' / f'{id}_{ch}_psdbin.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in psd_bin))
            de_array = generate_de(trim_eeg, sample_rate)
            with (output_folder / name / 'reach' / f'{id}_{ch}_de.csv').open('w') as fp:
                print(fp.name)
                fp.write('\n'.join(str(x) for x in de_array))
        c3_data = eeg_data['C3'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        c4_data = eeg_data['C4'][base_sample : base_sample + num_eeg_samples : sample_rate_scale]
        _, coh = signal.coherence(c3_data, c4_data, fs=sample_rate, nfft=sample_rate)
        with (output_folder / name / 'reach' / f'{id}_C3C4_coh.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh))
        coh_delta = np.sum(coh[1:4])
        coh_theta = np.sum(coh[4:8])
        coh_alpha = np.sum(coh[8:13])
        coh_beta  = np.sum(coh[13:31])
        coh_gamma = np.sum(coh[31:51])
        coh_bin = np.array([coh_delta, coh_theta, coh_alpha, coh_beta, coh_gamma])
        with (output_folder / name / 'reach' / f'{id}_C3C4_cohbin.csv').open('w') as fp:
            print(fp.name)
            fp.write('\n'.join(str(x) for x in coh_bin))

# output the json file
with (output_folder / 'file_metadata.json').open('w') as fp:
    json.dump(output_metadata, fp, indent=2)

print(f"Done! Look into {output_folder} for all outputted data!")