import numpy as np
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from eeg_utils import EEG_Ch
from pathlib import Path
from tqdm import tqdm

class Codes:
    rest_base = 1
    roll_base = 2
    laf = 3
    raf = 4
    rest = 5

def five_seconds(string):
    print(f'Beginning {string} in ', end='')
    for i in range(5, 0, -1):
        print(f'{i}...')
        time.sleep(1)
    print(f'Starting {string}')

print('Starting board connection.')
params = BrainFlowInputParams()
params.serial_port = 'COM3'
BoardShim.enable_dev_board_logger()
board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream()
sample_rate = board.get_sampling_rate(board.board_id)

print('Here we are performing data collection. We will perform the following:')
print('\t5 seconds of unrecorded rest after enter is hit')
print('\t15 seconds of recorded rest / eye blinks')
print('\t15 seconds of head tilt and roll')
print('\t10 trials of the following:')
print('\t\t5 seconds of left arm flexion')
print('\t\t5 seconds of recorded rest')
print('\t\t5 seconds of right arm flexion')
print('\t\t5 seconds of recorded rest')
print('\t\t5 seconds of unrecorded rest\n')

save_name = input('Please type a name for this recording: ').replace('/', '').replace('\\', '').replace(' ', '_')
output_folder = Path('output') / save_name
output_folder.mkdir(parents=True, exist_ok=True)
input('Please hit the enter key when you are ready to start the trial.\n')

# Clear the buffer
board.get_board_data()

five_seconds('15 seconds of recorded rest')
for s in tqdm(range(15), ncols=100):
    time.sleep(1)
rest_baseline = board.get_board_data()

c3_rest_base = rest_baseline[EEG_Ch.C3.value, :]
c4_rest_base = rest_baseline[EEG_Ch.C4.value, :]
rest_event_base = np.zeros(c3_rest_base.size)
rest_event_base[- sample_rate * 15] = Codes.rest_base
rest_event_base[-1] = Codes.rest_base

five_seconds('15 seconds of head rolling')
for s in tqdm(range(15), ncols=100):
    time.sleep(1)
roll_baseline = board.get_board_data()

c3_roll_base = roll_baseline[EEG_Ch.C3.value, :]
c4_roll_base = roll_baseline[EEG_Ch.C4.value, :]
roll_event_base = np.zeros(c3_roll_base.size)
roll_event_base[- sample_rate * 15] = Codes.roll_base
roll_event_base[-1] = Codes.roll_base

c3_data_stream = [c3_rest_base, c3_roll_base]
c4_data_stream = [c4_rest_base, c4_roll_base]
event_data_stream = [rest_event_base, roll_event_base]

for trial_num in range(1, 11):
    five_seconds(f'Trial {trial_num}, left arm flexions for 5s (5s rest after)')
    for s in tqdm(range(5), ncols=100):
        time.sleep(1)
    laf_data = board.get_board_data()

    c3_laf = laf_data[EEG_Ch.C3.value, :]
    c4_laf = laf_data[EEG_Ch.C4.value, :]
    event_laf = np.zeros(c3_laf.size)
    event_laf[- sample_rate * 5] = Codes.laf
    event_laf[-1] = Codes.laf

    print('Stop!')
    time.sleep(1)

    print('Recording rest for 5s, (right arm flexions after)')
    for s in tqdm(range(5), ncols=100):
        time.sleep(1)
    rest1_data = board.get_board_data()

    c3_rest1 = rest1_data[EEG_Ch.C3.value, :]
    c4_rest1 = rest1_data[EEG_Ch.C4.value, :]
    event_rest1 = np.zeros(c3_rest1.size)
    event_rest1[- sample_rate * 5] = Codes.rest
    event_rest1[-1] = Codes.rest

    print('Stop!')
    time.sleep(1)

    print('Recording right arm flexions for 5s (5s rest after)')
    for s in tqdm(range(5), ncols=100):
        time.sleep(1)
    raf_data = board.get_board_data()

    c3_raf = raf_data[EEG_Ch.C3.value, :]
    c4_raf = raf_data[EEG_Ch.C4.value, :]
    event_raf = np.zeros(c3_raf.size)
    event_raf[- sample_rate * 5] = Codes.raf
    event_raf[-1] = Codes.raf

    print('Stop!')
    time.sleep(1)

    print('Recording rest data for 5s (Next trial with left arm flexions after)')
    for s in tqdm(range(5), ncols=100):
        time.sleep(1)
    rest2_data = board.get_board_data()

    c3_rest2 = rest2_data[EEG_Ch.C3.value, :]
    c4_rest2 = rest2_data[EEG_Ch.C4.value, :]
    event_rest2 = np.zeros(c3_rest2.size)
    event_rest2[- sample_rate * 5] = Codes.rest
    event_rest2[-1] = Codes.rest
    
    print('Stop!')
    time.sleep(1)

    c3_data_stream.extend([c3_laf, c3_rest1, c3_raf, c3_rest2])
    c4_data_stream.extend([c4_laf, c4_rest1, c4_raf, c4_rest2])
    event_data_stream.extend([event_laf, event_rest1, event_raf, event_rest2])


print('Writing out data')
with (output_folder / 'c3.csv').open('w') as fp:
    for snippet in c3_data_stream:
        fp.write('\n'.join(str(sample) for sample in snippet))
        fp.write('\n')
with (output_folder / 'c4.csv').open('w') as fp:
    for snippet in c4_data_stream:
        fp.write('\n'.join(str(sample) for sample in snippet))
        fp.write('\n')
with (output_folder / 'event.csv').open('w') as fp:
    for snippet in event_data_stream:
        fp.write('\n'.join(str(sample) for sample in snippet))
        fp.write('\n')

print('Thank you for completing the data collection!')