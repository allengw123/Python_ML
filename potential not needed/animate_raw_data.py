import matplotlib.pyplot as plt
import mne
import numpy as np

from matplotlib.animation import FuncAnimation
from env_handler import get_patient_dir
from eeg_utils import output_trim_eeg

"""
sample_rate = 1024 # Hz
t_len = 10 # seconds
t = np.linspace(0, t_len, num=t_len*sample_rate)
sin10 = np.sin(10 * 2 * np.pi * t)
sin40 = np.sin(40 * 2 * np.pi * t)
sin5 = np.sin(5 * 2 * np.pi * t)
sig = sin5 + sin10 + sin40 # sig = sin(5t) + sin(10t) + sin(40t)
"""

DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2', 'Fp1', 'Fp2']
NOTCH_FREQS = [60, 120, 180]

patient_dir = get_patient_dir()
if not patient_dir.exists():
    raise Exception(f"Folder {patient_dir} does not exist. Please check your configuration settings.")

subject_name = 'pro00087153_0041' # right arm sham
subject_folder = patient_dir / subject_name
if not subject_folder.exists():
    raise Exception(f"Folder {subject_folder} does not exist. Please check your selected subject.")

edf_file_path = subject_folder / f'{subject_name}.edf'
fif_file_path = subject_folder / f'{subject_name}_trim_raw.fif'

mne.set_log_level(verbose='WARNING')

if not fif_file_path.is_file():
    print("Trimming base EEG file")
    print(f"\tDropping the following channels: X*, " + ", ".join(DROP_CHANNELS))
    print(f"\tApplying notch filters at following frequencies (Hz): " + ", ".join(NOTCH_FREQS))
    output_trim_eeg(edf_file_path, DROP_CHANNELS, NOTCH_FREQS)
else:
    print("Using found trimmed EEG file")

trim_fif = mne.io.read_raw_fif(fif_file_path)
num_ch = trim_fif.info['nchan']
sample_rate = int(trim_fif.info['sfreq'])
sig_width = 10 * sample_rate
sig_base = 3100000
c3_base = 0.0002
c4_base = -0.0002
c3_sig = trim_fif.get_data('C3')[0][sig_base:sig_base + sig_width] + c3_base # offset for showing both at the same time
c4_sig = trim_fif.get_data('C4')[0][sig_base:sig_base + sig_width] + c4_base # " "

raw_data_fps = 120
raw_data_t_width = 2
raw_data_s_width = raw_data_t_width * sample_rate
update_rate = 1000 // raw_data_fps # milliseconds
c3_buffer = np.zeros(raw_data_s_width) + c3_base
c4_buffer = np.zeros(raw_data_s_width) + c4_base
t_label = np.linspace(-raw_data_t_width, 0, num=raw_data_s_width)

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlim(-raw_data_t_width, 0)
ax1.set_ylim(-.0005, .0005)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Amplitude')
ax1.set_title('Raw Electrode Data Starting at an Arbitrary Point')

c3_plot, = ax1.plot([], [], label='C3')
c4_plot, = ax1.plot([], [], label='C4')

ax1.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

def animate(i):
    global c3_buffer, c3_sig, c4_buffer, c4_sig, sig_width, update_rate, raw_data_t_width, sample_rate, raw_data_s_width
    data_shown_s = int(i * update_rate / 1000 * sample_rate)

    if data_shown_s == 0:
        pass
    elif data_shown_s < raw_data_s_width:
        np.copyto(c3_buffer[raw_data_s_width - data_shown_s : ], c3_sig[ : data_shown_s])
        np.copyto(c4_buffer[raw_data_s_width - data_shown_s : ], c4_sig[ : data_shown_s])
    elif data_shown_s - sig_width >= raw_data_s_width:
        c3_buffer.fill(c3_base)
        c4_buffer.fill(c4_base)
    elif data_shown_s > sig_width:
        c3_buffer.fill(c3_base)
        c4_buffer.fill(c4_base)
        np.copyto(c3_buffer[0:(raw_data_s_width - (data_shown_s - sig_width))], c3_sig[(data_shown_s - sig_width) - raw_data_s_width : ])
        np.copyto(c4_buffer[0:(raw_data_s_width - (data_shown_s - sig_width))], c4_sig[(data_shown_s - sig_width) - raw_data_s_width : ])
    else:
        np.copyto(c3_buffer, c3_sig[data_shown_s - raw_data_s_width : data_shown_s])
        np.copyto(c4_buffer, c4_sig[data_shown_s - raw_data_s_width : data_shown_s])

    c3_plot.set_data([t_label,c3_buffer])
    c4_plot.set_data([t_label,c4_buffer])

    return c3_plot, c4_plot

ani = FuncAnimation(fig, animate, interval=update_rate, blit=True, save_count=int(np.ceil(1000 * (sig_width + raw_data_s_width) / update_rate / sample_rate)))
plt.show()

if False:
    c3_buffer.fill(c3_base)
    c4_buffer.fill(c4_base)
    ani.save('test.mp4', fps=raw_data_fps, codec='libx264')