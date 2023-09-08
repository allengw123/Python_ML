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

DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2', 'Fp1', 'Fp2', 'Fpz']
NOTCH_FREQS = [60, 120, 180]
BAND_PASS = [0.5, 50]

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
    print(f"\tApplying notch filters at following frequencies (Hz): " + ", ".join([str(x) for x in NOTCH_FREQS]))
    output_trim_eeg(edf_file_path, DROP_CHANNELS, NOTCH_FREQS, BAND_PASS)
else:
    print("Using found trimmed EEG file")

trim_fif = mne.io.read_raw_fif(fif_file_path)
num_ch = trim_fif.info['nchan']
sample_rate = int(trim_fif.info['sfreq'])
sig = trim_fif.get_data('C3')[0][3100000:3110240]

hz_freq = sample_rate // 2 + 1
nfft = sample_rate

raw_data_fps = 10
raw_data_t_width = 2
raw_data_s_width = raw_data_t_width * sample_rate
update_rate = 1000 // raw_data_fps
raw_data_buffer = np.zeros(raw_data_s_width)
spec_buffer = np.zeros([hz_freq, raw_data_t_width * raw_data_fps])
t_label = np.linspace(-raw_data_t_width, 0, num=raw_data_t_width * raw_data_fps, endpoint=False)
f_label = np.fft.rfftfreq(nfft, d=1/sample_rate)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xticks(np.arange(0, raw_data_t_width * raw_data_fps, 5))
print(ax1.get_xticks())
print(t_label)
ax1.set_xticklabels(t_label[::5])
ax1.set_yticks(np.arange(0, hz_freq, 4))
ax1.set_yticklabels(f_label[::4])
ax1.set_ylim(0, 50)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_title('Spectrogram of C3 Data Starting at an Extraordinarily Arbitrary Point')
#ax1.set_xticks(t_label)
ax1_art = ax1.imshow(spec_buffer, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest', vmin=0, vmax=5)
ax1_cb = plt.colorbar(ax1_art, ax=ax1)

def animate(i):
    global raw_data_buffer, sig, update_rate, raw_data_t_width, sample_rate, raw_data_s_width, spec_buffer
    data_shown_s = int(i * update_rate / 1000 * sample_rate)
    if data_shown_s == 0:
        pass
    elif data_shown_s < raw_data_s_width:
        np.copyto(raw_data_buffer[raw_data_s_width - data_shown_s : ], sig[ : data_shown_s])
    elif data_shown_s - len(sig) >= raw_data_s_width:
        raw_data_buffer.fill(0)
    elif data_shown_s > len(sig):
        raw_data_buffer.fill(0)
        np.copyto(raw_data_buffer[0:(raw_data_s_width - (data_shown_s - len(sig)))], sig[(data_shown_s - len(sig)) - raw_data_s_width : ])
    else:
        np.copyto(raw_data_buffer, sig[data_shown_s - raw_data_s_width : data_shown_s])

    np.copyto(spec_buffer[:, :-1], spec_buffer[:, 1:])
    spec_buffer[:, -1] = np.log10(np.abs(np.fft.rfft(raw_data_buffer[-nfft:], n=nfft, norm=None))) + 5
    spec_buffer[:, -1] = np.where(spec_buffer[:, -1] < 0, 0, spec_buffer[:, -1])
    #print(spec_buffer[:, -1])
    ax1_art.set_data(spec_buffer)
    return ax1_art,

ani = FuncAnimation(fig, animate, interval=update_rate, blit=True, save_count=int(np.ceil(1000 * (len(sig) + raw_data_s_width) / update_rate / sample_rate)))
plt.show()
if False:
    raw_data_buffer.fill(0)
    spec_buffer.fill(0)
    ani.save('test.mp4', fps=raw_data_fps, codec='libx264')