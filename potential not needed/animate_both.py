import matplotlib.pyplot as plt
import mne
import numpy as np
import pyfftw
import scipy.signal
import time

from matplotlib.animation import FuncAnimation
from env_handler import get_patient_dir
from datetime import datetime

DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2', 'Fp1', 'Fp2', 'Fpz']

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

edf = mne.io.read_raw_edf(edf_file_path)
fif = mne.io.read_raw_fif(fif_file_path)
num_ch = edf.info['nchan']
sample_rate = int(edf.info['sfreq'])
sig_width = 10 * sample_rate
sig_base = 3100000
c3_base = 0.0002
c4_base = -0.0002

raw_sig = {
    'C3': fif.get_data('C3')[0][sig_base:sig_base + sig_width],
    'C4': fif.get_data('C4')[0][sig_base:sig_base + sig_width],
}

raw_data_fps = 60
raw_data_t_width = 4
raw_data_s_width = raw_data_t_width * sample_rate
raw_data_update_rate = 1000 // raw_data_fps # milliseconds

SPEC_MIN = -7
SPEC_MAX = 0
c3_spec_fps = 10
c3_spec_update_rate = 1000 // c3_spec_fps
c3_spec_t_width = 2
hz_freq = sample_rate // 2 + 1
nfft = sample_rate

start = None

raw_buffer = {
    'C3': np.zeros(raw_data_s_width),
    'C4': np.zeros(raw_data_s_width),
}
spec_buffer = {
    'C3': np.zeros([hz_freq, c3_spec_t_width * c3_spec_fps]) + SPEC_MIN
}
pyfftw_buffer = pyfftw.empty_aligned(nfft, dtype='float64')

raw_data_t_label = np.linspace(-raw_data_t_width, 0, num=raw_data_s_width)
c3_spec_t_label = np.linspace(-c3_spec_t_width, 0, num=c3_spec_t_width * c3_spec_fps, endpoint=False)
f_label = np.fft.rfftfreq(nfft, d=1/sample_rate)

fig = plt.figure()
fig.set_tight_layout(True)

# Set up the raw data plot
raw_data_ax = fig.add_subplot(1, 2, 1)
raw_data_ax.set_xlim(-raw_data_t_width, 0)
raw_data_ax.set_ylim(-0.0005, 0.0005)
raw_data_ax.set_xlabel('Time [sec]')
raw_data_ax.set_ylabel('Amplitude')
raw_data_ax.set_title('Raw Electrode Data Starting at an Arbitrary Point')

c3_raw_plot, = raw_data_ax.plot([], [], label='C3')
c4_raw_plot, = raw_data_ax.plot([], [], label='C4')

raw_data_ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

def animate_raw_data(i):
    delta_t = datetime.now() - start
    delta_s = int((delta_t.seconds + delta_t.microseconds / 1e6) * sample_rate)

    if delta_s < raw_data_s_width:
        np.copyto(raw_buffer['C3'][raw_data_s_width - delta_s : ], raw_sig['C3'][ : delta_s])
        np.copyto(raw_buffer['C4'][raw_data_s_width - delta_s : ], raw_sig['C4'][ : delta_s])
    elif delta_s > sig_width:
        raw_buffer['C3'][ : ].fill(0)
        raw_buffer['C4'][ : ].fill(0)
        if delta_s - sig_width < raw_data_s_width:
            np.copyto(raw_buffer['C3'][ : raw_data_s_width - (delta_s - sig_width)], raw_sig['C3'][- (raw_data_s_width - (delta_s - sig_width)) : ])
            np.copyto(raw_buffer['C4'][ : raw_data_s_width - (delta_s - sig_width)], raw_sig['C4'][- (raw_data_s_width - (delta_s - sig_width)) : ])
    else:
        np.copyto(raw_buffer['C3'][:], raw_sig['C3'][delta_s - raw_data_s_width : delta_s])
        np.copyto(raw_buffer['C4'][:], raw_sig['C4'][delta_s - raw_data_s_width : delta_s])

    """
    if data_shown_s == 0:
        pass
    elif data_shown_s < raw_data_s_width:
        np.copyto(raw_buffer['C3'][raw_data_s_width - data_shown_s : ], raw_sig['C3'][ : data_shown_s])
        np.copyto(raw_buffer['C4'][raw_data_s_width - data_shown_s : ], raw_sig['C4'][ : data_shown_s])
    elif data_shown_s - sig_width >= raw_data_s_width:
        raw_buffer['C3'].fill(0)
        raw_buffer['C4'].fill(0)
    elif data_shown_s > sig_width:
        raw_buffer['C3'].fill(0)
        raw_buffer['C4'].fill(0)
        np.copyto(raw_buffer['C3'][0:(raw_data_s_width - (data_shown_s - sig_width))], raw_sig['C3'][(data_shown_s - sig_width) - raw_data_s_width : ])
        np.copyto(raw_buffer['C4'][0:(raw_data_s_width - (data_shown_s - sig_width))], raw_sig['C4'][(data_shown_s - sig_width) - raw_data_s_width : ])
    else:
        np.copyto(raw_buffer['C3'], raw_sig['C3'][data_shown_s - raw_data_s_width : data_shown_s])
        np.copyto(raw_buffer['C4'], raw_sig['C4'][data_shown_s - raw_data_s_width : data_shown_s])
    """

    c3_raw_plot.set_data([raw_data_t_label, raw_buffer['C3'] + c3_base])
    c4_raw_plot.set_data([raw_data_t_label, raw_buffer['C4'] + c4_base])

    return c3_raw_plot, c4_raw_plot

# Set up the C3 spectrogram plot
c3_spec_ax = fig.add_subplot(1, 2, 2)
c3_spec_ax.set_xticks(np.arange(0, c3_spec_t_width * c3_spec_fps, 5))
c3_spec_ax.set_xticklabels(c3_spec_t_label[::5])
c3_spec_ax.set_yticks(np.arange(0, hz_freq, 4))
c3_spec_ax.set_yticklabels(f_label[::4])
c3_spec_ax.set_ylim(0, 50)
c3_spec_ax.set_xlabel('Time [sec]')
c3_spec_ax.set_ylabel('Frequency [Hz]')
c3_spec_ax.set_title('Spectrogram of C3 Starting at an Arbitrary Point')

c3_spec_plot = c3_spec_ax.imshow(spec_buffer['C3'], aspect='auto', origin='lower', cmap='inferno', interpolation='nearest', vmin=SPEC_MIN, vmax=SPEC_MAX)
c3_spec_cb = plt.colorbar(c3_spec_plot, ax=c3_spec_ax)

def animate_c3_spec(i):
    np.copyto(spec_buffer['C3'][:, :-1], spec_buffer['C3'][:, 1:])
    pyfftw_buffer[:] = raw_buffer['C3'][-nfft:]
    np.copyto(spec_buffer['C3'][:, -1], np.log10(np.abs(pyfftw.interfaces.numpy_fft.rfft(pyfftw_buffer, n=nfft, norm=None))))
    np.copyto(spec_buffer['C3'][:, -1], np.where(spec_buffer['C3'][:, -1] < SPEC_MIN, SPEC_MIN, spec_buffer['C3'][:, -1]))
    np.copyto(spec_buffer['C3'][:, -1], np.where(spec_buffer['C3'][:, -1] > SPEC_MAX, SPEC_MAX, spec_buffer['C3'][:, -1]))
    c3_spec_plot.set_data(spec_buffer['C3'])
    return c3_spec_plot,

# Set up the animation
start = datetime.now()
raw_ani = FuncAnimation(fig, animate_raw_data, interval=raw_data_update_rate, blit=True,
                        save_count=int(np.ceil(1000 * (sig_width + raw_data_s_width) / raw_data_update_rate / sample_rate)))
c3_spec_ani = FuncAnimation(fig, animate_c3_spec, interval=c3_spec_update_rate, blit=True,
                        save_count=int(np.ceil(1000 * (sig_width * raw_data_s_width) / c3_spec_update_rate / sample_rate)))

plt.show()