import ctypes
import matplotlib.pyplot as plt
import mne
import multiprocessing as mp
import numpy as np
import pyfftw
import sys
import time

from datetime import datetime
from env_handler import get_patient_dir
from matplotlib.animation import FuncAnimation

def update_buffer_process(start_time, raw_sig, raw_buffer, sample_rate):
    c3_arr = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)
    c4_arr = np.frombuffer(raw_buffer['C4'].get_obj(), dtype=np.float64)

    while True:
        delta_t = datetime.now() - start_time
        delta_s = int((delta_t.seconds + delta_t.microseconds / 1e6) * sample_rate)

        raw_buffer['C3'].acquire()
        raw_buffer['C4'].acquire()

        if delta_s < c3_arr.shape[0]:
            np.copyto(c3_arr[c3_arr.shape[0] - delta_s : ], raw_sig['C3'][ : delta_s ])
            np.copyto(c4_arr[c4_arr.shape[0] - delta_s : ], raw_sig['C4'][ : delta_s ])
        elif delta_s > raw_sig['C3'].shape[0]:
            c3_arr.fill(0)
            c4_arr.fill(0)
            if delta_s - raw_sig['C3'].shape[0] < c3_arr.shape[0]:
                np.copyto(c3_arr[ : c3_arr.shape[0] - (delta_s - raw_sig['C3'].shape[0])], raw_sig['C3'][- (c3_arr.shape[0] - (delta_s - raw_sig['C3'].shape[0])) : ])
                np.copyto(c4_arr[ : c4_arr.shape[0] - (delta_s - raw_sig['C4'].shape[0])], raw_sig['C4'][- (c4_arr.shape[0] - (delta_s - raw_sig['C4'].shape[0])) : ])
        else:
            np.copyto(c3_arr[:], raw_sig['C3'][delta_s - c3_arr.shape[0] : delta_s])
            np.copyto(c4_arr[:], raw_sig['C4'][delta_s - c4_arr.shape[0] : delta_s])

        raw_buffer['C3'].release()
        raw_buffer['C4'].release()

        time.sleep(0.001)

def update_spec_process(raw_buffer, spec_buffer, spec_dim, pyfftw_buffer, nfft, spec_min, spec_max):
    c3_spec = np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64)

    while True:
        spec_buffer['C3'].acquire()
        pyfftw_buffer[:] = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)[-nfft:]
        np.copyto(c3_spec[:], np.log10(np.abs(pyfftw.interfaces.numpy_fft.rfft(pyfftw_buffer, n=nfft, norm=None))))
        np.copyto(c3_spec[:], np.where(c3_spec[:] < spec_min, spec_min, c3_spec[:]))
        np.copyto(c3_spec[:], np.where(c3_spec[:] > spec_max, spec_max, c3_spec[:]))
        spec_buffer['C3'].release()
        time.sleep(0.001)

if __name__ == '__main__':
    DROP_CHANNELS = ['DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR', 'A1', 'A2', 'Fp1', 'Fp2', 'Fpz']
    NOTCH_FREQS = [60, 120, 180]
    BANDPASS_FREQ = [0.5, 50]

    patient_dir = get_patient_dir()
    subject_name = 'pro00087153_0041' # right arm sham
    subject_folder = patient_dir / subject_name
    if not subject_folder.exists():
        raise Exception(f"Folder {subject_folder} does not exist. Please check your configuration.")

    edf_file_path = subject_folder / f'{subject_name}.edf'
    fif_file_path = subject_folder / f'{subject_name}_trim_raw.fif'

    mne.set_log_level(verbose='WARNING')

    edf = mne.io.read_raw_edf(edf_file_path)
    fif = mne.io.read_raw_fif(fif_file_path)
    num_ch = fif.info['nchan']
    sample_rate = int(edf.info['sfreq'])
    sig_width = 10 * sample_rate
    sig_base = 3100000
    c3_base = 0.0002
    c4_base = -0.0002

    raw_sig = {
        'C3': fif.get_data('C3')[0][sig_base:sig_base + sig_width],
        'C4': fif.get_data('C4')[0][sig_base:sig_base + sig_width]
    }

    raw_data_fps = 60
    raw_data_t_width = 4
    raw_data_s_width = raw_data_t_width * sample_rate
    raw_data_update_rate = int(1000 / raw_data_fps)

    SPEC_MIN = -7
    SPEC_MAX = 0
    c3_spec_fps = 10
    c3_spec_update_rate = 1000 // c3_spec_fps
    c3_spec_t_width = 2
    c3_spec_s_width = c3_spec_t_width * c3_spec_fps
    hz_freq = sample_rate // 2 + 1
    c3_spec_dim = (hz_freq, c3_spec_s_width)
    nfft = sample_rate

    raw_buffer = {
        'C3': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock()),
        'C4': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock())
    }

    spec_buffer = {
        'C3': mp.Array(ctypes.c_double, hz_freq, lock=mp.Lock())
    }
    np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64).fill(SPEC_MIN)
    c3_spec = np.zeros([hz_freq, c3_spec_s_width]) + SPEC_MIN
    pyfftw_buffer = pyfftw.empty_aligned(nfft, dtype="float64")

    raw_data_t_label = np.linspace(-raw_data_t_width, 0, raw_data_s_width)
    c3_spec_t_label = np.linspace(-c3_spec_t_width, 0, c3_spec_s_width, endpoint=False)
    f_label = np.fft.rfftfreq(nfft, d=1/sample_rate)

    """
    sample_rate = 1024 # Hz
    t_len = 10 # seconds
    t = np.linspace(0, t_len, num=t_len*sample_rate)
    sin10 = np.sin(10 * 2 * np.pi * t)
    sin40 = np.sin(40 * 2 * np.pi * t)
    sin5 = np.sin(5 * 2 * np.pi * t)
    sig = sin5 + sin10 + sin40 # sig = sin(5t) + sin(10t) + sin(40t)
    sig_width = sample_rate * t_len
    raw_t_width = 2
    raw_s_width = 2 * sample_rate
    t_label = np.linspace(-raw_t_width, 0, num=raw_s_width)
    """

    fig = plt.figure()
    fig.set_tight_layout(True)

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
        c3_np = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)
        c4_np = np.frombuffer(raw_buffer['C4'].get_obj(), dtype=np.float64)

        c3_raw_plot.set_data([raw_data_t_label, c3_np + c3_base])
        c4_raw_plot.set_data([raw_data_t_label, c4_np + c4_base])

        return c3_raw_plot, c4_raw_plot

    c3_spec_ax = fig.add_subplot(1, 2, 2)
    c3_spec_ax.set_xticks(np.arange(0, c3_spec_t_width * c3_spec_fps, 5))
    c3_spec_ax.set_xticklabels(c3_spec_t_label[::5])
    c3_spec_ax.set_yticks(np.arange(0, hz_freq, 4))
    c3_spec_ax.set_yticklabels(f_label[::4])
    c3_spec_ax.set_ylim(0, 50)
    c3_spec_ax.set_xlabel('Time [sec]')
    c3_spec_ax.set_ylabel('Frequency [Hz]')
    c3_spec_ax.set_title('Spectrogram of C3 starting at an Arbitrary Point')

    c3_spec_plot = c3_spec_ax.imshow(c3_spec, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest', vmin=SPEC_MIN, vmax=SPEC_MAX)
    c3_spec_cb = plt.colorbar(c3_spec_plot, ax=c3_spec_ax)

    def animate_c3_spec(i):
        np.copyto(c3_spec[:, :-1], c3_spec[:, 1:])
        np.copyto(c3_spec[:, -1], np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64))
        c3_spec_plot.set_data(c3_spec)
        return c3_spec_plot,

    raw_ani = FuncAnimation(fig, animate_raw_data, interval=raw_data_update_rate, blit=True,
                            save_count=int(np.ceil(1000 * (sig_width + raw_data_s_width) / raw_data_update_rate / sample_rate)))
    c3_spec_ani = FuncAnimation(fig, animate_c3_spec, interval=c3_spec_update_rate, blit=True,
                            save_count=int(np.ceil(1000 * (sig_width * raw_data_s_width) / c3_spec_update_rate / sample_rate)))

    buffer_update = mp.Process(target=update_buffer_process, args=(datetime.now(), raw_sig, raw_buffer, sample_rate))
    spec_update = mp.Process(target=update_spec_process, args=(raw_buffer, spec_buffer, c3_spec_dim, pyfftw_buffer, nfft, SPEC_MIN, SPEC_MAX))

    buffer_update.start()
    spec_update.start()
    plt.show()
    buffer_update.kill()
    spec_update.kill()