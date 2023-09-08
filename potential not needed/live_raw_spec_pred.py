import ctypes
import datetime
import joblib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pyfftw
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from matplotlib.animation import FuncAnimation
from eeg_utils import EEG_Ch, eeg_bands
from sklearn import ensemble
from scipy import signal

def update_buffer_process(raw_buffer, filt_buffer):
    print(f"[DEBUG] Update buffer process launched: {os.getpid()}")

    params = BrainFlowInputParams()
    params.serial_port = 'COM3'
    BoardShim.enable_dev_board_logger()
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
    sample_rate = board.get_sampling_rate(board.board_id)
    board.prepare_session()
    board.start_stream()

    c3_arr = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)
    c4_arr = np.frombuffer(raw_buffer['C4'].get_obj(), dtype=np.float64)

    c3_filt = np.frombuffer(filt_buffer['C3'].get_obj(), dtype=np.float64)
    c4_filt = np.frombuffer(filt_buffer['C4'].get_obj(), dtype=np.float64)

    f1 = 5
    f2 = 50
    fc = (f2 - f1) / 2
    fw = fc - f1

    while True:
        board_buf = board.get_board_data()
        if board_buf.shape[1]:
            raw_buffer['C3'].acquire()
            raw_buffer['C4'].acquire()

            c3_arr[:] = np.hstack([c3_arr, board_buf[EEG_Ch.C3.value, :]])[-c3_arr.shape[0]:]
            c4_arr[:] = np.hstack([c4_arr, board_buf[EEG_Ch.C4.value, :]])[-c4_arr.shape[0]:]

            raw_buffer['C3'].release()
            raw_buffer['C4'].release()

            filt_buffer['C3'].acquire()
            filt_buffer['C4'].acquire()

            np.copyto(c3_filt, c3_arr)
            # Check written notes (in notebook) for the two filters used on each of these channels
            DataFilter.perform_bandstop(c3_filt, sample_rate, 60, 4, 2, FilterTypes.BUTTERWORTH, 0)
            DataFilter.perform_bandpass(c3_filt, sample_rate, fc, fw, 2, FilterTypes.BUTTERWORTH, 0)

            np.copyto(c4_filt, c4_arr)
            DataFilter.perform_bandstop(c4_filt, sample_rate, 60, 4, 2, FilterTypes.BUTTERWORTH, 0)
            DataFilter.perform_bandpass(c4_filt, sample_rate, fc, fw, 2, FilterTypes.BUTTERWORTH, 0)

            filt_buffer['C3'].release()
            filt_buffer['C4'].release()

        time.sleep(0.001)

def update_c3_spec_process(raw_buffer, spec_buffer, pyfftw_buffer, nfft, spec_min, spec_max):
    print(f"[DEBUG] Update C3 spec process launched: {os.getpid()}")

    c3_spec = np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64)

    while True:
        spec_buffer['C3'].acquire()
        pyfftw_buffer[:] = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)[-nfft:]
        np.copyto(c3_spec[:], np.log10(np.abs(pyfftw.interfaces.numpy_fft.rfft(pyfftw_buffer, n=nfft, norm=None))))
        np.copyto(c3_spec[:], np.where(c3_spec[:] < spec_min, spec_min, c3_spec[:]))
        np.copyto(c3_spec[:], np.where(c3_spec[:] > spec_max, spec_max, c3_spec[:]))
        spec_buffer['C3'].release()

        time.sleep(0.001)

def update_c4_spec_process(raw_buffer, spec_buffer, pyfftw_buffer, nfft, spec_min, spec_max):
    print(f"[DEBUG] Update C4 spec process launched: {os.getpid()}")

    c4_spec = np.frombuffer(spec_buffer['C4'].get_obj(), dtype=np.float64)

    while True:
        spec_buffer['C4'].acquire()
        pyfftw_buffer[:] = np.frombuffer(raw_buffer['C4'].get_obj(), dtype=np.float64)[-nfft:]
        np.copyto(c4_spec[:], np.log10(np.abs(pyfftw.interfaces.numpy_fft.rfft(pyfftw_buffer, n=nfft, norm=None))))
        np.copyto(c4_spec[:], np.where(c4_spec[:] < spec_min, spec_min, c4_spec[:]))
        np.copyto(c4_spec[:], np.where(c4_spec[:] > spec_max, spec_max, c4_spec[:]))
        spec_buffer['C4'].release()

        time.sleep(0.001)
    
def update_prediction(raw_buffer, sample_rate, predict_buffer):
    print(f"[DEBUG] Prediction process launched: {os.getpid()}")

    model = joblib.load('output/maurice6/rf.out')

    c3_arr = np.frombuffer(raw_buffer['C3'].get_obj(), dtype=np.float64)
    c4_arr = np.frombuffer(raw_buffer['C4'].get_obj(), dtype=np.float64)
    pred_arr = np.frombuffer(predict_buffer.get_obj(), dtype=np.float64)

    while True:
        raw_buffer['C3'].acquire()
        raw_buffer['C4'].acquire()

        c3_avg = np.mean(c3_arr[-4 * sample_rate : ])
        c3_std = np.std(c3_arr[-4 * sample_rate : ])
        c4_avg = np.mean(c4_arr[-4 * sample_rate : ])
        c4_std = np.std(c4_arr[-4 * sample_rate : ])

        c3_use = c3_arr[-sample_rate :]
        c4_use = c4_arr[-sample_rate :]

        raw_buffer['C3'].release()
        raw_buffer['C4'].release()

        if c3_std == 0:
            c3_std = 0.01
        if c4_std == 0:
            c4_std = 0.01

        c3_use = (c3_use - c3_avg) / c3_std
        c4_use = (c4_use - c4_avg) / c4_std

        c3_welch = signal.welch(c3_use, fs=sample_rate, nperseg=sample_rate)[1]
        c4_welch = signal.welch(c4_use, fs=sample_rate, nperseg=sample_rate)[1]
        feature = [
            np.sum(c3_welch[eeg_bands.DELTA[0] : eeg_bands.DELTA[1] + 1]),
            np.sum(c3_welch[eeg_bands.THETA[0] : eeg_bands.THETA[1] + 1]),
            np.sum(c3_welch[eeg_bands.ALPHA[0] : eeg_bands.ALPHA[1] + 1]),
            np.sum(c3_welch[eeg_bands.BETA[0] : eeg_bands.BETA[1] + 1]),
            np.sum(c3_welch[eeg_bands.GAMMA[0] : eeg_bands.GAMMA[1] + 1]),
            np.sum(c4_welch[eeg_bands.DELTA[0] : eeg_bands.DELTA[1] + 1]),
            np.sum(c4_welch[eeg_bands.THETA[0] : eeg_bands.THETA[1] + 1]),
            np.sum(c4_welch[eeg_bands.ALPHA[0] : eeg_bands.ALPHA[1] + 1]),
            np.sum(c4_welch[eeg_bands.BETA[0] : eeg_bands.BETA[1] + 1]),
            np.sum(c4_welch[eeg_bands.GAMMA[0] : eeg_bands.GAMMA[1] + 1])
        ]

        predict_buffer.acquire()
        pred_arr[:] = model.predict_proba([feature])[0]
        predict_buffer.release()

        #print(f'{feature}, {datetime.datetime.now()}')

        time.sleep(0.1)

if __name__ == '__main__':
    print(f"[DEBUG] Main process launched: {os.getpid()}")

    num_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
    print(num_ch)
    sample_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
    c3_base = 40
    c4_base = -40

    raw_data_fps = 60
    raw_data_t_width = 16
    raw_data_s_width = raw_data_t_width * sample_rate
    display_t_width = 4
    display_s_width = display_t_width * sample_rate
    raw_data_update_rate = int(1000 / raw_data_fps)

    SPEC_MIN = -7
    SPEC_MAX = 10
    spec_fps = 10
    spec_update_rate = 1000 // spec_fps
    spec_t_width = 2
    spec_s_width = spec_t_width * spec_fps
    hz_freq = sample_rate // 2 + 1
    spec_dim = (hz_freq, spec_s_width)
    nfft = sample_rate

    raw_buffer = {
        'C3': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock()),
        'C4': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock())
    }

    filt_buffer = {
        'C3': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock()),
        'C4': mp.Array(ctypes.c_double, raw_data_s_width, lock=mp.Lock())
    }

    spec_buffer = {
        'C3': mp.Array(ctypes.c_double, hz_freq, lock=mp.Lock()),
        'C4': mp.Array(ctypes.c_double, hz_freq, lock=mp.Lock())
    }

    pred_buffer = mp.Array(ctypes.c_double, 2, lock=mp.Lock())

    np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64).fill(SPEC_MIN)
    np.frombuffer(spec_buffer['C4'].get_obj(), dtype=np.float64).fill(SPEC_MIN)
    c3_spec = np.zeros([hz_freq, spec_s_width]) + SPEC_MIN
    c4_spec = np.zeros([hz_freq, spec_s_width]) + SPEC_MIN
    c3_pyfftw_buffer = pyfftw.empty_aligned(nfft, dtype="float64")
    c4_pyfftw_buffer = pyfftw.empty_aligned(nfft, dtype="float64")

    raw_data_t_label = np.linspace(-display_t_width, 0, display_s_width)
    c3_spec_t_label = np.linspace(-spec_t_width, 0, spec_s_width, endpoint=False)
    c4_spec_t_label = np.linspace(-spec_t_width, 0, spec_s_width, endpoint=False)
    f_label = np.fft.rfftfreq(nfft, d=1/sample_rate)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    raw_data_ax = fig.add_subplot(gs[:, 0])
    raw_data_ax.set_xlim(-display_t_width, 0)
    raw_data_ax.set_ylim(-80, 80)
    raw_data_ax.set_xlabel('Time [sec]')
    raw_data_ax.set_ylabel('Amplitude')
    raw_data_ax.set_title('Raw Electrode Data')

    c3_raw_plot, = raw_data_ax.plot([], [], label='C3')
    c4_raw_plot, = raw_data_ax.plot([], [], label='C4')

    raw_data_ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

    def animate_raw_data(i):
        c3_np = np.frombuffer(filt_buffer['C3'].get_obj(), dtype=np.float64)
        c4_np = np.frombuffer(filt_buffer['C4'].get_obj(), dtype=np.float64)

        filt_buffer['C3'].acquire()
        c3_raw_plot.set_data([raw_data_t_label, c3_np[- display_s_width : ] + c3_base])
        filt_buffer['C3'].release()

        filt_buffer['C4'].acquire()
        c4_raw_plot.set_data([raw_data_t_label, c4_np[- display_s_width : ] + c4_base])
        filt_buffer['C4'].release()

        return c3_raw_plot, c4_raw_plot

    c3_spec_ax = fig.add_subplot(gs[0, 1])
    c3_spec_ax.set_xticks(np.arange(0, spec_t_width * spec_fps, 5))
    c3_spec_ax.set_xticklabels(c3_spec_t_label[::5])
    c3_spec_ax.set_yticks(np.arange(0, hz_freq, 4))
    c3_spec_ax.set_yticklabels(f_label[::4])
    c3_spec_ax.set_ylim(0, 50)
    c3_spec_ax.set_xlabel('Time [sec]')
    c3_spec_ax.set_ylabel('Frequency [Hz]')
    c3_spec_ax.set_title('Spectrogram of C3')

    c3_spec_plot = c3_spec_ax.imshow(c3_spec, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest', vmin=SPEC_MIN, vmax=SPEC_MAX)
    c3_spec_cb = plt.colorbar(c3_spec_plot, ax=c3_spec_ax)

    def animate_c3_spec(i):
        np.copyto(c3_spec[:, :-1], c3_spec[:, 1:])
        spec_buffer['C3'].acquire()
        np.copyto(c3_spec[:, -1], np.frombuffer(spec_buffer['C3'].get_obj(), dtype=np.float64))
        spec_buffer['C3'].release()
        c3_spec_plot.set_data(c3_spec)
        return c3_spec_plot,

    c4_spec_ax = fig.add_subplot(gs[1, 1])
    c4_spec_ax.set_xticks(np.arange(0, spec_t_width * spec_fps, 5))
    c4_spec_ax.set_xticklabels(c4_spec_t_label[::5])
    c4_spec_ax.set_yticks(np.arange(0, hz_freq, 4))
    c4_spec_ax.set_yticklabels(f_label[::4])
    c4_spec_ax.set_ylim(0, 50)
    c4_spec_ax.set_xlabel('Time [sec]')
    c4_spec_ax.set_ylabel('Frequency [Hz]')
    c4_spec_ax.set_title('Spectrogram of C4')

    c4_spec_plot = c4_spec_ax.imshow(c4_spec, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest', vmin=SPEC_MIN, vmax=SPEC_MAX)
    c4_spec_cb = plt.colorbar(c4_spec_plot, ax=c4_spec_ax)

    def animate_c4_spec(i):
        np.copyto(c4_spec[:, :-1], c4_spec[:, 1:])
        spec_buffer['C4'].acquire()
        np.copyto(c4_spec[:, -1], np.frombuffer(spec_buffer['C4'].get_obj(), dtype=np.float64))
        spec_buffer['C4'].release()
        c4_spec_plot.set_data(c4_spec)
        return c4_spec_plot,
    
    pred_data_ax = fig.add_subplot(gs[0, 2])
    pred_data_ax.set_xlim(-1, 2)
    pred_data_ax.set_ylim(0, 1)
    pred_data_ax.set_xticks([0, 1])
    pred_data_ax.set_xticklabels(['Rest', 'Move'])
    pred_data_ax.set_xlabel('Action')
    pred_data_ax.set_ylabel('Probability')
    pred_data_ax.set_title('Predicted State of Action')

    rest_plot, = pred_data_ax.plot([], [], linewidth=5)
    move_plot, = pred_data_ax.plot([], [], linewidth=5)
    def animate_prediction(i):
        pred_np = np.frombuffer(pred_buffer.get_obj(), dtype=np.float64)

        pred_buffer.acquire()
        rest_plot.set_data([0, 0], [0, pred_np[0]])
        move_plot.set_data([1, 1], [0, pred_np[1]])
        pred_buffer.release()

        return rest_plot, move_plot

    raw_ani = FuncAnimation(fig, animate_raw_data, interval=raw_data_update_rate, blit=True,)
    c3_spec_ani = FuncAnimation(fig, animate_c3_spec, interval=spec_update_rate, blit=True,)
    c4_spec_ani = FuncAnimation(fig, animate_c4_spec, interval=spec_update_rate, blit=True,)
    pred_ani = FuncAnimation(fig, animate_prediction, interval=spec_update_rate, blit=True,)

    buffer_update = mp.Process(target=update_buffer_process, args=(raw_buffer, filt_buffer))
    c3_spec_update = mp.Process(target=update_c3_spec_process, args=(filt_buffer, spec_buffer, c3_pyfftw_buffer, nfft, SPEC_MIN, SPEC_MAX))
    c4_spec_update = mp.Process(target=update_c4_spec_process, args=(filt_buffer, spec_buffer, c4_pyfftw_buffer, nfft, SPEC_MIN, SPEC_MAX))
    pred_update = mp.Process(target=update_prediction, args=(raw_buffer, sample_rate, pred_buffer))

    buffer_update.start()
    c3_spec_update.start()
    c4_spec_update.start()
    pred_update.start()
    plt.show()
    buffer_update.kill()
    c3_spec_update.kill()
    c4_spec_update.kill()
    pred_update.kill()