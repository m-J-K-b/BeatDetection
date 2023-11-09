import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio


def filterbank(audio, fs, freq_splits=[200, 400, 800, 1600, 3200]):
    freq_splits = np.array(freq_splits)
    freq_splits = np.concatenate((np.array([0]), freq_splits, np.array([np.inf])))
    ns = audio.shape[-1]

    fft = np.fft.fft(audio)

    freq = np.abs(np.fft.fftfreq(ns, 1 / fs))
    freq_bands = np.zeros(shape=(freq_splits.shape[-1] + 1, ns), dtype=np.complex128)
    for i in range(1, freq_splits.shape[-1]):
        greater = freq > freq_splits[i - 1]
        less = freq < freq_splits[i]
        window = greater & less
        freq_bands[i] = np.fft.ifft(fft * window)

    return freq_bands


def smoothing(freq_bands, fs):
    full_wave_rectified_bands = [np.abs(band) for band in freq_bands]
    right_half_hanning_window = np.hanning(int(0.4 * fs))[int(0.2 * fs) :]
    convolved_bands = [
        np.convolve(band, right_half_hanning_window, mode="same")
        for band in full_wave_rectified_bands
    ]
    return convolved_bands


def diff_rect(bands):
    differenciated_bands = [np.gradient(band) for band in bands]
    half_rectified_bands = [[band * [band > 0]] for band in differenciated_bands]
    return half_rectified_bands


def comb_filtering(bands, bpms=[]):
    pass


t = np.linspace(0, 1, 1000)
s = np.sin(t * 2 * np.pi * 2)
sg = np.gradient(s)
sg = sg * [sg > 0]


wf = wave.open("./assets/sounds/sampleSong3.wav", "r")
buffer = wf.readframes(wf.getnframes())
interleaved = np.frombuffer(buffer, dtype=f"int{wf.getsampwidth() * 8}")
audio = np.reshape(interleaved, (wf.getnchannels(), -1))
if audio.shape[0] > 1:
    audio = (audio[0] + audio[1]) * 0.5
fs = wf.getframerate()
audio = audio[: 5 * fs]

split_signal = filterbank(audio, fs)
smoothed_bands = smoothing(split_signal, fs)
diff_rected_bands = diff_rect(split_signal)

s = np.sum(np.abs(diff_rected_bands), axis=0)[0][0]
plt.plot(np.arange(0, audio.shape[-1]), s)
plt.show()
