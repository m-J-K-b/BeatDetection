### Here i tried to implement the algorithm described here:
# https://archive.gamedev.net/archive/reference/programming/features/beatdetection/index.html


import pathlib
import time
import wave
from threading import Thread

import init
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import pygame as pg
from numpy.lib import stride_tricks
from util import time_it


def resample_array(arr, new_sample_num):
    arr_strided = stride_tricks.as_strided(
        arr,
        shape=(arr.shape[-1], new_sample_num // arr.shape[-1]),
        strides=(arr.itemsize, 0),
    )
    b = arr_strided.reshape((arr_strided.shape[0] * arr_strided.shape[1]))
    b_ext = np.concatenate(
        (b, np.full(shape=(new_sample_num % arr.shape[-1]), fill_value=b[-1]))
    )
    return b_ext


@time_it
def get_beat(
    audio,
    Ns,
    Fs,
    local_window_size,
    instant_window_size,
    min_power_factor=1.4,
    freq_split_positions=np.array([]),
):
    instant_window_size = int(instant_window_size)
    local_window_size = int(local_window_size)
    inverse_Fs = 1 / Fs

    freq_split_positions = np.concatenate(
        (np.array([0]), freq_split_positions, np.array([np.inf]))
    )
    freq_split_num = freq_split_positions.shape[-1]

    overhead = instant_window_size - audio.shape[-1] % instant_window_size
    audio = np.concatenate((audio, np.zeros(shape=(overhead))))

    instant_energy_data = np.zeros(
        shape=(freq_split_num, audio.shape[-1] // instant_window_size)
    )

    for i, data in enumerate(np.reshape(audio, newshape=(-1, instant_window_size))):
        fft = np.fft.fft(data)[0 : instant_window_size // 2]
        fft_freqs = np.fft.fftfreq(instant_window_size, inverse_Fs)[
            0 : instant_window_size // 2
        ]

        for j in range(0, freq_split_num - 1):
            start_freq = freq_split_positions[j]
            end_freq = freq_split_positions[j + 1]

            instant_indecies = [
                idx
                for (idx, val) in enumerate(fft_freqs)
                if start_freq < val < end_freq
            ]
            instant_energy_data[j, i] = np.sum(np.abs(fft[instant_indecies]))

    local_normalization_factor = instant_window_size / local_window_size

    window_size = local_window_size // instant_window_size
    half_window_size = window_size // 2
    instant_energy_data_ext = np.concatenate(
        (
            np.zeros(shape=(instant_energy_data.shape[0], half_window_size)),
            instant_energy_data,
            np.zeros(shape=(instant_energy_data.shape[0], half_window_size)),
        ),
        axis=1,
    )
    instant_energy_data_strided = stride_tricks.as_strided(
        instant_energy_data_ext,
        shape=(
            instant_energy_data.shape[-1],
            instant_energy_data.shape[0],
            window_size,
        ),
        strides=(
            instant_energy_data_ext.itemsize,
            instant_energy_data_ext.itemsize * (instant_energy_data_ext.shape[-1]),
            instant_energy_data_ext.itemsize,
        ),
    )

    beat = np.zeros_like(instant_energy_data)

    for i, local_data in enumerate(instant_energy_data_strided):
        for j in range(instant_energy_data.shape[0]):
            if (
                instant_energy_data[j, i]
                > np.sum(local_data[j, :])
                * local_normalization_factor
                * min_power_factor
            ):
                beat[j, i] = 1

    resampled_beat = np.zeros(shape=(freq_split_num, Ns))
    for i in range(freq_split_num):
        resampled_beat[i, :] = resample_array(beat[i, :], Ns)
    return resampled_beat.copy()


def get_time_stamps_from_beat(beat, fs, normalized=False):
    beat_indicies = []
    for i in range(beat.shape[0]):
        current_range = np.logical_xor(beat[i, :-1], beat[i, 1:])
        a = np.arange(0, current_range.shape[-1])
        filter_arr = current_range > 0
        if len(current_range) != 0:
            beat_indicies.append(a[filter_arr] / fs)
    return beat_indicies


def _play_audio(wf, stream, chunk):
    global time_s
    dt = chunk / wf.getframerate()
    while len(data := wf.readframes(chunk)):
        stream.write(data)
        time_s += dt


def play_audio(wf, pa, chunk):
    global time_s, done
    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    while not done:
        print("new audio")
        wf.setpos(0)
        time_s = 0
        _play_audio(wf, stream, chunk)

    stream.close()
    wf.close()


def main():
    pa = pyaudio.PyAudio()
    global done

    ### load audio to be analyzed
    audio_path = str(pathlib.Path("./../assets/sounds/simpleKickBeat.wav").resolve())
    wf = wave.open(audio_path, "r")

    # extract numpy array fromw wave object
    buffer = wf.readframes(wf.getnframes())
    interleaved = np.frombuffer(buffer, dtype=f"int{wf.getsampwidth()*8}")
    CHANNELS = wf.getnchannels()
    audio_data = None
    if CHANNELS == 1:
        audio_data = interleaved
    else:
        # turn stereo to mono
        audio_data = np.reshape(interleaved, (CHANNELS, -1))
        audio_data = 0.5 * audio_data[0, :] + 0.5 * audio_data[1, :]

    # reset reading position of wave object
    wf.setpos(0)

    # join right and left channel if audio is stereo
    fs = wf.getframerate()
    samples = audio_data.shape[-1]
    duration_s = samples / fs

    ### analyze audio
    freq_split_positions = np.array([20, 60, 250, 500, 2000, 4000, 6000, 20000])
    # freq_ranges = np.array([])
    freq_splits = freq_split_positions.shape[0] + 1
    beat = get_beat(
        audio=audio_data,
        Ns=samples,
        Fs=fs,
        local_window_size=fs / 2,
        instant_window_size=1024,
        min_power_factor=5,
        freq_split_positions=freq_split_positions,
    )
    contact_time_stamps = [
        time_stamps
        for time_stamps in get_time_stamps_from_beat(beat, fs, normalized=False)
    ]
    for time_stamp in contact_time_stamps:
        print(time_stamp)
    normalized_contact_time_stamps = [
        time_stamps / duration_s for time_stamps in contact_time_stamps
    ]

    ### initialize pygame
    pg.init()
    RES = WIDTH, HEIGHT = 1600, 900
    screen = pg.display.set_mode(RES)
    font = pg.font.SysFont("Arial", 20)

    active_color = (255, 128, 128)
    inactive_color = (80, 80, 80)
    track_surface = pg.Surface((int(WIDTH * 4 / 5), HEIGHT))
    track_surface_rect = track_surface.get_rect(topleft=(WIDTH / 5, 0))

    ### play the audio that was loaded
    t = Thread(target=play_audio, args=(wf, pa, 1024), daemon=True)
    t.start()

    ### start main loop
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True

        track_surface.fill((40, 40, 40))
        screen.fill((60, 60, 60))

        for i, r in enumerate(freq_split_positions):
            text = ""
            if i > 0:
                text = f"{freq_split_positions[i - 1]}Hz - {r}Hz"
            else:
                text = f"0Hz - {r}Hz"
            rendered_text = font.render(text, True, (255, 255, 255))
            x = (WIDTH / 5 - rendered_text.get_width()) / 2
            y = (i + 0.5) / freq_splits * HEIGHT - rendered_text.get_height() / 2
            screen.blit(rendered_text, (x, y))
        text = f"{freq_split_positions[-1]}Hz - Inf Hz"
        rendered_text = font.render(text, True, (255, 255, 255))
        x = (WIDTH / 5 - rendered_text.get_width()) / 2
        y = (freq_splits - 0.5) / freq_splits * HEIGHT - rendered_text.get_height() / 2
        screen.blit(rendered_text, (x, y))

        for i, time_stamps in enumerate(normalized_contact_time_stamps):
            pg.draw.rect(
                track_surface,
                (20, 20, 20),
                (
                    0,
                    i / freq_splits * track_surface_rect.height,
                    track_surface_rect.width,
                    track_surface_rect.height / freq_splits,
                ),
                width=2,
            )
            for j, factor in enumerate(time_stamps):
                time_stamp = contact_time_stamps[i][j]
                blend = (1 - time_s + time_stamp) ** 3
                blend = blend if 1 > blend > 0 else 0
                color = [
                    inactive_color[i] * (1 - blend) + active_color[i] * blend
                    for i in range(3)
                ]
                x = factor * track_surface_rect.width
                y = (i + 0.5) / freq_splits * track_surface_rect.height
                pg.draw.circle(track_surface, color, (x, y), 10)
        pg.draw.line(
            track_surface,
            (120, 120, 120),
            (time_s / duration_s * track_surface_rect.width, 0),
            (time_s / duration_s * track_surface_rect.width, track_surface_rect.height),
            4,
        )

        screen.blit(track_surface, track_surface_rect)

        pg.display.update()
    pg.quit()
    quit()


if __name__ == "__main__":
    global time_s, done
    time_s = 0
    done = False
    main()
