# -*- coding: utf-8 -*-

"""
data augmentation for training data

Usage:
    none

"""

import librosa
import numpy as np


def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal


def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, rate=stretch_rate)




if __name__ == "__main__":
    audio, sr = librosa.load("test.wav")
    time_stretch(audio, 0.8)
