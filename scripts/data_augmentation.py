# -*- coding: utf-8 -*-

"""
data augmentation for training data

Usage:
    none

"""
import librosa
import numpy as np
from imblearn.over_sampling import SMOTE


def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_factor
    return augmented_signal


def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, rate=stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=num_semitones)


def augment_data(examples):
    outputs=[]
    labels=[]
    for audio, label in zip(examples['audio'], examples['label']):
        label = label
        path = audio['path']
        sampling_rate = audio['sampling_rate']
        audio_array = audio["array"]

        augmented_white_noise = add_white_noise(audio_array, .035)
        augmented_pitch_scale = pitch_scale(audio_array, sampling_rate, .7)

        augmented_white_noise_output = [
            {"path": path, "array": augmented_white_noise, "sampling_rate": sampling_rate}]
        augmented_pitch_scale_output = [
            {"path": path, "array": augmented_pitch_scale, "sampling_rate": sampling_rate}]

        outputs += [audio] + augmented_white_noise_output + augmented_pitch_scale_output
        labels.extend([label] * 3)

    return {"audio": outputs, "label": labels}


def up_sampling(examples):
    x = [x for x in examples["input_values"]]
    y = [y for y in examples['label']]

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    return {"input_values": x_resampled, "label": y_resampled}
