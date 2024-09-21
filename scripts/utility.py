from __future__ import unicode_literals

from pydub import AudioSegment
import yaml
import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")


def parse_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def convert_audio(audio_file):
    """
    Corrects the channels, sample rate, and sample width of the audios.
    Replaces the original audio file with the one generated.
    """
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)  # 2 corresponds to 16-bit sample width in Pydub
    sound.export(audio_file, format="wav")


# Evaluate Model
def compute_metrics(eval_prediction):
    predictions = np.argmax(eval_prediction.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_prediction.label_ids)
