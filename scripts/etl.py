# -*- coding: utf-8 -*-

"""
This script is used to run convert the raw data to train and test data
It is designed to be idempotent [stateless transformation]

Usage:
    none

"""

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
dataset = load_dataset("audiofolder", data_dir="../data/examples", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

dataset = dataset.train_test_split(test_size=.2)
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


def preprocess_function(examples):
    """
        ETL function that load indonesian_ser data and convert to train and test set

        Args:
            config_file [str]: path to config file

        Returns:
            inputs [any]: preprocess audio
        """
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,  # length of feature vectors * 768 dimensions  * 768 dimensions
        truncation=True,
        padding=True
    )
    return inputs


encoded_ser = dataset.map(preprocess_function, remove_columns="audio", batched=True)
