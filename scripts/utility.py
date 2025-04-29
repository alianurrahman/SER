# -*- coding: utf-8 -*-

"""
all utilities for trainer

Usage:
    none

"""

import evaluate
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb

from scripts.etl import labels


def parse_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def _plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fix, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="augmented signal")
    plt.show()


# Evaluate Model
def compute_metrics(eval_prediction):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    predictions = np.argmax(eval_prediction.predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=eval_prediction.label_ids)["accuracy"]
    precision = \
        precision_metric.compute(predictions=predictions, references=eval_prediction.label_ids, average="weighted")[
            "precision"]
    recall = recall_metric.compute(predictions=predictions, references=eval_prediction.label_ids, average="weighted")[
        "recall"]
    f1 = f1_metric.compute(predictions=predictions, references=eval_prediction.label_ids, average="weighted")["f1"]

    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       preds=predictions,
                                                       y_true=eval_prediction.label_ids,
                                                       class_names=labels)})

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
