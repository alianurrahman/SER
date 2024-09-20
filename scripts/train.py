# -*- coding: utf-8 -*-

"""
This script is used to train and export ML model according to config

Usage:
    python3 ./scripts/train.py

"""
from pathlib import Path
from pickle import dump

import click
import pandas as pd
import sklearn
from transformers import AutoFeatureExtractor

from utility import load_data, parse_config


@click.command()
@click.argument("config_file", type=str, default="scripts/config.yml")
def train(config_file):
    """
    Main function that trains & persists model based on training set

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    training_args = TrainingArguments(
        output_dir="my_awesome_mind_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def initiate_model(ensemble_model, model_config):
    """
    initiate model using eval, implement with defensive programming

    Args:
        ensemble_model [str]: name of the ensemble model

    Returns:
        [sklearn.model]: initiated model
    """
    if ensemble_model in dir(sklearn.ensemble):
        return eval("sklearn.ensemble." + ensemble_model)(**model_config)
    else:
        raise NameError(f"{ensemble_model} is not in sklearn.ensemble")


if __name__ == "__main__":
    train()
