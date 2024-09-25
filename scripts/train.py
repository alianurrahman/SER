# -*- coding: utf-8 -*-

"""
This script is used to train and export ML model according to config

Usage:
    python3 ./scripts/train.py

"""

import click
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from scripts.etl import feature_extractor, encoded_ser, id2label, label2id
from scripts.utility import parse_config, compute_metrics


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
    config = parse_config(config_file)
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base",
        num_labels=config["dataset"]["num_label"], id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(**config["training_arguments"]  # training args
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=encoded_ser["train"],
        eval_dataset=encoded_ser["val"], tokenizer=feature_extractor, compute_metrics=compute_metrics, )

    trainer.train()


if __name__ == "__main__":
    train()
