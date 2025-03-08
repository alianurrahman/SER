# -*- coding: utf-8 -*-

"""
This script is used to train and export ML model according to config

Usage:
    python3 ./scripts/train.py
"""

import click
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from scripts.etl import feature_extractor, prepare_dataset
from scripts.utility import parse_config, compute_metrics

encoded_ser, label2id, id2label = prepare_dataset("../data")


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def train(config_file):
    """
    Main function that trains & persists model based on training set

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    config = parse_config(config_file)
    model = AutoModelForAudioClassification.from_pretrained(
        config["pre_train_model"]["wav2vec2_base"],
        num_labels=config["dataset"]["num_label"],
        id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(**config["training_arguments"]  # training args
                                      )

    trainer = Trainer(model=model, args=training_args, train_dataset=encoded_ser["train"].with_format("torch"),
                      eval_dataset=encoded_ser["test"].with_format("torch"), processing_class=feature_extractor,
                      compute_metrics=compute_metrics, )

    print(trainer.args.device)
    trainer.train()


if __name__ == "__main__":
    train()
