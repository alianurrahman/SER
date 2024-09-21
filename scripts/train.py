# -*- coding: utf-8 -*-

"""
This script is used to train and export ML model according to config

Usage:
    python3 ./scripts/train.py

"""

import click
import sklearn
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

from scripts.etl import feature_extractor
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
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=config["training_arguments"]["output_dir"],
        evaluation_strategy=config["training_arguments"]["evaluation_strategy"],
        save_strategy=config["training_arguments"]["save_strategy"],
        learning_rate=config["training_arguments"]["learning_rate"],
        per_device_train_batch_size=config["training_arguments"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training_arguments"]["gradient_accumulation_steps"],
        per_device_eval_batch_size=config["training_arguments"]["per_device_eval_batch_size"],
        num_train_epochs=config["training_arguments"]["num_train_epochs"],
        warmup_ratio=config["training_arguments"]["warmup_ratio"],
        logging_steps=config["training_arguments"]["logging_steps"],
        load_best_model_at_end=config["training_arguments"]["load_best_model_at_end"],
        metric_for_best_model=config["training_arguments"]["accuracy"],
        push_to_hub=config["training_arguments"]["push_to_hub"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=config["trainer"]["train_dataset"],
        eval_dataset=config["trainer"]["eval_dataset"],
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
