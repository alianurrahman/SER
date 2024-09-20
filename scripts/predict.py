# -*- coding: utf-8 -*-

"""
This script is used to do prediction based on trained model

Usage:
    python3 ./scripts/predict.py

"""
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


if __name__ == "__main__":
    compute_metrics()
