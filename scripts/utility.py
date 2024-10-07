import evaluate
import numpy as np
import yaml


def parse_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


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

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
