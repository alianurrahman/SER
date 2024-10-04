import evaluate
import numpy as np
import yaml


def parse_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


# Evaluate Model
def compute_metrics(eval_prediction):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("f1")
    predictions = np.argmax(eval_prediction.predictions, axis=1)

    accuracy = metric1.compute(predictions=predictions, references=eval_prediction.label_ids)["accuracy"]
    f1 = metric2.compute(predictions=predictions, references=eval_prediction.label_ids, average="micro")["f1"]

    return {"accuracy": accuracy, "f1": f1}
