import evaluate
from datasets import load_dataset


def coba():
    dataset = load_dataset("audiofolder", data_dir="../data")
    print(dataset["train"].features["label"].names)
    print(dataset["train"].features)
    print(dataset["train"].features["audio"])


import yaml

with open('config.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file


def process(**params):  # pass in variable numbers of args
    for key, value in params.items():
        print('%s: %s' % (key, value))


def evaluasi():
    confusion_metric = evaluate.load("confusion_matrix")
    result = confusion_metric.compute(references=[0, 1, 1, 2, 0, 2, 2], predictions=[0, 2, 1, 1, 0, 2, 0])
    print(result)


if __name__ == "__main__":
    # process(**conf["training_arguments"])
    evaluasi()
