import torch

import evaluate
from datasets import load_dataset

def config_yaml_test():
    dataset = load_dataset("audiofolder", data_dir="../data")
    print(dataset["train"].features["label"].names)
    print(dataset["train"].features)
    print(dataset["train"].features["audio"])


import yaml

with open('config.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

def cuda_device_test():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


def process(**params):  # pass in variable numbers of args
    for key, value in params.items():
        print('%s: %s' % (key, value))


def conf_matrix_evaluate():
    confusion_matrix = evaluate.load("confusion_matrix")
    result = confusion_matrix.compute(references=[0, 1, 1, 2, 0, 2, 2], predictions=[0, 2, 1, 1, 0, 2, 0])
    print(result)


if __name__ == "__main__":
    cuda_device_test()