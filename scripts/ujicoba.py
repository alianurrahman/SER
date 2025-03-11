# -*- coding: utf-8 -*-

"""
This script is used to run convert the raw data to train and test data
It is designed to be idempotent [stateless transformation]

Usage:
    none

"""
from IPython.core.display_functions import display
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
from data_augmentation import augment_data
from imblearn.over_sampling import SMOTE

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def up_sampling(examples):
    x = [x for x in examples["input_values"]]
    y = [y for y in examples['label']]

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    return {"input_values": x_resampled, "label": y_resampled}

def preprocess_function(examples):
    """
        ETL function that load indonesian_ser data and convert to train and test set

        Args:
            config_file [str]: path to config file

        Returns:
            inputs [any]: preprocess audio
        """

    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        # max_length=16000,  # length of feature vectors * 768 dimensions  * 768 dimensions
        # truncation=True,
        padding="longest"
    )

    print('')
    return inputs


def prepare_dataset(path: str):
    """
        ETL function that load indonesian_ser data and convert to train and test set

        Args:
            path [str]: path to dataset

        Returns:
            encoded_ser [any]: preprocessed audio
            label2id [any] :
            id2label [any] :
        """
    dataset = load_dataset("audiofolder", data_dir=path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # dataset = dataset.train_test_split(test_size=.2)
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    dataset["train"] = dataset["train"].map(augment_data, remove_columns=["audio", "label"], batched=True)
    encoded_ser = dataset.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_ser["train"] = encoded_ser["train"].map(up_sampling, remove_columns="input_values", batched=True)

    return encoded_ser, dataset, label2id, id2label


if __name__ == "__main__":
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    a, b, c, d = prepare_dataset('../data')
    #
    # train_features = a["train"]["input_values"]
    # test_features = a["test"]["input_values"]

    # Convert first 100 samples to NumPy arrays
    train_features = np.array(a["train"]["input_values"][:100])
    test_features = np.array(a["test"]["input_values"][:100])

    # Concatenate train and test features
    X = np.vstack([train_features, test_features])
    labels = ["Train"] * 100 + ["Test"] * 100

    # Apply t-SNE
    X_embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    # Create DataFrame
    import pandas as pd

    df = pd.DataFrame(X_embedded, columns=["Dim1", "Dim2"])
    df["Dataset"] = labels

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Dataset", palette={"Train": "blue", "Test": "red"}, alpha=0.7)
    plt.title("t-SNE Projection of Wav2Vec2 Features (Train vs Test)")
    plt.legend()
    plt.show()
    #
    # from collections import Counter
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # # import numpy as np

    #
    # # plt.figure(figsize=(8, 5))
    # # sns.histplot(train_features[0], bins=50, kde=True)
    # # plt.xlabel("Feature Value")
    # # plt.ylabel("Frequency")
    # # plt.title("Distribution of Wav2Vec2 Features (First Sample)")
    # # plt.show()
    #
    # train_mean = np.mean([np.mean(x) for x in train_features])
    # test_mean = np.mean([np.mean(x) for x in test_features])
    # train_var = np.var([np.var(x) for x in train_features])
    # test_var = np.var([np.var(x) for x in test_features])
    #
    # print(f"Train Mean: {train_mean}, Test Mean: {test_mean}")
    # print(f"Train Variance: {train_var}, Test Variance: {test_var}")


    # Extract durations from train and test sets
    # train_durations = [x["array"].shape[0] / x["sampling_rate"] for x in a["train"]["input_values"]]
    # test_durations = [x["array"].shape[0] / x["sampling_rate"] for x in a["test"]["input_values"]]
    #
    # # Plot duration distribution
    # plt.figure(figsize=(10, 5))
    # sns.histplot(train_durations, bins=30, kde=True, color="blue", label="Train", alpha=0.6)
    # sns.histplot(test_durations, bins=30, kde=True, color="red", label="Test", alpha=0.6)
    # plt.xlabel("Duration (seconds)")
    # plt.ylabel("Count")
    # plt.title("Distribution of Audio Durations")
    # plt.legend()
    # plt.show()


#distribution
    # train_labels = b["train"]["label"]
    # test_labels = b["test"]["label"]
    #
    # train_counts = Counter(train_labels)
    # test_counts = Counter(test_labels)
    #
    # train_df = pd.DataFrame(train_counts.items(), columns=["Label", "Count"])
    # train_df["Split"] = "Train"
    #
    # test_df = pd.DataFrame(test_counts.items(), columns=["Label", "Count"])
    # test_df["Split"] = "Test"
    #
    # df = pd.concat([train_df, test_df])
    #
    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=df, x="Label", y="Count", hue="Split", palette="viridis")
    # plt.xlabel("Emotion Label")
    # plt.ylabel("Count")
    # plt.title("Distribution of Labels in Train and Test Sets")
    # plt.xticks(rotation=45)  # Rotate for better readability if needed
    # plt.legend()
    # plt.show()


    # print(b)
    # dfb = b["train"].to_pandas()
    # print(dfb.shape)
    #
    # # print(dataset["train"][0])
    # # print(dataset["test"][0])
    #
    # print(a)
    # # print(encoded_ser["train"][0])
    # # print(encoded_ser["test"][0])
    #
    # # print(prepare_dataset("../data"))
    # # a, b, c = prepare_dataset("../data")
    # df = a["train"].to_pandas()
    # # display(df)
    # print(df.shape)
    # df2 = a["test"].to_pandas()
    # # display(df2)
    # print(df2.shape)
    # print(df.head())
    # print(df.describe())
    # print(df.info())
