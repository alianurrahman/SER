import click
from transformers import AutoFeatureExtractor

from scripts.utility import parse_config

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")


# Preprocessing Audio Data
@click.command
@click.argument("config_file", type=str, default="scripts/config.yml")
def preprocess_function(config_file):
    config = parse_config(config_file)

    audio_arrays = [x["array"] for x in config["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs


if __name__ == "__main__":
    preprocess_function()