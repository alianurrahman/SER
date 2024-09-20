from transformers import AutoFeatureExtractor
import numpy as np
import evaluate

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
accuracy = evaluate.load("accuracy")


# Preprocessing Audio Data
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs


# Evaluate Model
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
