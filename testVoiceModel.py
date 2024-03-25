# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

# load first sample of English common_voice
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
dataset_iter = iter(dataset)
sample = next(dataset_iter)

# forward sample through model to get greedily predicted transcription ids
input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
logits = model(input_values).logits[0]
pred_ids = torch.argmax(logits, axis=-1)

# retrieve word stamps (analogous commands for `output_char_offsets`)
outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
# compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

word_offsets = [
    {
        "word": d["word"],
        "start_time": round(d["start_offset"] * time_offset, 2),
        "end_time": round(d["end_offset"] * time_offset, 2),
    }
    for d in outputs.word_offsets
]
# compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
# https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
word_offsets[:3]

# Load model directly
from transformers import AutoProcessor, AutoModelForAudioClassification

processor = AutoProcessor.from_pretrained("HyperMoon/wav2vec2-base-960h-finetuned-deepfake")
model = AutoModelForAudioClassification.from_pretrained("HyperMoon/wav2vec2-base-960h-finetuned-deepfake")
