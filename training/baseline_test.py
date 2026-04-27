from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline
from evaluate import load

DATASET_DIR = Path("dataset")
WAV_DIR = DATASET_DIR / "extracted_wavs"
TEST_CSV = DATASET_DIR / "metadata_test.csv"

MODEL_NAME = "openai/whisper-small"
N_SAMPLES = 20

device = 0 if torch.cuda.is_available() else -1

df = pd.read_csv(TEST_CSV).head(N_SAMPLES)

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=device,
)

wer_metric = load("wer")
cer_metric = load("cer")

predictions = []
references = []

for _, row in df.iterrows():
    wav_path = WAV_DIR / f"{row['id']}.wav"
    reference = str(row["text"])

    result = asr(
        str(wav_path),
        generate_kwargs={
            "task": "transcribe",
            "language": "arabic",
        },
    )

    prediction = result["text"]

    predictions.append(prediction)
    references.append(reference)

    print("\nID:", row["id"])
    print("REF:", reference)
    print("PRED:", prediction)
    print("-" * 80)

wer = wer_metric.compute(predictions=predictions, references=references)
cer = cer_metric.compute(predictions=predictions, references=references)

print("\n======================")
print("BASELINE RESULT")
print("======================")
print("Samples:", N_SAMPLES)
print("WER:", wer)
print("CER:", cer)
