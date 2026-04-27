import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="arabic",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Model loaded:", MODEL_NAME)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    model = model.to("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

print("Whisper Small test passed.")
