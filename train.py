"""
Custom SFT training script for Voxtral-Mini-4B-Realtime.
Bypasses trl-jobs which is hardcoded to AutoModelForCausalLM.
Run on an A100 instance
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "trl",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "wandb",
#   "librosa",
#   "mistral-common",
#   "soundfile", 
# ]
# ///

import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import io
import soundfile as sf
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Any
from transformers import AutoProcessor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading skipped; rely on shell environment variables

HF_TOKEN    = os.getenv("HF_TOKEN")
WANDB_KEY   = os.getenv("WANDB_API_KEY")   # optional but recommended
MODEL_NAME  = "mistralai/Voxtral-Mini-4B-Realtime-2602"
DATASET     = "trishtan/voxtral-forensic-ds"
OUTPUT_DIR  = "trishtan/voxtral-sentinel-4b"

if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = "voxtral-sentinel"

@dataclass
class VoxtralDataCollator:
    processor: Any
    max_length: int = 512

    def __call__(self, examples):
        texts = []
        audios = []

        for ex in examples:
            texts.append(ex["text"])
            # Convert back from list to numpy array if needed
            audio = ex["audio"]
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            audios.append(audio)

        # Run the full processor — handles both mel features + tokenization
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Labels = input_ids shifted inside the model, but trainer needs them explicit
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.processor.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return batch

# ── 1. Processor & Model ────────────────────────────────────────────────────
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=HF_TOKEN,
)

model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── 2. LoRA / PEFT ──────────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",  # adapts all linear layers
    bias="none",
)
# Do NOT call get_peft_model here — SFTTrainer applies peft_config itself

# ── 3. Dataset ───────────────────────────────────────────────────────────────
print("Loading and manual decoding of dataset...")
from datasets import load_dataset, Value, Sequence

# Load with audio decoding DISABLED
ds = load_dataset(DATASET, token=HF_TOKEN, split="train")

# ── Critical: remove the Audio() feature type BEFORE any iteration ──
# Replace Audio() with raw bytes/path struct so torchcodec is never invoked
from datasets import Features, Image
import datasets

# Cast the audio column to a plain dict of bytes + metadata
# This prevents datasets from ever calling the audio decoder
new_features = {}
for col, feat in ds.features.items():
    if isinstance(feat, datasets.Audio):
        # Store as plain struct — no decoder triggered
        new_features[col] = {
            "bytes": Value("binary"),
            "path": Value("string"),
        }
    else:
        new_features[col] = feat

ds = ds.cast(datasets.Features(new_features))

# Now it's safe to map — audio is just a dict with bytes/path
def process_audio_and_text(example):
    import io, soundfile as sf, numpy as np

    audio_entry = example["audio"]
    
    if audio_entry.get("bytes"):
        audio_data, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
    elif audio_entry.get("path") and os.path.exists(audio_entry["path"]):
        audio_data, sr = sf.read(audio_entry["path"])
    else:
        raise ValueError(f"No audio data found in example: {audio_entry.keys()}")

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sr, target_sr=16000)

    return {
        "text": f"[INST] {example['prompt']} [/INST] {example['answer']}",
        "audio": audio_data.astype(np.float32).tolist(),
    }

ds = ds.map(
    process_audio_and_text,
    remove_columns=ds.column_names,  # drop ALL original columns
)

# ── 4. Training Arguments ────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=150,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    report_to="wandb" if WANDB_KEY else "none",
    push_to_hub=True,
    hub_model_id=OUTPUT_DIR,
    hub_token=HF_TOKEN,
)

# ── 5. Trainer ───────────────────────────────────────────────────────────────
collator = VoxtralDataCollator(processor=processor, max_length=512)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    peft_config=peft_config,
    data_collator=collator,
    processing_class=processor.tokenizer,  # for tokenizer-side things like pad token
)

print("Starting training...")
trainer.train()

print("Pushing final model to Hub...")
trainer.push_to_hub()
print(f"\n✅ Done! Model at: https://huggingface.co/{OUTPUT_DIR}")
