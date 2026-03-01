"""
Custom SFT training script for Voxtral-Mini-4B-Realtime.
Full fine-tune (no LoRA) on an A100.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "trl",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "wandb",
#   "soundfile",
#   "librosa",
#   "mistral-common",
# ]
# ///

import io
import os
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import datasets
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoProcessor, TrainerCallback, VoxtralRealtimeForConditionalGeneration
from trl import SFTConfig, SFTTrainer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN          = os.getenv("HF_TOKEN")
WANDB_KEY         = os.getenv("WANDB_API_KEY")
MODEL_NAME        = "mistralai/Voxtral-Mini-4B-Realtime-2602"
DATASET           = "trishtan/voxtral-forensic-ds"
OUTPUT_DIR        = "trishtan/voxtral-sentinel-4b"
DATASET_REPO      = "trishtan/voxtral-forensic-ds-splits"
SAMPLE_RATE       = 16000
MAX_TEXT_TOKENS   = 512
MEL_FRAMES        = 1500
HOP_LENGTH        = 160
AUDIO_MAX_SAMPLES = MEL_FRAMES * HOP_LENGTH  # 240000 = 15 seconds

# WandB setup — init before Trainer so we own the run lifecycle
# and wandb.finish() cleanly closes it (prevents "crashed" status)
if WANDB_KEY:
    import wandb
    os.environ["WANDB_PROJECT"]      = "voxtral-sentinel"
    os.environ["WANDB_RESUME"]       = "allow"
    os.environ["WANDB_HTTP_TIMEOUT"] = "300"
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="voxtral-sentinel",
        resume="allow",
    )


# =============================================================================
# 1. Processor and Model
# =============================================================================
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, trust_remote_code=True, token=HF_TOKEN,
)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto",
)
model.config.use_cache = False


# =============================================================================
# 2. Audio helper
# =============================================================================
def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n < AUDIO_MAX_SAMPLES:
        audio = np.pad(audio, (0, AUDIO_MAX_SAMPLES - n))
    elif n > AUDIO_MAX_SAMPLES:
        audio = audio[:AUDIO_MAX_SAMPLES]
    return audio


# =============================================================================
# 3. Dataset
# =============================================================================
print("Loading dataset...")
ds = load_dataset(DATASET, token=HF_TOKEN, split="train")

# Cast Audio() away before any iteration to avoid torchcodec import
new_features = {}
for col, feat in ds.features.items():
    if isinstance(feat, datasets.Audio):
        new_features[col] = {"bytes": Value("binary"), "path": Value("string")}
    else:
        new_features[col] = feat
ds = ds.cast(Features(new_features))


def process_audio_and_text(example):
    entry = example["audio"]
    if entry.get("bytes"):
        audio, sr = sf.read(io.BytesIO(entry["bytes"]))
    elif entry.get("path") and os.path.exists(entry["path"]):
        audio, sr = sf.read(entry["path"])
    else:
        raise ValueError(f"No usable audio. Keys: {list(entry.keys())}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return {
        "text":  f"[INST] {example['prompt']} [/INST] {example['answer']}",
        "audio": _pad_or_trim(audio).tolist(),
    }


print("Processing dataset...")
ds = ds.map(process_audio_and_text, remove_columns=ds.column_names)
ds = ds.cast(Features({
    "text":  Value("string"),
    "audio": Sequence(Value("float32")),
}))

# 90/10 train/eval split — fixed seed for reproducibility
split    = ds.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]
print(f"Train: {len(train_ds)} examples, Eval: {len(eval_ds)} examples")

# Export splits to Hub for benchmarking base vs fine-tuned model
print("Pushing train/eval splits to Hub...")
train_ds.push_to_hub(DATASET_REPO, split="train", token=HF_TOKEN, private=True)
eval_ds.push_to_hub(DATASET_REPO, split="test",  token=HF_TOKEN, private=True)
print(f"Splits at: https://huggingface.co/datasets/{DATASET_REPO}")


# =============================================================================
# 4. Data Collator
# =============================================================================
@dataclass
class VoxtralDataCollator:
    processor: Any
    max_text_tokens: int = MAX_TEXT_TOKENS
    max_mel_frames:  int = MEL_FRAMES

    def __call__(self, examples):
        texts  = [ex["text"] for ex in examples]
        audios = [_pad_or_trim(np.array(ex["audio"], dtype=np.float32)) for ex in examples]

        audio_batch = self.processor.feature_extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_mel_frames,
        )
        text_batch = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
        )

        batch = {**audio_batch, **text_batch}
        batch["labels"] = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            batch["labels"][batch["labels"] == pad_id] = -100
        return batch


collator = VoxtralDataCollator(processor=processor)


# =============================================================================
# 5. Early stopping callback
# =============================================================================
class LossThresholdCallback(TrainerCallback):
    """Stop training once eval_loss drops below threshold."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.threshold:
            print(f"\nEval loss {eval_loss:.4f} below threshold {self.threshold} — stopping.")
            control.should_training_stop = True
        return control


# =============================================================================
# 6. Training Arguments
# =============================================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",            # no intermediate saves — push once at end
    load_best_model_at_end=False,  # requires matching save_strategy, disabled
    report_to="wandb" if WANDB_KEY else "none",
    push_to_hub=False,             # single push at end instead
    hub_model_id=OUTPUT_DIR,
    hub_token=HF_TOKEN,
    dataset_text_field="",
    max_length=MAX_TEXT_TOKENS,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)


# =============================================================================
# 7. Trainer
# =============================================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    processing_class=processor.tokenizer,
    callbacks=[LossThresholdCallback(threshold=1.15)],
)

print("Starting training...")
trainer.train()

# Single save and push at end
print("Saving and pushing model to Hub...")
trainer.save_model(OUTPUT_DIR)
trainer.push_to_hub()
print(f"\nDone! Model at: https://huggingface.co/{OUTPUT_DIR}")

# Cleanly close WandB — prevents "crashed" status on HF job instances
if WANDB_KEY:
    wandb.finish()