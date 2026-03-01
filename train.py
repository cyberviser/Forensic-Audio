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

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainerCallback,
    VoxtralRealtimeForConditionalGeneration,
)
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
SAMPLE_RATE       = 16000
MEL_FRAMES        = 1500
HOP_LENGTH        = 160
AUDIO_MAX_SAMPLES = MEL_FRAMES * HOP_LENGTH  # 240000 = 15 seconds
MAX_NEW_TOKENS    = 512

if WANDB_KEY:
    import wandb
    os.environ["WANDB_PROJECT"]      = "voxtral-sentinel"
    os.environ["WANDB_RESUME"]       = "allow"
    os.environ["WANDB_HTTP_TIMEOUT"] = "300"
    wandb.login(key=WANDB_KEY)
    wandb.init(project="voxtral-sentinel", resume="allow")


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
# 2. Audio helpers
# =============================================================================
def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n < AUDIO_MAX_SAMPLES:
        audio = np.pad(audio, (0, AUDIO_MAX_SAMPLES - n))
    elif n > AUDIO_MAX_SAMPLES:
        audio = audio[:AUDIO_MAX_SAMPLES]
    return audio


def _audio_entry_to_array(entry) -> np.ndarray:
    """
    Convert a HuggingFace Audio column entry to a float32 numpy array at SAMPLE_RATE.
    Handles both the decoded dict format {'array': ..., 'sampling_rate': ...}
    and the raw bytes dict format {'bytes': ..., 'path': ...}.
    """
    if isinstance(entry, dict):
        if "array" in entry:
            audio = np.array(entry["array"], dtype=np.float32)
            sr    = entry.get("sampling_rate", SAMPLE_RATE)
        elif entry.get("bytes"):
            audio, sr = sf.read(io.BytesIO(entry["bytes"]))
            audio     = audio.astype(np.float32)
        elif entry.get("path") and os.path.exists(entry["path"]):
            audio, sr = sf.read(entry["path"])
            audio     = audio.astype(np.float32)
        else:
            raise ValueError(f"No usable audio in entry: {list(entry.keys())}")
    else:
        raise ValueError(f"Unexpected audio entry type: {type(entry)}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return _pad_or_trim(audio)


def _array_to_base64_wav(audio: np.ndarray) -> str:
    """Encode a float32 numpy array as a base64 WAV string for apply_chat_template."""
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# 3. Dataset
# =============================================================================
print("Loading dataset...")
# Dataset has columns: audio (Audio), prompt (string), answer (string)
# prompt = "Analyze this recording for forensic indicators."
# answer = "### TRANSCRIPT:\n...\n### ANALYSIS:\n...\n### CONCLUSION:\n..."
ds = load_dataset(DATASET, token=HF_TOKEN)

# Disable Audio decoding to avoid torchcodec dependency.
# Audio stays as a raw {"bytes": ..., "path": ...} dict;
# _audio_entry_to_array() in the collator decodes it via soundfile.
ds = datasets.DatasetDict({
    split: ds[split].cast_column("audio", datasets.Audio(decode=False))
    for split in ds
})

# Use existing train/test splits if present, otherwise create them
if "train" in ds and "test" in ds:
    train_ds = ds["train"]
    eval_ds  = ds["test"]
else:
    split    = ds["train"].train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")


# =============================================================================
# 4. Data Collator
# =============================================================================
@dataclass
class VoxtralDataCollator:
    """
    Builds batches using apply_chat_template so audio and text are properly
    interleaved in the model's expected format.

    Each example is formatted as:
        User:      [audio] + "Analyze this recording for forensic indicators."
        Assistant: "### TRANSCRIPT:...\n### ANALYSIS:...\n### CONCLUSION:..."

    Labels are masked on the user turn so loss is only computed on the
    assistant response -- this is critical for correct fine-tuning.
    """
    processor: Any

    def _build_messages(self, audio_b64: str, prompt: str, answer: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

    def _mask_user_turn(self, input_ids: list, answer: str) -> list:
        """
        Return labels where the user turn tokens are replaced with -100.
        Loss is only computed on the assistant response tokens.

        Scans input_ids from the right to find where the answer token sequence
        begins, so masking is anchored on actual content rather than assumed
        lengths. Robust to any special tokens apply_chat_template adds.
        """
        answer_ids = self.processor.tokenizer(
            answer, add_special_tokens=False
        )["input_ids"]
        answer_len = len(answer_ids)

        labels = list(input_ids)

        first_answer_tok = answer_ids[0]
        split_pos = None
        for i in range(len(labels) - answer_len, -1, -1):
            if labels[i] == first_answer_tok and labels[i:i + answer_len] == answer_ids:
                split_pos = i
                break

        if split_pos is None:
            # Fallback: mask everything except the last answer_len+1 tokens
            split_pos = max(0, len(labels) - answer_len - 1)

        for i in range(split_pos):
            labels[i] = -100

        return labels

    def __call__(self, examples: list[dict]) -> dict:
        all_input_ids        = []
        all_attention_mask   = []
        all_labels           = []
        all_audio_arrays     = []
        all_num_delay_tokens = []

        for ex in examples:
            audio_array = _audio_entry_to_array(ex["audio"])
            audio_b64   = _array_to_base64_wav(audio_array)
            messages    = self._build_messages(audio_b64, ex["prompt"], ex["answer"])

            # continue_final_message=True tells the tokenizer this conversation
            # ends with an assistant turn (training completion), not a user turn
            # (serving request). Without this flag mistral_common raises
            # InvalidMessageStructureException.
            tokenized = self.processor.tokenizer.apply_chat_template(
                messages,
                return_tensors=None,
                return_dict=True,
                continue_final_message=True,
            )

            labels = self._mask_user_turn(tokenized["input_ids"], ex["answer"])

            all_input_ids.append(tokenized["input_ids"])
            all_attention_mask.append(tokenized["attention_mask"])
            all_labels.append(labels)

            # num_delay_tokens is a scalar the model uses internally for the
            # streaming audio alignment -- must be included if present
            if "num_delay_tokens" in tokenized:
                all_num_delay_tokens.append(tokenized["num_delay_tokens"])

            # audio is a list of numpy arrays returned by apply_chat_template
            raw_audio = tokenized.get("audio", [audio_array])
            all_audio_arrays.append(raw_audio[0] if raw_audio else audio_array)

        # Pad text sequences to the longest in the batch
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id  = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attn_mask = []
        padded_labels    = []

        for ids, mask, labs in zip(all_input_ids, all_attention_mask, all_labels):
            pad = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad)
            padded_attn_mask.append(mask + [0] * pad)
            padded_labels.append(labs + [-100] * pad)

        # Build audio input_features via feature_extractor
        input_features = self.processor.feature_extractor(
            all_audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MEL_FRAMES,
        )["input_features"]

        batch = {
            "input_ids":      torch.tensor(padded_input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(padded_labels,     dtype=torch.long),
            "input_features": input_features.to(torch.bfloat16),
        }

        # Include num_delay_tokens if the processor returned it
        if all_num_delay_tokens:
            batch["num_delay_tokens"] = torch.tensor(
                all_num_delay_tokens, dtype=torch.long
            )

        return batch


collator = VoxtralDataCollator(processor=processor)

# Sanity-check the collator on one example before committing to a full run
print("Running collator sanity check...")
_sample_batch = collator([train_ds[0]])
assert "input_ids"      in _sample_batch, "Missing input_ids"
assert "input_features" in _sample_batch, "Missing input_features"
assert "labels"         in _sample_batch, "Missing labels"
_labels   = _sample_batch["labels"][0]
_unmasked = (_labels != -100).sum().item()
_total    = len(_labels)
assert _unmasked > 0,      "All labels are masked -- masking logic is broken"
assert _unmasked < _total, "No labels are masked -- user turn is included in loss"
print(f"  input_ids shape     : {_sample_batch['input_ids'].shape}")
print(f"  input_features shape: {_sample_batch['input_features'].shape}")
print(f"  labels              : {_unmasked}/{_total} tokens unmasked (assistant only)")
if "num_delay_tokens" in _sample_batch:
    print(f"  num_delay_tokens    : {_sample_batch['num_delay_tokens']}")
del _sample_batch
print("Collator sanity check passed.")


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
            print(f"\nEval loss {eval_loss:.4f} below threshold {self.threshold} -- stopping.")
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
    save_strategy="no",
    load_best_model_at_end=False,
    report_to="wandb" if WANDB_KEY else "none",
    push_to_hub=False,
    hub_model_id=OUTPUT_DIR,
    hub_token=HF_TOKEN,
    dataset_text_field="",
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

print("Saving and pushing model to Hub...")
trainer.save_model(OUTPUT_DIR)
trainer.push_to_hub()
print(f"\nDone! Model at: https://huggingface.co/{OUTPUT_DIR}")

if WANDB_KEY:
    wandb.finish()

# =============================================================================
# INFERENCE NOTE
# =============================================================================
# At inference, pass audio + prompt via apply_chat_template (NOT audio-only).
# The model learned: audio + prompt -> structured response.
# Audio-only input will produce degenerate output.
#
#   import base64, io, soundfile as sf
#   buf = io.BytesIO()
#   sf.write(buf, audio_array, 16000, format="WAV")
#   audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#
#   messages = [{
#       "role": "user",
#       "content": [
#           {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
#           {"type": "text", "text": "Analyze this recording for forensic indicators."},
#       ]
#   }]
#   tokenized = processor.tokenizer.apply_chat_template(
#       messages, return_tensors=None, return_dict=True
#   )
#   input_ids      = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(model.device)
#   attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(model.device)
#   input_features = processor.feature_extractor(
#       tokenized["audio"], sampling_rate=16000, return_tensors="pt"
#   )["input_features"].to(model.device, dtype=model.dtype)
#
#   output_ids = model.generate(
#       input_ids=input_ids, attention_mask=attention_mask,
#       input_features=input_features, max_new_tokens=512, do_sample=False,
#   )
#   print(processor.tokenizer.decode(output_ids[0], skip_special_tokens=True))