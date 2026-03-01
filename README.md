# Forensic-Audio

> Fine-tuning [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) to explain the context, environment, and emotional subtext of audio for forensic analysis purposes.

**Built for the [Mistral Worldwide Hackathon](https://mistral.ai).**

---

## What It Does

Given a raw audio recording, the fine-tuned model (**voxtral-sentinel-4b**) produces structured output containing:

1. **Transcript** — verbatim transcription of the speech
2. **Analysis** — expert assessment of vocal cues, emotion, tone, and environmental context
3. **Conclusion** — recommended action or risk classification

This enables real-time audio triage for **automated customer support** (intent classification, escalation routing) and **emergency services** (distress detection, dispatcher assistance) without human-in-the-loop intervention.

```
### TRANSCRIPT:
I need help immediately, my neighbour hasn't responded in hours and I can hear something...

### ANALYSIS:
The speaker exhibits elevated vocal stress indicators including increased speech rate and
pitch variance. Tone suggests genuine distress rather than rehearsed or non-urgent
communication. Situational context implies potential welfare concern for a third party.

### CONCLUSION:
Escalate to emergency services. Flag as high-priority welfare check. Do not route to
standard support queue.
```

---

## Repository Structure

```
Forensic-Audio/
├── create_dataset.py      # Generate JSONL annotations via Mistral API (MELD + DCASE sources)
├── dataset_packer.py      # Pack audio + annotations into a HuggingFace dataset and push to Hub
├── train.py               # training script (apply_chat_template, eval split, early stopping)
├── check_api.py           # Quick sanity check for Mistral API connectivity
├── debug_fields.py        # Inspect raw MELD dataset fields
├── requirements.txt       # Python dependencies
├── voxtral_forensic_train.jsonl        # Small annotation file
├── voxtral_forensic_train_large.jsonl  # Full annotation file (~12,500 samples)
├── LICENSE                # GPL-3.0
└── README.md
```

---

## Pipeline Overview

The project follows a three-stage pipeline:

### 1. Dataset Generation (`create_dataset.py`)

Streams audio from [MELD](https://huggingface.co/datasets/ajyy/MELD_audio) (emotional dialogue) and [DCASE/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) (acoustic scenes). For each sample, it calls **Mistral Small** to produce a structured forensic analysis (transcript, analysis, conclusion) conditioned on the speech text, detected emotion, and scene label. Results are written to a JSONL file.

### 2. Dataset Packing (`dataset_packer.py`)

Matches JSONL annotations to raw audio files from the MELD archive on HuggingFace. Normalises all answers into a canonical three-section format (regex-based, with Mistral fallback for edge cases). Builds a HuggingFace `Dataset` with `audio`, `prompt`, and `answer` columns, creates a 95/5 train/test split, and pushes to the Hub.

### 3. Training (`train.py`)

Full fine-tune (no LoRA) of Voxtral-Mini-4B-Realtime on an NVIDIA A100. Uses `apply_chat_template` to properly interleave audio and text tokens. Labels are masked on user turns so loss is computed only on the assistant response. Includes early stopping when eval loss drops below a threshold.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 recommended for training; inference works on smaller GPUs)
- Accounts: [HuggingFace](https://huggingface.co) and [Mistral AI](https://console.mistral.ai) (for dataset generation)

### Installation

```bash
git clone https://github.com/SageRish/Forensic-Audio.git
cd Forensic-Audio
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token
MISTRAL_API_KEY=your_mistral_api_key
WANDB_API_KEY=your_wandb_key          # optional, for experiment tracking
```

---

## Usage

### Generate Dataset Annotations

```bash
python create_dataset.py
```

Produces `voxtral_forensic_train_large.jsonl` with structured annotations for up to 9,988 audio samples.

### Pack and Upload Dataset

```bash
python dataset_packer.py
```

Streams audio from the MELD archive, matches it to annotations, normalises formatting, and pushes the final dataset to [`trishtan/voxtral-forensic-ds`](https://huggingface.co/datasets/trishtan/voxtral-forensic-ds).

### Train

```bash
python train.py
```

Runs full fine-tuning with the following configuration:

| Parameter | Value |
|---|---|
| Epochs | 5 (early stopping at eval loss < 1.15) |
| Learning rate | 5e-6 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Max grad norm | 1.0 |
| Precision | bfloat16 |
| Eval strategy | Every 100 steps |

Training logs are available on [Weights & Biases](https://wandb.ai/s222458666/voxtral-sentinel/runs/o39i1rfw).

### Inference

```python
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

model_id = "trishtan/voxtral-sentinel-4b"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
)

audio, sr = sf.read("your_audio.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)

PROMPT = "[INST] Analyze this recording for forensic indicators. [/INST]"

audio_inputs = processor.feature_extractor(
    [audio], sampling_rate=16000, return_tensors="pt", padding=True,
)
text_inputs = processor.tokenizer(
    [PROMPT], return_tensors="pt", padding=True,
)
inputs = {**audio_inputs, **text_inputs}
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

print(processor.tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Results

| Metric | Value |
|---|---|
| Final eval loss | **1.144** |
| Mean token accuracy | **74.25%** |
| Train/eval accuracy gap | ~0% |
| Stopped at epoch | 1.33 (early stopping) |

The near-zero train/eval gap indicates the model generalises well to unseen audio with no measurable overfitting.

---

## Model & Dataset Links

| Resource | Link |
|---|---|
| Fine-tuned model | [trishtan/voxtral-sentinel-4b](https://huggingface.co/trishtan/voxtral-sentinel-4b) |
| Training dataset | [trishtan/voxtral-forensic-ds](https://huggingface.co/datasets/trishtan/voxtral-forensic-ds) |
| Base model | [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) |
| W&B run | [voxtral-sentinel](https://wandb.ai/s222458666/voxtral-sentinel/runs/o39i1rfw) |

---

## Data Attribution

**MELD — Multimodal EmotionLines Dataset**
> Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.* ACL 2019, pp. 527–536.
> HuggingFace: [ajyy/MELD_audio](https://huggingface.co/datasets/ajyy/MELD_audio)

**DCASE 2025 — Acoustic Scene Classification**
> Mesaros, A., Heittola, T., & Virtanen, T. (2018). *A multi-device dataset for urban acoustic scene classification.* DCASE 2018 Workshop, pp. 9–13.
> https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

The fine-tuned model inherits the base model license from [Mistral AI](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).
