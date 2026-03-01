"""Simple Gradio demo for Voxtral Sentinel transcription.

Run:
    pip install gradio==6.8.0 transformers torch soundfile librosa accelerate
    python Demo/app.py

Optional env vars:
    HF_TOKEN - Hugging Face token for gated/private access.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Tuple

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

MODEL_ID = "trishtan/voxtral-sentinel-4b"
BASE_PROCESSOR_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SAMPLE_RATE = 16000
MAX_SECONDS = 30
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS
HF_TOKEN = os.getenv("HF_TOKEN")

processor: AutoProcessor | None = None
model: VoxtralRealtimeForConditionalGeneration | None = None


def _load_model() -> Tuple[AutoProcessor, VoxtralRealtimeForConditionalGeneration]:
    global processor, model
    if processor is None or model is None:
        processor = AutoProcessor.from_pretrained(
            BASE_PROCESSOR_ID,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()
    return processor, model


def _load_audio(audio_path: str) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    return audio


def _to_base64_wav(audio: np.ndarray) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def transcribe(audio_path: str | None) -> str:
    if audio_path is None:
        return "Please record or upload audio first."

    proc, model_obj = _load_model()
    audio = _load_audio(audio_path)
    audio_b64 = _to_base64_wav(audio)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]

    tokenized = proc.tokenizer.apply_chat_template(
        messages,
        return_tensors=None,
        return_dict=True,
    )

    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(model_obj.device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(model_obj.device)

    raw_audio = tokenized.get("audio", [audio])
    input_features = proc.feature_extractor(
        raw_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )["input_features"].to(model_obj.device, dtype=model_obj.dtype)

    num_delay = getattr(proc.feature_extractor, "num_delay_tokens", 0)

    with torch.no_grad():
        output_ids = model_obj.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            num_delay_tokens=torch.tensor([num_delay], device=model_obj.device),
            max_new_tokens=256,
            do_sample=False,
        )

    return proc.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


with gr.Blocks(title="Voxtral Sentinel Transcriber") as demo:
    gr.Markdown(
        """
        # Voxtral Sentinel 4B Voice Transcription
        Record audio from your microphone (or upload a file) and get a transcript.

        **Target stack:** `gradio==6.8.0` + `trishtan/voxtral-sentinel-4b`
        """
    )

    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice recording")
    transcribe_btn = gr.Button("Transcribe", variant="primary")
    transcript_output = gr.Textbox(label="Transcript", lines=10)

    transcribe_btn.click(fn=transcribe, inputs=[audio_input], outputs=[transcript_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
