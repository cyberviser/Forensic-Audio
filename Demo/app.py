"""
Voxtral Sentinel — Demo App
A side-by-side demonstration of how fine-tuned audio understanding
enriches autonomous agent responses for customer service and emergency services.

Requirements:
    pip install gradio mistralai transformers torch soundfile librosa numpy accelerate

Environment variables:
    HF_TOKEN        — HuggingFace token (for fine-tuned model)
    MISTRAL_API_KEY — Mistral API key (for conversational agent)

HOW TO ADD YOUR OWN EXAMPLE CLIPS:
    1. Record your audio clips (WAV or MP3, 5-15 seconds recommended)
    2. Upload them to this Space via Files tab -> Add file -> Upload files
    3. Update the EXAMPLES list below with the filenames and labels
    4. Restart the Space
"""

import os
import re
import numpy as np
import soundfile as sf
import librosa
import torch
import gradio as gr
from mistralai import Mistral

# Ensure OMP_NUM_THREADS is set to a valid integer to avoid libgomp warnings
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN        = os.getenv("HF_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SAMPLE_RATE     = 16000
MEL_FRAMES      = 1500
HOP_LENGTH      = 160
AUDIO_MAX       = MEL_FRAMES * HOP_LENGTH  # 240000 = 15 seconds

BASE_MODEL      = "mistralai/Voxtral-Mini-4B-Realtime-2602"
FINETUNED_MODEL = "trishtan/voxtral-sentinel-4b"
VOXTRAL_PROMPT  = "Analyze this recording for forensic indicators."

if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY is not set. Add it as a Secret in Space settings.")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set. Add it as a Secret in Space settings.")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ---------------------------------------------------------------------------
# Example clips
# ---------------------------------------------------------------------------
# HOW TO ADD YOUR CLIPS:
#   - Record using your phone voice memo or any microphone app
#   - Export as .wav or .mp3
#   - Upload to this Space (Files tab -> Add file -> Upload files)
#   - Add entries below as: ["filename.wav", "Label shown in UI"]
#
# Suggested scripts to record:
#
# frustrated_customer.wav:
#   "Hi, I placed my order over an hour ago and it still hasn't arrived.
#    I've been waiting and nobody is picking up the phone.
#    This is completely unacceptable, I want a refund."
#
# happy_customer.wav:
#   "Hey just calling to say my order arrived super fast today,
#    everything was perfect and the food was delicious.
#    Really impressed, thank you!"
#
# emergency_distressed.wav:
#   "Please help, I think my neighbour has fallen,
#    I can hear them calling out but they're not answering the door.
#    I don't know what to do, can you send someone?"
#
# non_urgent_report.wav:
#   "Hi yes, I'd like to report a broken street light
#    on the corner of Main and 5th.
#    It's been out for a few days, just wanted to let someone know."

EXAMPLES = [
    ["frustrated_customer.wav",  "😤 Frustrated customer (food delivery)"],
    ["happy_customer.wav",       "😊 Happy customer (food delivery)"],
    ["emergency_distressed.wav", "🆘 Distressed caller (emergency)"],
    ["non_urgent_report.wav",    "📋 Non-urgent report (emergency)"],
]

# Filter to only show examples where the file actually exists
EXAMPLES = [e for e in EXAMPLES if os.path.exists(e[0])]

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
PERSONAS = {
    "food_delivery": {
        "name": "Maya — Food Delivery Support",
        "emoji": "🍕",
        "base_system": """You are Maya, a warm and efficient customer service agent for SwiftEats food delivery.
You have received a voice message from a customer.

TRANSCRIPT:
{transcript}

Respond naturally as Maya. Be empathetic, solution-focused, and concise.
Do not mention that you received a transcript — speak as if you heard them directly.""",

        "enriched_system": """You are Maya, a warm and efficient customer service agent for SwiftEats food delivery.
You have received a voice message from a customer. Your advanced audio system has analysed it:

TRANSCRIPT:
{transcript}

AUDIO ANALYSIS:
{emotion}

CONCLUSION:
{situation}

Use this analysis to calibrate your response — if frustration or urgency is detected, be extra
apologetic and proactive. If the customer is happy, match their energy. Do not reveal you have
this analysis — speak as if you naturally understood their situation.""",
    },

    "emergency": {
        "name": "Delta-1 — Emergency Response",
        "emoji": "🚨",
        "base_system": """You are Delta-1, a calm and professional emergency response operator.
You have received an audio message.

TRANSCRIPT:
{transcript}

Respond as a trained emergency operator. Be clear, calm, and gather essential information.
Do not mention that you received a transcript.""",

        "enriched_system": """You are Delta-1, a calm and professional emergency response operator.
You have received an audio message. Your advanced audio system has provided this assessment:

TRANSCRIPT:
{transcript}

AUDIO ANALYSIS:
{emotion}

CONCLUSION:
{situation}

Use this assessment to prioritise your response. If the conclusion indicates high urgency or
distress, act immediately — dispatch while keeping the caller calm. For non-urgent matters,
follow standard intake protocol. Do not reveal you have this analysis.""",
    },
}

# ---------------------------------------------------------------------------
# Model loading (lazy — loaded on first use)
# ---------------------------------------------------------------------------
_models = {}

def _load_voxtral(model_id: str):
    if model_id not in _models:
        print(f"Loading {model_id}...")
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
        # Always load the processor from the base model — fine-tuned checkpoints
        # typically don't republish preprocessor_config.json / tokenizer files.
        processor_source = BASE_MODEL if model_id == FINETUNED_MODEL else model_id
        processor = AutoProcessor.from_pretrained(
            processor_source, trust_remote_code=True, token=HF_TOKEN,
        )
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.use_cache = True
        model.eval()
        _models[model_id] = (processor, model)
        print(f"Loaded {model_id}")
    return _models[model_id]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n < AUDIO_MAX:
        audio = np.pad(audio, (0, AUDIO_MAX - n))
    elif n > AUDIO_MAX:
        audio = audio[:AUDIO_MAX]
    return audio


def _load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return _pad_or_trim(audio)


# ---------------------------------------------------------------------------
# Audio → base64 WAV helper (required by apply_chat_template)
# ---------------------------------------------------------------------------
def _array_to_base64_wav(audio: np.ndarray) -> str:
    import io as _io, base64 as _b64
    buf = _io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return _b64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Voxtral inference
# ---------------------------------------------------------------------------
def _run_voxtral(audio_path: str, model_id: str) -> dict:
    processor, model = _load_voxtral(model_id)
    audio    = _load_audio(audio_path)
    audio_b64 = _array_to_base64_wav(audio)

    # Build the chat message with interleaved audio + text prompt.
    # This is the only inference path that works correctly — passing
    # audio and text separately (old approach) misaligns the model inputs.
    # For the base model, omit the text prompt — it was not trained on it
    # and adding it causes noise. For the fine-tuned model, include it.
    is_finetuned = model_id == FINETUNED_MODEL
    content = [
        {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        },
    ]
    if is_finetuned:
        content.append({"type": "text", "text": VOXTRAL_PROMPT})

    messages = [{"role": "user", "content": content}]

    # apply_chat_template returns plain lists when audio is present
    tokenized = processor.tokenizer.apply_chat_template(
        messages,
        return_tensors=None,
        return_dict=True,
    )

    input_ids      = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(model.device)

    # Process audio through feature_extractor separately
    raw_audio      = tokenized.get("audio", [audio])
    input_features = processor.feature_extractor(
        raw_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )["input_features"].to(model.device, dtype=model.dtype)

    # Include num_delay_tokens — required by the model's forward pass
    num_delay = getattr(processor.feature_extractor, "num_delay_tokens", 0)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            num_delay_tokens=torch.tensor([num_delay], device=model.device),
            max_new_tokens=512,
            do_sample=False,
        )

    raw = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _extract(text, header):
        pattern = r"#{1,4}\s*(?:" + re.escape(header) + r")\s*:?\s*\n(.*?)(?=#{1,4}|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    # Fine-tuned model outputs TRANSCRIPT / ANALYSIS / CONCLUSION.
    # Map these to agent context: analysis -> emotion context, conclusion -> situation summary.
    transcript = _extract(raw, "TRANSCRIPT") or raw
    analysis   = _extract(raw, "ANALYSIS")
    conclusion = _extract(raw, "CONCLUSION")

    return {
        "raw":        raw,
        "transcript": transcript,
        "emotion":    analysis,    # maps to AUDIO ANALYSIS in agent system prompt
        "situation":  conclusion,  # maps to CONCLUSION in agent system prompt
        "action":     "",          # not produced by this model
    }


# ---------------------------------------------------------------------------
# Mistral conversation
# ---------------------------------------------------------------------------
def _run_mistral(system_prompt: str, user_message: str, history: list) -> str:
    # history is a list of {"role": ..., "content": ...} dicts (Gradio messages format)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_audio(audio_path, model_choice, persona_choice, chat_history):
    if audio_path is None:
        yield chat_history, "⚠️ No audio provided — upload or record a clip first.", {}
        return

    persona       = PERSONAS[persona_choice]
    use_finetuned = model_choice == "Fine-tuned (Voxtral Sentinel)"
    model_id      = FINETUNED_MODEL if use_finetuned else BASE_MODEL

    yield chat_history, "🎙️ Analysing audio...", {}

    result = _run_voxtral(audio_path, model_id)

    if use_finetuned:
        system = persona["enriched_system"].format(
            transcript = result["transcript"],
            emotion    = result["emotion"]   or "Not detected",
            situation  = result["situation"] or "Not detected",
        )
        analysis_display = result["raw"]
    else:
        system = persona["base_system"].format(transcript=result["transcript"])
        analysis_display = f"### TRANSCRIPT:\n{result['transcript']}"

    agent_response = _run_mistral(
        system_prompt = system,
        user_message  = "[Audio message received — please respond to the caller]",
        history       = chat_history,
    )

    new_history = chat_history + [
        {"role": "user",      "content": "🎤 [Audio message]"},
        {"role": "assistant", "content": agent_response},
    ]
    yield new_history, analysis_display, result  # store full result dict for follow-up context


def continue_chat(user_message, chat_history, model_choice, persona_choice, last_analysis):
    if not user_message.strip():
        return chat_history, ""

    persona       = PERSONAS[persona_choice]
    use_finetuned = model_choice == "Fine-tuned (Voxtral Sentinel)"

    # last_analysis is the full result dict from _run_voxtral
    result = last_analysis if isinstance(last_analysis, dict) else {}

    system = (
        persona["enriched_system"].format(
            transcript = result.get("transcript", ""),
            emotion    = result.get("emotion",    "Not detected"),
            situation  = result.get("situation",  "Not detected"),
        )
        if use_finetuned
        else persona["base_system"].format(transcript=result.get("transcript", ""))
    )

    response    = _run_mistral(system, user_message, chat_history)
    new_history = chat_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return new_history, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0A0A0F;
    --surface: #12121A;
    --border:  #1E1E2E;
    --accent:  #00FF94;
    --accent2: #FF3CAC;
    --text:    #E8E8F0;
    --muted:   #6B6B8A;
    --radius:  12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--text) !important;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.app-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, #00B4FF 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
}

.app-header p { color: var(--muted); font-size: 0.9rem; margin: 0; }

.tag-base {
    display: inline-block; background: #1A1A2E; color: #6B6B8A;
    border: 1px solid #2A2A3E; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; font-family: 'DM Mono', monospace;
}

.tag-ft {
    display: inline-block; background: #001A0F; color: var(--accent);
    border: 1px solid #00331A; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; font-family: 'DM Mono', monospace;
}

button.primary { background: var(--accent) !important; color: #000 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important; }
button.primary:hover { opacity: 0.85 !important; }

.analysis-box textarea {
    font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important;
    background: #0D0D15 !important; color: var(--text) !important;
    border-color: var(--border) !important;
}

footer { display: none !important; }
"""

with gr.Blocks(title="Voxtral Sentinel", css=CSS) as demo:

    last_analysis = gr.State({})

    gr.HTML("""
    <div class="app-header">
        <h1>VOXTRAL SENTINEL</h1>
        <p>Fine-tuned audio intelligence for autonomous customer support &amp; emergency response</p>
    </div>
    """)

    with gr.Row():

        # Left column — controls + analysis
        with gr.Column(scale=1):
            gr.HTML("<h3 style='font-family:Syne,sans-serif;margin-bottom:1rem'>⚙️ Configuration</h3>")

            model_choice = gr.Radio(
                choices=["Base Model (Voxtral)", "Fine-tuned (Voxtral Sentinel)"],
                value="Fine-tuned (Voxtral Sentinel)",
                label="Audio Analysis Model",
            )

            gr.HTML("""
            <div style='font-size:0.75rem;color:#6B6B8A;margin:-0.5rem 0 1rem;padding:0.5rem;
                        background:#0D0D15;border-radius:6px;border:1px solid #1E1E2E'>
                <span class="tag-base">Base</span> Transcript only &rarr; generic agent response<br>
                <span class="tag-ft">Fine-tuned</span> Transcript + analysis + conclusion &rarr; contextual response
            </div>
            """)

            persona_choice = gr.Radio(
                choices=["food_delivery", "emergency"],
                value="food_delivery",
                label="Agent Persona",
                info="Select the deployment context",
            )

            gr.HTML("""
            <div style='font-size:0.75rem;color:#6B6B8A;margin:-0.5rem 0 1rem;padding:0.5rem;
                        background:#0D0D15;border-radius:6px;border:1px solid #1E1E2E'>
                🍕 <b>SwiftEats</b> — Food delivery customer support<br>
                🚨 <b>Delta-1</b> — Emergency response hotline
            </div>
            """)

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input (upload, record, or click an example below)",
            )

            analyse_btn = gr.Button("▶  Analyse & Respond", variant="primary")

            gr.HTML("<hr style='border-color:#1E1E2E;margin:1rem 0'>")
            gr.HTML("<h3 style='font-family:Syne,sans-serif;margin-bottom:0.5rem'>📊 Audio Analysis</h3>")

            analysis_out = gr.Textbox(
                label="Voxtral Output",
                lines=12,
                interactive=False,
                elem_classes=["analysis-box"],
                placeholder="Audio analysis will appear here after processing...",
            )

        # Right column — chat
        with gr.Column(scale=2):
            gr.HTML("<h3 style='font-family:Syne,sans-serif;margin-bottom:1rem'>💬 Agent Conversation</h3>")

            chatbot = gr.Chatbot(
                label="",
                height=480,
                show_label=False,
                type="messages",
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=sentinel&backgroundColor=00FF94",
                ),
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Continue the conversation...",
                    label="",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            gr.HTML("""
            <div style='font-size:0.75rem;color:#6B6B8A;margin-top:0.5rem;text-align:center'>
                Powered by <b>Voxtral Sentinel</b> (audio analysis) + <b>Mistral-small</b> (conversation)
            </div>
            """)

    # Example clips section — only renders if files exist in the Space
    if EXAMPLES:
        gr.HTML("<hr style='border-color:#1E1E2E;margin:1.5rem 0 1rem'>")
        gr.HTML("<h3 style='font-family:Syne,sans-serif;margin-bottom:0.25rem'>🎯 Try these examples</h3>")
        gr.HTML("<p style='color:#6B6B8A;font-size:0.8rem;margin-bottom:0.75rem'>Click any clip to load it into the audio player, then hit Analyse & Respond</p>")
        gr.Examples(
            examples=[[e[0]] for e in EXAMPLES],
            example_labels=[e[1] for e in EXAMPLES],
            inputs=[audio_input],
            examples_per_page=4,
        )
    else:
        gr.HTML("""
        <hr style='border-color:#1E1E2E;margin:1.5rem 0 1rem'>
        <div style='padding:1rem;background:#0D0D15;border-radius:8px;border:1px dashed #1E1E2E;
                    color:#6B6B8A;font-size:0.8rem;text-align:center'>
            📂 No example clips uploaded yet.<br>
            Upload <code>frustrated_customer.wav</code>, <code>happy_customer.wav</code>,
            <code>emergency_distressed.wav</code>, <code>non_urgent_report.wav</code>
            to the Space files to enable examples.
        </div>
        """)

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------
    analyse_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_choice, persona_choice, chatbot],
        outputs=[chatbot, analysis_out, last_analysis],
        show_progress=True,
    )

    send_btn.click(
        fn=continue_chat,
        inputs=[chat_input, chatbot, model_choice, persona_choice, last_analysis],
        outputs=[chatbot, chat_input],
    )

    chat_input.submit(
        fn=continue_chat,
        inputs=[chat_input, chatbot, model_choice, persona_choice, last_analysis],
        outputs=[chatbot, chat_input],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)