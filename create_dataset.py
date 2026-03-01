import os
import json
import random
from dotenv import load_dotenv
from datasets import load_dataset, Audio
from mistralai import Mistral
from tqdm import tqdm

# 1. SETUP & AUTHENTICATION
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY is not set. Run: $env:MISTRAL_API_KEY='your-key-here'")
client = Mistral(api_key=MISTRAL_API_KEY)

# 2. LOAD DATASETS (Streaming mode to save time/space)
print("Streaming MELD (Emotions) and DCASE (Environment) datasets...")
meld_ds = load_dataset("ajyy/MELD_audio", split="train", streaming=True, trust_remote_code=True)
meld_ds = meld_ds.cast_column("audio", Audio(decode=False))
dcase_ds = load_dataset("agkphysics/AudioSet", split="train", streaming=True)

meld_iter  = iter(meld_ds)
dcase_iter = iter(dcase_ds)


def get_forensic_insight(transcript: str, emotion: str, scene_label: str) -> dict | None:
    """
    Calls Mistral to produce all three sections separately.
    Returns a dict with keys TRANSCRIPT, ANALYSIS, CONCLUSION,
    or None on failure.
    """
    prompt = (
        f"You are an expert forensic audio analyst.\n\n"
        f"A speaker says: \"{transcript}\"\n"
        f"Detected emotion: {emotion}\n"
        f"Background environment: {scene_label}\n\n"
        "Produce a forensic audio analysis with exactly three sections.\n"
        "Return ONLY a JSON object with these three keys and no other text:\n"
        "{\n"
        "  \"TRANSCRIPT\": \"<verbatim or cleaned transcript of what was said>\",\n"
        "  \"ANALYSIS\": \"<2-3 sentence expert analysis of vocal cues, emotion, and environment>\",\n"
        "  \"CONCLUSION\": \"<1 sentence conclusion about the speaker's safety or situational risk>\"\n"
        "}"
    )

    try:
        response = client.chat.complete(
            model="mistral-small-2506",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        import re
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if all(k in parsed for k in ("TRANSCRIPT", "ANALYSIS", "CONCLUSION")):
            return parsed
        print(f"  Warning: missing keys in response: {list(parsed.keys())}")
        return None
    except Exception as e:
        print(f"  Mistral call failed: {e}")
        return None


CANONICAL_TEMPLATE = (
    "### TRANSCRIPT:\n{transcript}\n\n"
    "### ANALYSIS:\n{analysis}\n\n"
    "### CONCLUSION:\n{conclusion}"
)

# 3. GENERATION LOOP
OUTPUT_FILE = "voxtral_forensic_train_large.jsonl"
NUM_SAMPLES  = 12500
skipped      = 0

print(f"Generating {NUM_SAMPLES} samples...")
with open(OUTPUT_FILE, "w") as f:
    for i in tqdm(range(NUM_SAMPLES)):
        try:
            m_sample = next(meld_iter)
            d_sample = next(dcase_iter)
        except StopIteration:
            print("Dataset iterator exhausted early.")
            break

        transcript  = m_sample.get("text", "Unknown speech")
        emotion     = m_sample.get("emotion", "neutral")
        scene       = d_sample["labels"][0] if d_sample.get("labels") else "ambient environment"
        audio_path  = m_sample["audio"]["path"]

        sections = get_forensic_insight(transcript, emotion, scene)
        if sections is None:
            skipped += 1
            continue

        # Store prompt and answer as separate fields — NOT concatenated.
        # The training script is responsible for formatting into chat turns.
        entry = {
            "audio":  audio_path,
            "prompt": "Analyze this recording for forensic indicators.",
            "answer": CANONICAL_TEMPLATE.format(
                transcript=sections["TRANSCRIPT"],
                analysis=sections["ANALYSIS"],
                conclusion=sections["CONCLUSION"],
            ),
        }

        f.write(json.dumps(entry) + "\n")

print(f"\n✅ Done. Written to {OUTPUT_FILE}")
print(f"   Skipped (Mistral failures): {skipped}")
print("Next step: run dataset_packer.py to upload to HuggingFace.")