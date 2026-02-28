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
meld_ds = meld_ds.cast_column("audio", Audio(decode=False))  # Don't load audio from disk; we only need the path/metadata
dcase_ds = load_dataset("agkphysics/AudioSet", split="train", streaming=True) # Using AudioSet for Scene variety

# Create iterators
meld_iter = iter(meld_ds)
dcase_iter = iter(dcase_ds)

def get_forensic_insight(transcript, emotion, scene_label):
    """Calls Mistral Large to synthesize the 'New Capacity' reasoning."""
    prompt = (
        f"CONTEXT: A speaker says '{transcript}'. Their voice indicates {emotion}. "
        f"The background environment is identified as {scene_label}.\n\n"
        "TASK: Write a 1-sentence expert forensic audio analysis. "
        "Include a 'Conclusion' about the speaker's safety or situational risk."
    )
    
    try:
        response = client.chat.complete(
            model="mistral-small-2506",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis unavailable due to error: {e}"

# 3. GENERATION LOOP
OUTPUT_FILE = "voxtral_forensic_train_large.jsonl"
NUM_SAMPLES = 12500  # Perfect size for 4B fine-tuning in a short hackathon

print(f"Generating {NUM_SAMPLES} samples...")
with open(OUTPUT_FILE, "w") as f:
    for i in tqdm(range(NUM_SAMPLES)):
        try:
            # Get next samples from both datasets
            m_sample = next(meld_iter)
            d_sample = next(dcase_iter)
            
            # Extract metadata
            transcript = m_sample.get('text', 'Unknown speech')
            emotion = m_sample.get('emotion', 'neutral')
            # AudioSet labels are lists; take a relevant one or default to 'urban'
            scene = d_sample['labels'][0] if d_sample['labels'] else "ambient environment"
            
            # Generate the "Intelligence" label
            insight = get_forensic_insight(transcript, emotion, scene)
            
            # 4. VOXTRAL FORMATTING
            # 'audio' expects the local path or bytes. For HF training, path is best.
            entry = {
                "audio": m_sample['audio']['path'], 
                "prompt": "Analyze this recording for forensic indicators.",
                "answer": f"### TRANSCRIPT: {transcript} ### ANALYSIS: {insight}"
            }
            
            f.write(json.dumps(entry) + "\n")
            
        except StopIteration:
            break

print(f"\n✅ Success! Dataset created: {OUTPUT_FILE}")
print("Next Step: Upload this file to your A100 instance and start training.")