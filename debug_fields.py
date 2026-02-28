import json
from datasets import load_dataset, Audio
from dotenv import load_dotenv

load_dotenv()

print("Loading first MELD sample...")
meld_ds = load_dataset("ajyy/MELD_audio", split="train", streaming=True, trust_remote_code=True)
meld_ds = meld_ds.cast_column("audio", Audio(decode=False))

sample = next(iter(meld_ds))

print("\n--- Available keys ---")
print(list(sample.keys()))

print("\n--- Full sample (non-audio fields) ---")
for k, v in sample.items():
    if k != "audio":
        print(f"  {k!r}: {v!r}")
