import json
import os
import io
import tarfile
from dotenv import load_dotenv
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login, HfFileSystem
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set in your .env file.")
login(token=HF_TOKEN)

# 1. PATHS
INPUT_JSONL = "voxtral_forensic_train_large.jsonl"
OUTPUT_HF_REPO = "trishtan/voxtral-forensic-ds"

# 2. LOAD ANNOTATIONS from JSONL into a dict keyed by audio path
print("Loading annotations from JSONL...")
annotations = {}
with open(INPUT_JSONL, "r") as f:
    for line in f:
        item = json.loads(line)
        annotations[item["audio"]] = {   # key = e.g. "train/dia0_utt0.flac"
            "prompt": item["prompt"],
            "answer": item["answer"],
        }

# 3. STREAM train.tar.gz from HF and extract only the audio files we need
print(f"Streaming train.tar.gz from HuggingFace to extract {len(annotations)} files...")

hffs = HfFileSystem(token=HF_TOKEN)

data = []
tar_path = "datasets/ajyy/MELD_audio/archive/train.tar.gz"

with hffs.open(tar_path, "rb") as remote_file:
    with tarfile.open(fileobj=remote_file, mode="r|gz") as tar:  # r| = streaming mode
        for member in tqdm(tar):
            # member.name might be e.g. "train/dia0_utt0.flac" or "dia0_utt0.flac"
            # normalise to match our annotations keys
            name = member.name.lstrip("./")
            if not name.startswith("train/"):
                name = "train/" + os.path.basename(name)

            if name not in annotations:
                tar.members = []  # free memory
                continue

            ann = annotations[name]
            f = tar.extractfile(member)
            if f is None:
                continue
            audio_bytes = f.read()
            data.append({
                "audio": {"bytes": audio_bytes, "path": name},
                "prompt": ann["prompt"],
                "answer": ann["answer"],
            })
            if len(data) == len(annotations):
                break  # extracted everything we need

print(f"Matched {len(data)} / {len(annotations)} samples.")

# 4. DEFINE SCHEMA & BUILD DATASET
features = Features({
    "audio": Audio(sampling_rate=16000),
    "prompt": Value("string"),
    "answer": Value("string"),
})

ds = Dataset.from_list(data, features=features)

# 5. PUSH — audio bytes are baked in, no local file dependency
print("Pushing to Hugging Face Hub...")
ds.push_to_hub(OUTPUT_HF_REPO, private=True)

print(f"\n✅ Success! Dataset at: https://huggingface.co/datasets/{OUTPUT_HF_REPO}")