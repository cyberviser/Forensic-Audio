import json
import os
import re
import tarfile
from dotenv import load_dotenv
from datasets import Dataset, Audio, Features, Value, DatasetDict
from huggingface_hub import login, HfFileSystem
from tqdm import tqdm

load_dotenv()

HF_TOKEN       = os.getenv("HF_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set in your .env file.")
login(token=HF_TOKEN)

# 1. PATHS
INPUT_JSONL    = "voxtral_forensic_train_large.jsonl"
OUTPUT_HF_REPO = "trishtan/voxtral-forensic-ds"

# Canonical answer template — three clearly separated sections.
# NOTE: prompt and answer are stored as SEPARATE columns.
#       The training script will format them into chat turns using
#       apply_chat_template. They must NOT be concatenated here.
CANONICAL_TEMPLATE = (
    "### TRANSCRIPT:\n{transcript}\n\n"
    "### ANALYSIS:\n{analysis}\n\n"
    "### CONCLUSION:\n{conclusion}"
)

# Section header aliases
SECTION_ALIASES = {
    "TRANSCRIPT": [
        "TRANSCRIPT", "TRANSCRIPTION", "SPEECH TRANSCRIPT",
        "AUDIO TRANSCRIPT", "VERBATIM TRANSCRIPT",
    ],
    "ANALYSIS": [
        "ANALYSIS", "FORENSIC ANALYSIS", "EXPERT FORENSIC AUDIO ANALYSIS",
        "AUDIO ANALYSIS", "EXPERT ANALYSIS", "DETAILED ANALYSIS",
    ],
    "CONCLUSION": [
        "CONCLUSION", "CONCLUSIONS", "FINAL CONCLUSION", "SUMMARY",
        "FORENSIC CONCLUSION", "OVERALL CONCLUSION", "FINAL ASSESSMENT",
    ],
}

_heading_pattern = re.compile(
    r"#{1,4}\s*\*{0,2}\s*("
    + "|".join(
        alias
        for aliases in SECTION_ALIASES.values()
        for alias in aliases
    )
    + r")\s*:?\s*\*{0,2}",
    re.IGNORECASE,
)


def _canonical_section_name(raw_name: str) -> str:
    upper = raw_name.upper().strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if upper in aliases:
            return canonical
    return None


def clean_answer_regex(answer: str) -> dict | None:
    parts    = _heading_pattern.split(answer)
    sections = {}
    i = 1
    while i < len(parts) - 1:
        name    = _canonical_section_name(parts[i])
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if name:
            sections[name] = content
        i += 2
    if all(k in sections for k in ("TRANSCRIPT", "ANALYSIS", "CONCLUSION")):
        return sections
    return None


def clean_answer_mistral(answer: str) -> dict | None:
    if not MISTRAL_API_KEY:
        return None
    try:
        from mistralai import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
        prompt = (
            "Extract and reformat the following forensic audio analysis into exactly "
            "three sections.\nReturn ONLY a JSON object with keys \"TRANSCRIPT\", "
            "\"ANALYSIS\", and \"CONCLUSION\".\nDo not add any explanation or markdown "
            "— just the raw JSON.\n\nInput:\n" + answer
        )
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw    = response.choices[0].message.content.strip()
        raw    = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if all(k in parsed for k in ("TRANSCRIPT", "ANALYSIS", "CONCLUSION")):
            return parsed
    except Exception as e:
        print(f"  Mistral fallback failed: {e}")
    return None


def normalise_answer(answer: str) -> str:
    answer   = re.sub(r"\*\*(#{1,4})", r"\1", answer)
    answer   = re.sub(r"(#{1,4})\*\*", r"\1", answer)
    sections = clean_answer_regex(answer) or clean_answer_mistral(answer)
    if sections is None:
        return answer
    for key in sections:
        text = sections[key]
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        sections[key] = text.strip()
    return CANONICAL_TEMPLATE.format(
        transcript=sections["TRANSCRIPT"],
        analysis=sections["ANALYSIS"],
        conclusion=sections["CONCLUSION"],
    )


# 2. LOAD ANNOTATIONS
print("Loading annotations from JSONL...")
annotations = {}
with open(INPUT_JSONL, "r") as f:
    for line in f:
        item = json.loads(line)
        annotations[item["audio"]] = {
            "prompt": item["prompt"],
            "answer": item["answer"],
        }

# 3. STREAM train.tar.gz from HF
print(f"Streaming train.tar.gz from HuggingFace to extract {len(annotations)} files...")

hffs = HfFileSystem(token=HF_TOKEN)
data = []
tar_path = "datasets/ajyy/MELD_audio/archive/train.tar.gz"

with hffs.open(tar_path, "rb") as remote_file:
    with tarfile.open(fileobj=remote_file, mode="r|gz") as tar:
        for member in tqdm(tar):
            name = member.name.lstrip("./")
            if not name.startswith("train/"):
                name = "train/" + os.path.basename(name)
            if name not in annotations:
                tar.members = []
                continue
            ann = annotations[name]
            f   = tar.extractfile(member)
            if f is None:
                continue
            audio_bytes = f.read()
            data.append({
                "audio":  {"bytes": audio_bytes, "path": name},
                "prompt": ann["prompt"],   # "Analyze this recording for forensic indicators."
                "answer": ann["answer"],   # "### TRANSCRIPT:\n...\n### ANALYSIS:\n...\n### CONCLUSION:\n..."
            })
            if len(data) == len(annotations):
                break

print(f"Matched {len(data)} / {len(annotations)} samples.")

# 4. CLEAN ANSWERS
print("Normalising answer formatting...")
failed = 0
for item in tqdm(data):
    original    = item["answer"]
    cleaned     = normalise_answer(original)
    if cleaned  == original:
        failed += 1
    item["answer"] = cleaned

print(f"Normalised: {len(data) - failed}/{len(data)} answers")
print(f"Unchanged (regex + Mistral both failed): {failed}")

if data:
    print("\n--- Sample entry (spot check) ---")
    print(f"prompt : {data[0]['prompt']}")
    print(f"answer :\n{data[0]['answer'][:400]}")
    print("---")
    # Verify the old 'text' concatenation is GONE
    assert "text" not in data[0], "ERROR: 'text' column should not exist — prompt/answer must stay separate."

# 5. BUILD DATASET
# Store prompt and answer as separate string columns.
# Audio is stored as raw bytes so the training script can decode it.
# Do NOT merge prompt+answer into a 'text' column here.
features = Features({
    "audio":  Audio(sampling_rate=16000),
    "prompt": Value("string"),
    "answer": Value("string"),
})

ds = Dataset.from_list(data, features=features)

# 6. TRAIN / TEST SPLIT
ds_split = ds.train_test_split(test_size=0.1, seed=42)
ds_dict  = DatasetDict({
    "train": ds_split["train"],
    "test":  ds_split["test"],
})

print(f"\nTrain: {len(ds_dict['train'])} | Test: {len(ds_dict['test'])}")

# 7. PUSH
print("Pushing to Hugging Face Hub...")
ds_dict.push_to_hub(OUTPUT_HF_REPO, private=True)

print(f"\n✅ Success! Dataset at: https://huggingface.co/datasets/{OUTPUT_HF_REPO}")
print("\nIMPORTANT — training script must format examples like this:")
print("""
messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": <base64_wav>, "format": "wav"}},
            {"type": "text", "text": example["prompt"]},
        ]
    },
    {
        "role": "assistant",
        "content": example["answer"]   # ### TRANSCRIPT / ANALYSIS / CONCLUSION
    }
]
tokenized = processor.tokenizer.apply_chat_template(messages, return_tensors=None, return_dict=True)
# Then pass tokenized["audio"] through processor.feature_extractor → input_features
# Mask user turn tokens in labels (loss only on assistant response)
""")