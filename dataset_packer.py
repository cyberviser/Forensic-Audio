import json
import os
import re
import tarfile
from dotenv import load_dotenv
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login, HfFileSystem
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # optional — only used for fallback
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set in your .env file.")
login(token=HF_TOKEN)

# 1. PATHS
INPUT_JSONL    = "voxtral_forensic_train_large.jsonl"
OUTPUT_HF_REPO = "trishtan/voxtral-forensic-ds"

# Target format — every answer will be normalised to exactly this structure:
#
# ### TRANSCRIPT:
# <transcript text>
#
# ### ANALYSIS:
# <analysis text>
#
# ### CONCLUSION:
# <conclusion text>

CANONICAL_TEMPLATE = "### TRANSCRIPT:\n{transcript}\n\n### ANALYSIS:\n{analysis}\n\n### CONCLUSION:\n{conclusion}"

# Section header aliases — any of these will be mapped to the canonical name
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

# Build a flat regex that matches any heading variant
# Matches: ##, ###, #### followed by optional ** and the section name
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
    """Map any alias to TRANSCRIPT / ANALYSIS / CONCLUSION."""
    upper = raw_name.upper().strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if upper in aliases:
            return canonical
    return None


def clean_answer_regex(answer: str) -> dict | None:
    """
    Extract transcript/analysis/conclusion from a raw answer using regex.
    Returns a dict with keys TRANSCRIPT, ANALYSIS, CONCLUSION, or None if
    the structure cannot be determined.
    """
    # Split on any heading that matches our known aliases
    parts = _heading_pattern.split(answer)
    # parts alternates between: [preamble, section_name, content, section_name, content, ...]
    sections = {}
    i = 1
    while i < len(parts) - 1:
        name = _canonical_section_name(parts[i])
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if name:
            sections[name] = content
        i += 2

    if all(k in sections for k in ("TRANSCRIPT", "ANALYSIS", "CONCLUSION")):
        return sections
    return None


def clean_answer_mistral(answer: str) -> dict | None:
    """
    Fallback: use Mistral API to extract and reformat the answer.
    Only called when regex parsing fails.
    """
    if not MISTRAL_API_KEY:
        return None

    try:
        from mistralai import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)

        prompt = f"""Extract and reformat the following forensic audio analysis into exactly three sections.
Return ONLY a JSON object with keys "TRANSCRIPT", "ANALYSIS", and "CONCLUSION".
Do not add any explanation or markdown — just the raw JSON.

Input:
{answer}"""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if all(k in parsed for k in ("TRANSCRIPT", "ANALYSIS", "CONCLUSION")):
            return parsed
    except Exception as e:
        print(f"  Mistral fallback failed: {e}")
    return None


def normalise_answer(answer: str) -> str:
    """
    Clean and standardise an answer into the canonical format.
    Tries regex first (fast, free), falls back to Mistral if needed.
    If both fail, returns the original answer unchanged.
    """
    # Remove stray bold markers around section headers that slip through
    answer = re.sub(r"\*\*(#{1,4})", r"\1", answer)
    answer = re.sub(r"(#{1,4})\*\*", r"\1", answer)

    sections = clean_answer_regex(answer)

    if sections is None:
        sections = clean_answer_mistral(answer)

    if sections is None:
        # Both methods failed — return original so we don't silently drop data
        return answer

    # Clean up each section's content
    for key in sections:
        text = sections[key]
        # Remove any residual markdown bold/italic
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        # Collapse multiple blank lines to one
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
            f = tar.extractfile(member)
            if f is None:
                continue
            audio_bytes = f.read()
            data.append({
                "audio":  {"bytes": audio_bytes, "path": name},
                "prompt": ann["prompt"],
                "answer": ann["answer"],  # raw — cleaned below
            })
            if len(data) == len(annotations):
                break

print(f"Matched {len(data)} / {len(annotations)} samples.")

# 4. CLEAN ANSWERS
print("Normalising answer formatting...")
failed = 0
for item in tqdm(data):
    original = item["answer"]
    cleaned  = normalise_answer(original)
    if cleaned == original:
        failed += 1
    item["answer"] = cleaned

print(f"Normalised: {len(data) - failed}/{len(data)} answers")
print(f"Unchanged (regex + Mistral both failed): {failed}")

# Spot-check one example
if data:
    print("\n--- Sample cleaned answer ---")
    print(data[0]["answer"][:500])
    print("---")

# 5. BUILD DATASET
features = Features({
    "audio":  Audio(sampling_rate=16000),
    "prompt": Value("string"),
    "answer": Value("string"),
})

ds = Dataset.from_list(data, features=features)

# 6. PUSH
print("Pushing to Hugging Face Hub...")
ds.push_to_hub(OUTPUT_HF_REPO, private=True)

print(f"\n✅ Success! Dataset at: https://huggingface.co/datasets/{OUTPUT_HF_REPO}")