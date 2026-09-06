from datasets import load_dataset
from tqdm import tqdm
import os

LANGUAGE_SPLIT = "guj_Gujr"
TARGET_LINES = 100_000

OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    split=LANGUAGE_SPLIT,
    streaming=True,
    trust_remote_code=True,
)

sentences = []

with tqdm(total=TARGET_LINES) as pbar:
    for sample in dataset:

        text = sample.get("text", "").strip()

        if text:
            sentences.append(text)
            pbar.update(1)

        if len(sentences) >= TARGET_LINES:
            break

output_file = os.path.join(
    OUTPUT_DIR,
    "gujarati.txt"
)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))

print(f"\nSaved {len(sentences)} lines.")
print(f"Location: {output_file}")