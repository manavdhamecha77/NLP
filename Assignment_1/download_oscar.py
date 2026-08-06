from datasets import load_dataset
from tqdm import tqdm
import os

TARGET_LINES = 1000

os.makedirs("data", exist_ok=True)

print("Loading OSCAR dataset...")

dataset = load_dataset(
    "oscar-corpus/OSCAR-2301",
    split="train",
    streaming=True
)

sentences = []

print("Collecting sentences...")

with tqdm(total=TARGET_LINES) as pbar:
    for sample in dataset:

        text = sample.get("text", "").strip()

        if text:
            sentences.append(text)
            pbar.update(1)

        if len(sentences) >= TARGET_LINES:
            break

with open(
    "data/oscar.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(sentences))

print(f"\nSaved {len(sentences)} lines.")
print("Output file: data/oscar.txt")