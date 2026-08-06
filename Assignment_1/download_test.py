from datasets import load_dataset

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    split="guj_Gujr"
)

print(dataset)