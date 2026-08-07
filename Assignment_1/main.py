import pandas as pd

from tokenizer import sentence_tokenizer, word_tokenizer
from statistics import corpus_statistics


def process_file(input_file, text_file, parquet_file):

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split Sentences
    sentences = sentence_tokenizer(text)

    # Split Words
    tokenized = []

    for sentence in sentences:
        tokenized.append(word_tokenizer(sentence))

    # Save tokens 
    with open(text_file, "w", encoding="utf-8") as f:
        for sentence in tokenized:
            f.write(" ".join(sentence) + "\n")

    # Parquet Files
    data = []

    for sentence in tokenized:
        data.append(" ".join(sentence))

    df = pd.DataFrame({"sentence": data})

    df.to_parquet(parquet_file)

    # Calculate statistics
    return corpus_statistics(tokenized)


stats = process_file(
    "data/gujarati.txt",
    "output/indic_tokens.txt",
    "output/indic_tokens.parquet"
)

print(stats)