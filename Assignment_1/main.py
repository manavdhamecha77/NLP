import pandas as pd

from tokenizer import sentence_tokenizer
from tokenizer import word_tokenizer

from statistics import corpus_statistics


def process_file(
    input_path,
    txt_output,
    parquet_output
):

    tokenized_sentences = []

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    sentences = sentence_tokenizer(text)

    for sentence in sentences:

        tokens = word_tokenizer(sentence)

        tokenized_sentences.append(tokens)

    with open(
        txt_output,
        "w",
        encoding="utf-8"
    ) as file:

        for sentence in tokenized_sentences:

            line = " ".join(sentence)

            file.write(line + "\n")

    df = pd.DataFrame(
        {
            "sentence": [
                " ".join(sentence)
                for sentence in tokenized_sentences
            ]
        }
    )

    df.to_parquet(parquet_output)

    stats = corpus_statistics(
        tokenized_sentences
    )

    return stats


indic_stats = process_file(
    "data/gujarati.txt",
    "output/indic_tokens.txt",
    "output/indic_tokens.parquet"
)

# oscar_stats = process_file(
#     "data/oscar.txt",
#     "output/oscar_tokens.txt",
#     "output/oscar_tokens.parquet"
# )

print("\nIndicCorp Statistics\n")

for key, value in indic_stats.items():
    print(key, ":", value)

# print("\nOSCAR Statistics\n")

# for key, value in oscar_stats.items():
#     print(key, ":", value)