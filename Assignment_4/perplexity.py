import math
import pickle
from tqdm import tqdm


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_vocab():
    with open("models/vocab.pkl", "rb") as f:
        return pickle.load(f)


def encode_sentence(line, vocab):
    ids = [vocab["<s>"]]

    for word in line.split():
        ids.append(vocab.get(word, vocab["<UNK>"]))

    ids.append(vocab["</s>"])

    return ids


def pack(ids, bits):
    key = 0

    for x in ids:
        key = (key << bits) | x

    return key


def perplexity(model_name, data_path, lower_model_name=None):

    model = load_model(
        f"models/{model_name}.pkl"
    )

    lower_model = None

    if lower_model_name:
        lower_model = load_model(
            f"models/{lower_model_name}.pkl"
        )

    vocab = load_vocab()

    counts = model["counts"]
    V = model["vocab_size"]
    bits = model["bits"]
    n = model["n"] if "n" in model else {
        "unigram": 1,
        "bigram": 2,
        "trigram": 3,
        "quadgram": 4
    }[model_name]

    log_probability = 0.0
    N = 0

    with open(data_path, "r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            total=1000,
            desc=model_name,
            unit="sent"
        ):

            ids = encode_sentence(line, vocab)

            if n == 1:

                total = model["total"]

                for word_id in ids[1:]:

                    count = counts.get(
                        word_id,
                        0
                    )

                    probability = (
                        count + 1
                    ) / (
                        total + V
                    )

                    log_probability += math.log(
                        probability
                    )

                    N += 1

            else:

                lower_counts = lower_model["counts"]

                for i in range(len(ids) - n + 1):

                    ngram = ids[i:i + n]
                    context = ids[i:i + n - 1]

                    ngram_key = pack(
                        ngram,
                        bits
                    )

                    context_key = pack(
                        context,
                        bits
                    )

                    ngram_count = counts.get(
                        ngram_key,
                        0
                    )

                    context_count = lower_counts.get(
                        context_key,
                        0
                    )

                    probability = (
                        ngram_count + 1
                    ) / (
                        context_count + V
                    )

                    log_probability += math.log(
                        probability
                    )

                    N += 1

    return math.exp(
        -log_probability / N
    )


def main():

    models = [
        ("unigram", None),
        ("bigram", "unigram"),
        ("trigram", "bigram"),
        ("quadgram", "trigram")
    ]

    print("LAPLACE SMOOTHING - PERPLEXITY")

    for data_name, data_path in [
        ("Development", "data/dev.txt"),
        ("Test", "data/test.txt")
    ]:

        print(f"\n{data_name}")
        print("-" * 60)

        for model, lower_model in models:

            score = perplexity(
                model,
                data_path,
                lower_model
            )

            print(
                f"{model.capitalize():10s}: "
                f"{score:.4f}"
            )


if __name__ == "__main__":
    main()