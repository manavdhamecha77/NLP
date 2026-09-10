import pickle
import time
from collections import Counter
from tqdm import tqdm


BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"


def build_vocab(path):
    vocab = {UNK: 0, BOS: 1, EOS: 2}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                if word not in vocab:
                    vocab[word] = len(vocab)

    return vocab


def pack(a, b, bits):
    return (a << bits) | b


def pack3(a, b, c, bits):
    return (a << (2 * bits)) | (b << bits) | c


def pack4(a, b, c, d, bits):
    return (a << (3 * bits)) | (b << (2 * bits)) | (c << bits) | d


def train(train_path):

    print("Building vocabulary...")
    start = time.time()

    vocab = build_vocab(train_path)

    print(f"Vocabulary: {len(vocab):,}")
    print(f"Time: {time.time() - start:.2f}s")

    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    bits = max(1, (len(vocab) - 1).bit_length())

    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    quadgram = Counter()

    print("\nTraining all N-gram models...")

    start = time.time()

    with open(train_path, "r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            total=2442262,
            unit="sent",
            dynamic_ncols=True
        ):

            words = line.split()

            ids = [vocab.get(w, 0) for w in words]

            # Add EOS
            ids.append(vocab[EOS])

            # Unigram
            unigram.update(ids)

            # Bigram
            if len(ids) >= 2:
                bigram.update(
                    pack(a, b, bits)
                    for a, b in zip(
                        [vocab[BOS]] + ids[:-1],
                        ids
                    )
                )

            # Trigram
            if len(ids) >= 3:
                seq = [vocab[BOS]] + ids

                trigram.update(
                    pack3(a, b, c, bits)
                    for a, b, c in zip(
                        seq,
                        seq[1:],
                        seq[2:]
                    )
                )

            # Quadgram
            if len(ids) >= 4:
                seq = [vocab[BOS]] + ids

                quadgram.update(
                    pack4(a, b, c, d, bits)
                    for a, b, c, d in zip(
                        seq,
                        seq[1:],
                        seq[2:],
                        seq[3:]
                    )
                )

    print(f"\nTraining time: {(time.time() - start) / 60:.2f} minutes")

    print("\nSaving models...")

    models = {
        "unigram": {
            "counts": unigram,
            "total": sum(unigram.values()),
            "vocab_size": len(vocab) - 1,
            "bits": bits
        },

        "bigram": {
            "counts": bigram,
            "vocab_size": len(vocab) - 1,
            "bits": bits
        },

        "trigram": {
            "counts": trigram,
            "vocab_size": len(vocab) - 1,
            "bits": bits
        },

        "quadgram": {
            "counts": quadgram,
            "vocab_size": len(vocab) - 1,
            "bits": bits
        }
    }

    for name, model in models.items():

        with open(
            f"models/{name}.pkl",
            "wb"
        ) as f:

            pickle.dump(
                model,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print(
            f"{name}: "
            f"{len(model['counts']):,} unique n-grams"
        )

    print("\nDone.")


if __name__ == "__main__":
    train("data/train.txt")