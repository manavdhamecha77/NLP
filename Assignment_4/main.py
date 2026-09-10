import os

from unigram import UnigramModel
from bigram import BigramModel
from trigram import TrigramModel
from quadgram import QuadgramModel

from perplexity import (
    unigram_perplexity,
    bigram_perplexity,
    trigram_perplexity,
    quadgram_perplexity
)


def load_sentences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


os.makedirs("models", exist_ok=True)

train_file = "data/train.txt"
dev_file = "data/dev.txt"
test_file = "data/test.txt"

dev = load_sentences(dev_file)
test = load_sentences(test_file)

print("Training Unigram model...")
unigram = UnigramModel()
unigram.train(train_file)

print("Training Bigram model...")
bigram = BigramModel()
bigram.train(train_file)

print("Training Trigram model...")
trigram = TrigramModel()
trigram.train(train_file)

print("Training Quadgram model...")
quadgram = QuadgramModel()
quadgram.train(train_file)

print("Models trained.")

print("\nDevelopment perplexity:")

print(
    "Unigram:",
    unigram_perplexity(
        "models/unigram.db",
        dev
    )
)

print(
    "Bigram:",
    bigram_perplexity(
        "models/bigram.db",
        dev
    )
)

print(
    "Trigram:",
    trigram_perplexity(
        "models/trigram.db",
        dev
    )
)

print(
    "Quadgram:",
    quadgram_perplexity(
        "models/quadgram.db",
        dev
    )
)

print("\nTest perplexity:")

print(
    "Unigram:",
    unigram_perplexity(
        "models/unigram.db",
        test
    )
)

print(
    "Bigram:",
    bigram_perplexity(
        "models/bigram.db",
        test
    )
)

print(
    "Trigram:",
    trigram_perplexity(
        "models/trigram.db",
        test
    )
)

print(
    "Quadgram:",
    quadgram_perplexity(
        "models/quadgram.db",
        test
    )
)