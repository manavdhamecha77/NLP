from perplexity import perplexity


def load_sentences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


dev = load_sentences("../Assignment_4/data/dev.txt")
test = load_sentences("../Assignment_4/data/test.txt")

models = {
    "Unigram": ("../Assignment_4/models/unigram.db", 1),
    "Bigram": ("../Assignment_4/models/bigram.db", 2),
    "Trigram": ("../Assignment_4/models/trigram.db", 3),
    "Quadgram": ("../Assignment_4/models/quadgram.db", 4)
}

print("Add-K smoothing: K = 0.3")

print("\nDevelopment perplexity:")

for name, (path, n) in models.items():
    print(
        f"{name}:",
        perplexity(path, dev, n)
    )

print("\nTest perplexity:")

for name, (path, n) in models.items():
    print(
        f"{name}:",
        perplexity(path, test, n)
    )