from perplexity import calculate


def main():

    models = [
        ("unigram", None),
        ("bigram", "unigram"),
        ("trigram", "bigram"),
        ("quadgram", "trigram")
    ]

    print("ADD-K SMOOTHING")
    print("K = 0.3")

    for name, lower in models:

        print(f"\n{name.capitalize()}")

        dev = calculate(
            name,
            "../Assignment_4/data/dev.txt",
            lower
        )

        test = calculate(
            name,
            "../Assignment_4/data/test.txt",
            lower
        )

        print(f"Development Perplexity: {dev:.4f}")
        print(f"Test Perplexity:        {test:.4f}")


if __name__ == "__main__":
    main()