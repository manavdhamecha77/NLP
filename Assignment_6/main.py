from perplexity import calculate, load_all_models, load_vocab

DEV_PATH = "../Assignment_4/data/dev.txt"
TEST_PATH = "../Assignment_4/data/test.txt"

TECHNIQUES = [
    ("interpolated", "INTERPOLATED SMOOTHING"),
    ("good_turing", "GOOD-TURING SMOOTHING"),
    ("katz", "KATZ BACKOFF SMOOTHING"),
    ("stupid_backoff", "STUPID BACKOFF SMOOTHING"),
    ("kneser_ney", "KNESER-NEY SMOOTHING"),
]

MODELS = [(2, "Bigram"), (3, "Trigram"), (4, "Quadgram")]

def main():
    models = load_all_models()
    vocab = load_vocab()
    results = {}

    print("=" * 80)
    print("NLP ASSIGNMENT 5 - SMOOTHING TECHNIQUES")
    print("=" * 80)

    for technique, title in TECHNIQUES:
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
        results[technique] = {}

        for order, name in MODELS:
            dev = calculate(technique, order, DEV_PATH, models, vocab)
            test = calculate(technique, order, TEST_PATH, models, vocab)
            results[technique][name] = (dev, test)

            print(f"\n{name}")
            print(f"Development Perplexity: {dev:.4f}")
            print(f"Test Perplexity:        {test:.4f}")

    print("\n\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Technique':<24}{'Model':<12}{'Dev':>15}{'Test':>15}")
    print("-" * 80)

    names = {
        "interpolated": "Interpolated",
        "good_turing": "Good-Turing",
        "katz": "Katz Backoff",
        "stupid_backoff": "Stupid Backoff",
        "kneser_ney": "Kneser-Ney",
    }

    for technique, _ in TECHNIQUES:
        for _, name in MODELS:
            dev, test = results[technique][name]
            print(f"{names[technique]:<24}{name:<12}{dev:>15.4f}{test:>15.4f}")

if __name__ == "__main__":
    main()
