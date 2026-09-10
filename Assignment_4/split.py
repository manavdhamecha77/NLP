import random

INPUT_FILE = "data/indic_tokens.txt"

random.seed(42)


def split_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    random.shuffle(sentences)

    test = sentences[:1000]
    dev = sentences[1000:2000]
    train = sentences[2000:]

    with open("data/train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train))

    with open("data/dev.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dev))

    with open("data/test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test))

    print("Total:", len(sentences))
    print("Training:", len(train))
    print("Development:", len(dev))
    print("Test:", len(test))


if __name__ == "__main__":
    split_data()