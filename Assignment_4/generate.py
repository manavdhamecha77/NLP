import sqlite3
import random


def get_vocabulary(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    conn.close()

    return vocabulary


def generate_bigram(db_path, text, max_words=20):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    words = text.split()

    if not words:
        words = ["<s>"]

    for _ in range(max_words):
        previous = words[-1]

        cursor.execute(
            """
            SELECT w2, count
            FROM counts
            WHERE w1 = ?
            ORDER BY count DESC
            LIMIT 20
            """,
            (previous,)
        )

        candidates = cursor.fetchall()

        if not candidates:
            break

        next_word = random.choices(
            [word for word, count in candidates],
            weights=[count for word, count in candidates]
        )[0]

        if next_word == "</s>":
            break

        words.append(next_word)

    conn.close()

    return " ".join(words)


def generate_trigram(db_path, text, max_words=20):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    words = text.split()

    if len(words) == 0:
        words = ["<s>", "<s>"]
    elif len(words) == 1:
        words = ["<s>", words[0]]

    for _ in range(max_words):
        w1 = words[-2]
        w2 = words[-1]

        cursor.execute(
            """
            SELECT w3, count
            FROM counts
            WHERE w1 = ? AND w2 = ?
            ORDER BY count DESC
            LIMIT 20
            """,
            (w1, w2)
        )

        candidates = cursor.fetchall()

        if not candidates:
            break

        next_word = random.choices(
            [word for word, count in candidates],
            weights=[count for word, count in candidates]
        )[0]

        if next_word == "</s>":
            break

        words.append(next_word)

    conn.close()

    return " ".join(
        word for word in words
        if word != "<s>"
    )


def generate_quadgram(db_path, text, max_words=20):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    words = text.split()

    while len(words) < 3:
        words.insert(0, "<s>")

    for _ in range(max_words):
        w1 = words[-3]
        w2 = words[-2]
        w3 = words[-1]

        cursor.execute(
            """
            SELECT w4, count
            FROM counts
            WHERE w1 = ? AND w2 = ? AND w3 = ?
            ORDER BY count DESC
            LIMIT 20
            """,
            (w1, w2, w3)
        )

        candidates = cursor.fetchall()

        if not candidates:
            break

        next_word = random.choices(
            [word for word, count in candidates],
            weights=[count for word, count in candidates]
        )[0]

        if next_word == "</s>":
            break

        words.append(next_word)

    conn.close()

    return " ".join(
        word for word in words
        if word != "<s>"
    )


def main():
    print("N-gram Text Generator")
    print()

    model = input(
        "Choose model (bigram/trigram/quadgram): "
    ).strip().lower()

    text = input("Enter starting text: ").strip()

    try:
        max_words = int(
            input("Number of words to generate: ")
        )
    except ValueError:
        max_words = 20

    if model == "bigram":
        result = generate_bigram(
            "models/bigram.db",
            text,
            max_words
        )

    elif model == "trigram":
        result = generate_trigram(
            "models/trigram.db",
            text,
            max_words
        )

    elif model == "quadgram":
        result = generate_quadgram(
            "models/quadgram.db",
            text,
            max_words
        )

    else:
        print("Invalid model.")
        return

    print()
    print(result)


if __name__ == "__main__":
    main()