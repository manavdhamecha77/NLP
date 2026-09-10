import math
import sqlite3


def unigram_perplexity(db_path, sentences):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT total, vocabulary FROM meta")
    total, vocabulary = cursor.fetchone()

    log_probability = 0
    count = 0

    for sentence in sentences:
        for word in sentence.split():
            cursor.execute(
                "SELECT count FROM counts WHERE word=?",
                (word,)
            )

            result = cursor.fetchone()
            word_count = result[0] if result else 0

            probability = (
                (word_count + 1)
                / (total + vocabulary)
            )

            log_probability += math.log(probability)
            count += 1

    conn.close()

    return math.exp(-log_probability / count)


def bigram_perplexity(db_path, sentences):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    log_probability = 0
    count = 0

    for sentence in sentences:
        words = ["<s>"] + sentence.split() + ["</s>"]

        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]

            cursor.execute(
                "SELECT count FROM counts WHERE w1=? AND w2=?",
                (w1, w2)
            )

            result = cursor.fetchone()
            ngram_count = result[0] if result else 0

            cursor.execute(
                "SELECT count FROM contexts WHERE word=?",
                (w1,)
            )

            result = cursor.fetchone()
            context_count = result[0] if result else 0

            probability = (
                (ngram_count + 1)
                / (context_count + vocabulary)
            )

            log_probability += math.log(probability)
            count += 1

    conn.close()

    return math.exp(-log_probability / count)


def trigram_perplexity(db_path, sentences):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    log_probability = 0
    count = 0

    for sentence in sentences:
        words = ["<s>", "<s>"] + sentence.split() + ["</s>"]

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i:i + 3]

            cursor.execute(
                """
                SELECT count
                FROM counts
                WHERE w1=? AND w2=? AND w3=?
                """,
                (w1, w2, w3)
            )

            result = cursor.fetchone()
            ngram_count = result[0] if result else 0

            cursor.execute(
                """
                SELECT count
                FROM contexts
                WHERE w1=? AND w2=?
                """,
                (w1, w2)
            )

            result = cursor.fetchone()
            context_count = result[0] if result else 0

            probability = (
                (ngram_count + 1)
                / (context_count + vocabulary)
            )

            log_probability += math.log(probability)
            count += 1

    conn.close()

    return math.exp(-log_probability / count)


def quadgram_perplexity(db_path, sentences):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    log_probability = 0
    count = 0

    for sentence in sentences:
        words = (
            ["<s>", "<s>", "<s>"]
            + sentence.split()
            + ["</s>"]
        )

        for i in range(len(words) - 3):
            w1, w2, w3, w4 = words[i:i + 4]

            cursor.execute(
                """
                SELECT count
                FROM counts
                WHERE w1=? AND w2=? AND w3=? AND w4=?
                """,
                (w1, w2, w3, w4)
            )

            result = cursor.fetchone()
            ngram_count = result[0] if result else 0

            cursor.execute(
                """
                SELECT count
                FROM contexts
                WHERE w1=? AND w2=? AND w3=?
                """,
                (w1, w2, w3)
            )

            result = cursor.fetchone()
            context_count = result[0] if result else 0

            probability = (
                (ngram_count + 1)
                / (context_count + vocabulary)
            )

            log_probability += math.log(probability)
            count += 1

    conn.close()

    return math.exp(-log_probability / count)