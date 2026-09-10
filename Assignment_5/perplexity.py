import math
import sqlite3


def perplexity(db_path, sentences, n, k=0.3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    if n == 1:
        cursor.execute("SELECT total FROM meta")
        total = cursor.fetchone()[0]

    log_probability = 0
    token_count = 0

    for sentence in sentences:
        words = (
            ["<s>"] * (n - 1)
            + sentence.split()
            + ["</s>"]
        )

        for i in range(len(words) - n + 1):
            ngram = words[i:i + n]

            if n == 1:
                cursor.execute(
                    "SELECT count FROM counts WHERE word=?",
                    (ngram[0],)
                )

                result = cursor.fetchone()
                count = result[0] if result else 0

                p = (count + k) / (
                    total + k * vocabulary
                )

            elif n == 2:
                cursor.execute(
                    """
                    SELECT count FROM counts
                    WHERE w1=? AND w2=?
                    """,
                    tuple(ngram)
                )

                result = cursor.fetchone()
                count = result[0] if result else 0

                cursor.execute(
                    "SELECT count FROM contexts WHERE word=?",
                    (ngram[0],)
                )

                result = cursor.fetchone()
                context = result[0] if result else 0

                p = (count + k) / (
                    context + k * vocabulary
                )

            elif n == 3:
                cursor.execute(
                    """
                    SELECT count FROM counts
                    WHERE w1=? AND w2=? AND w3=?
                    """,
                    tuple(ngram)
                )

                result = cursor.fetchone()
                count = result[0] if result else 0

                cursor.execute(
                    """
                    SELECT count FROM contexts
                    WHERE w1=? AND w2=?
                    """,
                    tuple(ngram[:2])
                )

                result = cursor.fetchone()
                context = result[0] if result else 0

                p = (count + k) / (
                    context + k * vocabulary
                )

            else:
                cursor.execute(
                    """
                    SELECT count FROM counts
                    WHERE w1=? AND w2=? AND w3=? AND w4=?
                    """,
                    tuple(ngram)
                )

                result = cursor.fetchone()
                count = result[0] if result else 0

                cursor.execute(
                    """
                    SELECT count FROM contexts
                    WHERE w1=? AND w2=? AND w3=?
                    """,
                    tuple(ngram[:3])
                )

                result = cursor.fetchone()
                context = result[0] if result else 0

                p = (count + k) / (
                    context + k * vocabulary
                )

            log_probability += math.log(p)
            token_count += 1

    conn.close()

    return math.exp(
        -log_probability / token_count
    )