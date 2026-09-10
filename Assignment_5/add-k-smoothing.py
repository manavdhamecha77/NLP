import sqlite3

K = 0.3


def probability(db_path, ngram):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    n = len(ngram)

    if n == 1:
        cursor.execute(
            "SELECT count FROM counts WHERE word=?",
            (ngram[0],)
        )
        result = cursor.fetchone()
        count = result[0] if result else 0

        cursor.execute("SELECT total, vocabulary FROM meta")
        total, vocabulary = cursor.fetchone()

        conn.close()

        return (count + K) / (total + K * vocabulary)

    if n == 2:
        cursor.execute(
            "SELECT count FROM counts WHERE w1=? AND w2=?",
            tuple(ngram)
        )
        result = cursor.fetchone()
        count = result[0] if result else 0

        cursor.execute(
            "SELECT count FROM contexts WHERE word=?",
            (ngram[0],)
        )
        result = cursor.fetchone()
        context_count = result[0] if result else 0

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
        context_count = result[0] if result else 0

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
        context_count = result[0] if result else 0

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    conn.close()

    return (count + K) / (context_count + K * vocabulary)