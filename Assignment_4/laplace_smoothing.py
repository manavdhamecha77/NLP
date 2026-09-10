import sqlite3


def unigram_probability(db_path, word):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT count FROM counts WHERE word = ?",
        (word,)
    )

    result = cursor.fetchone()
    count = result[0] if result else 0

    cursor.execute(
        "SELECT total, vocabulary FROM meta"
    )

    total, vocabulary = cursor.fetchone()

    conn.close()

    return (count + 1) / (total + vocabulary)


def bigram_probability(db_path, w1, w2):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT count FROM counts WHERE w1=? AND w2=?",
        (w1, w2)
    )

    result = cursor.fetchone()
    count = result[0] if result else 0

    cursor.execute(
        "SELECT count FROM contexts WHERE word=?",
        (w1,)
    )

    result = cursor.fetchone()
    context_count = result[0] if result else 0

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    conn.close()

    return (count + 1) / (context_count + vocabulary)


def trigram_probability(db_path, w1, w2, w3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT count
        FROM counts
        WHERE w1=? AND w2=? AND w3=?
        """,
        (w1, w2, w3)
    )

    result = cursor.fetchone()
    count = result[0] if result else 0

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

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    conn.close()

    return (count + 1) / (context_count + vocabulary)


def quadgram_probability(db_path, w1, w2, w3, w4):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT count
        FROM counts
        WHERE w1=? AND w2=? AND w3=? AND w4=?
        """,
        (w1, w2, w3, w4)
    )

    result = cursor.fetchone()
    count = result[0] if result else 0

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

    cursor.execute("SELECT vocabulary FROM meta")
    vocabulary = cursor.fetchone()[0]

    conn.close()

    return (count + 1) / (context_count + vocabulary)