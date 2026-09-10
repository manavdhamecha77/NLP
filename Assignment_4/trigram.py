import sqlite3


class TrigramModel:

    def __init__(self, db_path="models/trigram.db"):
        self.db_path = db_path

    def train(self, input_file):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS counts")
        cursor.execute("""
            CREATE TABLE counts (
                w1 TEXT,
                w2 TEXT,
                w3 TEXT,
                count INTEGER,
                PRIMARY KEY(w1, w2, w3)
            )
        """)

        cursor.execute("DROP TABLE IF EXISTS contexts")
        cursor.execute("""
            CREATE TABLE contexts (
                w1 TEXT,
                w2 TEXT,
                count INTEGER,
                PRIMARY KEY(w1, w2)
            )
        """)

        cursor.execute("DROP TABLE IF EXISTS meta")
        cursor.execute("""
            CREATE TABLE meta (
                vocabulary INTEGER
            )
        """)

        batch = {}

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()

                if not words:
                    continue

                words = ["<s>", "<s>"] + words + ["</s>"]

                for i in range(len(words) - 2):
                    context = (words[i], words[i + 1])
                    ngram = (
                        words[i],
                        words[i + 1],
                        words[i + 2]
                    )

                    batch[("context", *context)] = (
                        batch.get(("context", *context), 0) + 1
                    )

                    batch[ngram] = batch.get(ngram, 0) + 1

                if len(batch) >= 50000:
                    self._insert_batch(cursor, batch)
                    conn.commit()
                    batch.clear()

        if batch:
            self._insert_batch(cursor, batch)

        cursor.execute(
            "SELECT COUNT(DISTINCT w1) FROM counts"
        )
        vocabulary = cursor.fetchone()[0]

        cursor.execute(
            "INSERT INTO meta VALUES (?)",
            (vocabulary + 1,)
        )

        conn.commit()
        conn.close()

    def _insert_batch(self, cursor, batch):
        ngrams = []
        contexts = []

        for key, count in batch.items():
            if key[0] == "context":
                contexts.append(
                    (key[1], key[2], count)
                )
            else:
                ngrams.append(
                    (key[0], key[1], key[2], count)
                )

        if ngrams:
            cursor.executemany(
                """
                INSERT INTO counts(w1, w2, w3, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(w1, w2, w3)
                DO UPDATE SET count = count + excluded.count
                """,
                ngrams
            )

        if contexts:
            cursor.executemany(
                """
                INSERT INTO contexts(w1, w2, count)
                VALUES (?, ?, ?)
                ON CONFLICT(w1, w2)
                DO UPDATE SET count = count + excluded.count
                """,
                contexts
            )