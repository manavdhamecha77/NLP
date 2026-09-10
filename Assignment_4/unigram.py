import sqlite3


class UnigramModel:

    def __init__(self, db_path="models/unigram.db"):
        self.db_path = db_path

    def train(self, input_file):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS counts")
        cursor.execute("""
            CREATE TABLE counts (
                word TEXT PRIMARY KEY,
                count INTEGER
            )
        """)

        cursor.execute("DROP TABLE IF EXISTS meta")
        cursor.execute("""
            CREATE TABLE meta (
                total INTEGER,
                vocabulary INTEGER
            )
        """)

        batch = {}

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                for word in line.split():
                    batch[word] = batch.get(word, 0) + 1

                if len(batch) >= 50000:
                    cursor.executemany(
                        """
                        INSERT INTO counts(word, count)
                        VALUES (?, ?)
                        ON CONFLICT(word)
                        DO UPDATE SET count = count + excluded.count
                        """,
                        batch.items()
                    )
                    conn.commit()
                    batch.clear()

        if batch:
            cursor.executemany(
                """
                INSERT INTO counts(word, count)
                VALUES (?, ?)
                ON CONFLICT(word)
                DO UPDATE SET count = count + excluded.count
                """,
                batch.items()
            )

        cursor.execute("SELECT SUM(count), COUNT(*) FROM counts")
        total, vocabulary = cursor.fetchone()

        cursor.execute(
            "INSERT INTO meta VALUES (?, ?)",
            (total, vocabulary + 1)
        )

        conn.commit()
        conn.close()