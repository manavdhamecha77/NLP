def load_nouns(filename):
    nouns = set()

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip().lower()

            if word:
                nouns.add(word)

    return nouns


def get_fst_transition_table():
    return [
        ("q0", "word in lexicon", "copy+N+SG", "q_sg"),
        ("q0", "ends with ies", "ies -> y+N+PL", "q_pl"),
        ("q0", "ends with es", "remove es", "q_special"),
        ("q_special", "root ends s/z/x/ch/sh", "+N+PL", "q_pl"),
        ("q0", "ends with s", "remove s", "q_normal"),
        ("q_normal", "root is normal noun", "+N+PL", "q_pl"),
        ("q0", "invalid", "Invalid Word", "q_dead")
    ]


def analyze_noun(word, nouns):
    word = word.lower()
    state = "q0"

    if word.endswith("ies") and len(word) > 3:
        state = "q_y"
        root = word[:-3] + "y"

        if root in nouns:
            state = "q_pl"
            return f"{root}+N+PL"

    if word.endswith("es") and len(word) > 2:
        state = "q_es"
        root = word[:-2]

        state = "q_special"

        special_endings = ("s", "z", "x", "ch", "sh")

        if root.endswith(special_endings) and root in nouns:
            state = "q_pl"
            return f"{root}+N+PL"

    if word.endswith("s") and len(word) > 1:
        state = "q_s"
        root = word[:-1]

        special_endings = ("s", "z", "x", "ch", "sh")

        if root in nouns and not root.endswith(special_endings):
            state = "q_normal"
            state = "q_pl"
            return f"{root}+N+PL"

    if word in nouns:
        state = "q_sg"
        return f"{word}+N+SG"

    state = "q_dead"
    return "Invalid Word"


def show_fst_transition_table():
    print("\nInput Alphabet: {a, b, ..., z}")
    print("Output Alphabet: {a, b, ..., z, +, N, SG, PL}\n")

    print(f"{'Current State':<15}{'Input':<28}{'Output':<25}{'Next State'}")
    print("-" * 85)

    for current, input_symbol, output, next_state in get_fst_transition_table():
        print(f"{current:<15}{input_symbol:<28}{output:<25}{next_state}")