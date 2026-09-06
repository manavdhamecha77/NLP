def get_transition_table():
    return {
        "q0": {
            "lowercase": "q1",
            "other": "q_dead"
        },
        "q1": {
            "lowercase": "q1",
            "other": "q_dead"
        },
        "q_dead": {
            "lowercase": "q_dead",
            "other": "q_dead"
        }
    }


def check_word(word):
    transitions = get_transition_table()
    state = "q0"

    for char in word:
        if "a" <= char <= "z":
            symbol = "lowercase"
        else:
            symbol = "other"

        state = transitions[state][symbol]

    if state == "q1":
        return "Accepted"

    return "Not Accepted"


def show_dfa_transition_table():
    transitions = get_transition_table()

    print("\nInput Alphabet: {a-z, other}")
    print("Accepting State: q1\n")

    print(f"{'State':<12}{'a-z':<15}{'other'}")
    print("-" * 40)

    print(f"{'-> q0':<12}{transitions['q0']['lowercase']:<15}{transitions['q0']['other']}")
    print(f"{'* q1':<12}{transitions['q1']['lowercase']:<15}{transitions['q1']['other']}")
    print(f"{'q_dead':<12}{transitions['q_dead']['lowercase']:<15}{transitions['q_dead']['other']}")