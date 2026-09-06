from dfa import check_word, show_dfa_transition_table
from fst import load_nouns, analyze_noun, show_fst_transition_table


def main():
    nouns = load_nouns("brown_nouns.txt")

    while True:
        print("\n1. DFA - Check English Word")
        print("2. DFA - Show Transition Table")
        print("3. FST - Analyze Noun")
        print("4. FST - Show Transition Table")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            word = input("Enter a word: ")
            print(check_word(word))

        elif choice == "2":
            show_dfa_transition_table()

        elif choice == "3":
            word = input("Enter a noun: ")
            result = analyze_noun(word, nouns)
            print(f"{word} = {result}")

        elif choice == "4":
            show_fst_transition_table()

        elif choice == "5":
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()