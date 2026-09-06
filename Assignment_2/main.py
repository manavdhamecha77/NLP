import json
import math

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]

        node.is_word = True

    def build(self, vocabulary):
        for word in vocabulary:
            self.insert(word)

def greedy_segmentation(text, trie):
    """
    Segment the text by choosing the longest matching word
    from the vocabulary at each position.
    """
    words = []
    i = 0

    while i < len(text):
        node = trie.root
        longest_word = None
        j = i

        while j < len(text) and text[j] in node.children:
            node = node.children[text[j]]
            j += 1

            if node.is_word:
                longest_word = text[i:j]

        if longest_word is not None:
            words.append(longest_word)
            i += len(longest_word)
        else:
            words.append(text[i])
            i += 1

    return words

def dynamic_programming_segmentation(
    text,
    log_probabilities,
    trie
):
    """
    Find the segmentation with the maximum total log probability.

    dp[i] stores the best log probability for the first i characters.
    parent[i] stores where the final word started.
    """
    n = len(text)

    dp = [-math.inf] * (n + 1)
    parent = [-1] * (n + 1)

    dp[0] = 0

    for start in range(n):
        if dp[start] == -math.inf:
            continue

        node = trie.root
        end = start

        while end < n and text[end] in node.children:
            node = node.children[text[end]]
            end += 1

            if node.is_word:
                word = text[start:end]

                score = dp[start] + log_probabilities[word]

                if score > dp[end]:
                    dp[end] = score
                    parent[end] = start

    if parent[n] == -1:
        return [text]

    words = []
    current = n

    while current > 0:
        previous = parent[current]
        words.append(text[previous:current])
        current = previous

    words.reverse()

    return words


def levenshtein_distance(words1, words2):
    """
    Calculate Levenshtein edit distance between two
    sequences of words.
    """
    m = len(words1)
    n = len(words2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):

            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            else:
                insert = dp[i][j - 1]
                delete = dp[i - 1][j]
                replace = dp[i - 1][j - 1]

                dp[i][j] = 1 + min(
                    insert,
                    delete,
                    replace
                )

    return dp[m][n]


def calculate_word_accuracy(predicted, actual):
    """
    Calculate the fraction of ground-truth words that
    are correctly predicted at the same position.
    """
    if len(actual) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    correct = 0

    for i in range(min(len(predicted), len(actual))):
        if predicted[i] == actual[i]:
            correct += 1

    return correct / len(actual)


def main():
    input_file = "text_segmentation_dataset.json"

    predictions_file = "text_segmentation_predictions.json"
    metrics_file = "text_segmentation_metrics.json"

    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    word_counts = data["word_counts"]
    test_cases = data["test_cases"]
    total_words = data["metadata"]["total_corpus_words"]

    vocabulary = set(word_counts.keys())
    
    trie = Trie()
    trie.build(vocabulary)

    # Calculate log probability for every vocabulary word
    log_probabilities = {}

    for word, count in word_counts.items():
        probability = count / total_words
        log_probabilities[word] = math.log(probability)

    predictions = []

    greedy_exact_matches = 0
    dp_exact_matches = 0

    greedy_word_accuracy_total = 0.0
    dp_word_accuracy_total = 0.0

    greedy_edit_distance_total = 0
    dp_edit_distance_total = 0

    # Process every test case
    for case_number, case in enumerate(test_cases, start=1):

        text = case["input"]
        ground_truth = case["ground_truth"]

        actual_words = ground_truth.split()

        # Greedy prediction
        greedy_words = greedy_segmentation(
            text,
            trie
        )

        # DP prediction
        dp_words = dynamic_programming_segmentation(
            text,
            log_probabilities,
            trie
        )

        greedy_prediction = " ".join(greedy_words)
        dp_prediction = " ".join(dp_words)

        # Exact accuracy
        greedy_exact = greedy_words == actual_words
        dp_exact = dp_words == actual_words

        if greedy_exact:
            greedy_exact_matches += 1

        if dp_exact:
            dp_exact_matches += 1

        # Word accuracy
        greedy_acc = calculate_word_accuracy(
            greedy_words,
            actual_words
        )

        dp_acc = calculate_word_accuracy(
            dp_words,
            actual_words
        )

        greedy_word_accuracy_total += greedy_acc
        dp_word_accuracy_total += dp_acc

        # Edit distance
        greedy_ed = levenshtein_distance(
            greedy_words,
            actual_words
        )

        dp_ed = levenshtein_distance(
            dp_words,
            actual_words
        )

        greedy_edit_distance_total += greedy_ed
        dp_edit_distance_total += dp_ed

        # Store predictions for this test case
        predictions.append({
            "case_number": case_number,
            "input": text,
            "ground_truth": ground_truth,
            "greedy": greedy_prediction,
            "dp": dp_prediction
        })

    total_cases = len(test_cases)

    # Calculate final metrics
    greedy_exact_accuracy = (
        greedy_exact_matches / total_cases
    )

    dp_exact_accuracy = (
        dp_exact_matches / total_cases
    )

    greedy_word_accuracy = (
        greedy_word_accuracy_total / total_cases
    )

    dp_word_accuracy = (
        dp_word_accuracy_total / total_cases
    )

    greedy_average_edit_distance = (
        greedy_edit_distance_total / total_cases
    )

    dp_average_edit_distance = (
        dp_edit_distance_total / total_cases
    )

    # JSON containing every prediction
    prediction_output = {
        "test_case_count": total_cases,
        "predictions": predictions
    }

    # JSON containing evaluation results
    metrics_output = {
        "dataset": {
            "file": input_file,
            "vocabulary_size": data["metadata"]["vocabulary_size"],
            "test_case_count": total_cases,
            "total_corpus_words": total_words
        },
        "greedy": {
            "exact_matches": greedy_exact_matches,
            "incorrect_cases": total_cases - greedy_exact_matches,
            "exact_accuracy": greedy_exact_accuracy,
            "exact_accuracy_percent": round(
                greedy_exact_accuracy * 100,
                2
            ),
            "word_accuracy": greedy_word_accuracy,
            "word_accuracy_percent": round(
                greedy_word_accuracy * 100,
                2
            ),
            "total_edit_distance": greedy_edit_distance_total,
            "average_edit_distance": round(
                greedy_average_edit_distance,
                3
            )
        },
        "dynamic_programming": {
            "exact_matches": dp_exact_matches,
            "incorrect_cases": total_cases - dp_exact_matches,
            "exact_accuracy": dp_exact_accuracy,
            "exact_accuracy_percent": round(
                dp_exact_accuracy * 100,
                2
            ),
            "word_accuracy": dp_word_accuracy,
            "word_accuracy_percent": round(
                dp_word_accuracy * 100,
                2
            ),
            "total_edit_distance": dp_edit_distance_total,
            "average_edit_distance": round(
                dp_average_edit_distance,
                3
            )
        }
    }

    # Save all predictions
    with open(
        predictions_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            prediction_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Save metrics
    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Print summary
    print("Results")

    print("\nGreedy approach")
    print(
        "Exact accuracy:",
        round(greedy_exact_accuracy * 100, 2),
        "%"
    )
    print(
        "Word accuracy:",
        round(greedy_word_accuracy * 100, 2),
        "%"
    )
    print(
        "Average edit distance:",
        round(greedy_average_edit_distance, 3)
    )

    print("\nDynamic programming approach")
    print(
        "Exact accuracy:",
        round(dp_exact_accuracy * 100, 2),
        "%"
    )
    print(
        "Word accuracy:",
        round(dp_word_accuracy * 100, 2),
        "%"
    )
    print(
        "Average edit distance:",
        round(dp_average_edit_distance, 3)
    )

    print("\nFiles created:")
    print(predictions_file)
    print(metrics_file)


if __name__ == "__main__":
    main()