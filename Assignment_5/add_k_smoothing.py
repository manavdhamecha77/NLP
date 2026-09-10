K = 0.3


def add_k_probability(ngram_count, context_count, vocabulary_size):
    return (
        ngram_count + K
    ) / (
        context_count + K * vocabulary_size
    )