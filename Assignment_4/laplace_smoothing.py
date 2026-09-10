def laplace_probability(ngram_count, context_count, vocabulary_size):
    return (
        ngram_count + 1
    ) / (
        context_count + vocabulary_size
    )