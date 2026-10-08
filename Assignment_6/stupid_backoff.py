"""
Stupid Backoff.

For a seen n-gram:
    S(w_i | h) = count(h, w_i) / count(h)

For an unseen n-gram:
    S(w_i | h) = alpha * S(w_i | h_without_first_word)

The usual value is:
    alpha = 0.4

Note:
Stupid Backoff scores are not normalized probabilities. They are
used directly for perplexity in this implementation, with a small
floor to avoid log(0).
"""

DEFAULT_ALPHA = 0.4
EPSILON = 1e-12


def pack(ids, bits):
    key = 0
    for x in ids:
        key = (key << bits) | x
    return key


def mle_probability(ngram, model, lower_model):
    """MLE probability for a seen n-gram."""
    bits = model["bits"]

    ngram_key = pack(ngram, bits)
    context_key = pack(ngram[:-1], bits)

    ngram_count = model["counts"].get(ngram_key, 0)
    context_count = lower_model["counts"].get(context_key, 0)

    if context_count == 0:
        return 0.0

    return ngram_count / context_count


def stupid_backoff_probability(
    ngram,
    models,
    alpha=DEFAULT_ALPHA
):
    """
    Calculate Stupid Backoff score recursively.

    models:
        {
            1: unigram_model,
            2: bigram_model,
            3: trigram_model,
            4: quadgram_model
        }
    """
    n = len(ngram)

    if n == 1:
        model = models[1]
        count = model["counts"].get(ngram[0], 0)
        total = sum(model["counts"].values())

        if total == 0:
            return EPSILON

        return max(count / total, EPSILON)

    model = models[n]
    lower_model = models[n - 1]

    bits = model["bits"]
    key = pack(ngram, bits)
    count = model["counts"].get(key, 0)

    if count > 0:
        probability = mle_probability(
            ngram,
            model,
            lower_model
        )

        return max(probability, EPSILON)

    # Back off by removing the oldest word.
    lower_score = stupid_backoff_probability(
        ngram[1:],
        models,
        alpha
    )

    return max(alpha * lower_score, EPSILON)
