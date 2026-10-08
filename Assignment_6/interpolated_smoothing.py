"""
Interpolated Smoothing
Supports Bigram, Trigram and Quadgram models.

Formula:
P_interp(w_i | h) =
    lambda_1 P(w_i) +
    lambda_2 P(w_i | w_{i-1}) +
    ...
    + lambda_n P(w_i | w_{i-n+1}, ..., w_{i-1})

The lambdas supplied to the function should sum to 1.
"""

import math


def pack(ids, bits):
    key = 0
    for x in ids:
        key = (key << bits) | x
    return key


def mle_probability(ngram, model, lower_model=None):
    """Maximum-likelihood probability of an n-gram."""
    n = len(ngram)

    if n == 1:
        count = model["counts"].get(ngram[0], 0)
        total = model["total"]
        return count / total if total else 0.0

    counts = model["counts"]
    bits = model["bits"]

    ngram_key = pack(ngram, bits)
    context_key = pack(ngram[:-1], bits)

    ngram_count = counts.get(ngram_key, 0)
    context_count = lower_model["counts"].get(context_key, 0)

    return ngram_count / context_count if context_count else 0.0


def interpolated_probability(ngram, models, lambdas=None):
    """
    Calculate interpolated probability.

    models:
        {
            1: unigram_model,
            2: bigram_model,
            3: trigram_model,
            4: quadgram_model
        }

    For a trigram, for example:
        lambda_1 P(w_i)
        + lambda_2 P(w_i | w_{i-1})
        + lambda_3 P(w_i | w_{i-2}, w_{i-1})
    """

    n = len(ngram)

    if n < 2:
        raise ValueError("Interpolated smoothing is for bigram, trigram and quadgram models.")

    if lambdas is None:
        lambdas = [1.0 / n] * n

    if len(lambdas) != n:
        raise ValueError("Number of lambdas must equal the n-gram order.")

    if not math.isclose(sum(lambdas), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Lambdas must sum to 1.")

    probability = 0.0

    # Order 1 through n.
    for order in range(1, n + 1):
        sub_ngram = ngram[-order:]

        if order == 1:
            p = mle_probability(sub_ngram, models[1])
        else:
            p = mle_probability(
                sub_ngram,
                models[order],
                models[order - 1]
            )

        probability += lambdas[order - 1] * p

    return probability
