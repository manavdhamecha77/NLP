"""
Katz Backoff Smoothing.

For a seen n-gram:
    P_Katz = d(c) * P_ML

For an unseen n-gram:
    P_Katz = alpha(h) * P_Katz(lower-order)

Good-Turing discounting is used for d(c).

All context-level statistics and backoff weights are precomputed once
to avoid repeatedly scanning the full n-gram vocabulary.
"""

from collections import Counter, defaultdict


EPSILON = 1e-12
DEFAULT_CUTOFF = 5


def pack(ids, bits):
    key = 0
    for x in ids:
        key = (key << bits) | x
    return key


def frequency_of_frequencies(model):
    return Counter(model["counts"].values())


def good_turing_discount(count, counts_of_counts, cutoff=DEFAULT_CUTOFF):
    """Return the Good-Turing discount d(c) = c* / c."""
    if count <= 0 or count > cutoff:
        return 1.0

    Nc = counts_of_counts.get(count, 0)
    Ncp1 = counts_of_counts.get(count + 1, 0)

    if Nc == 0 or Ncp1 == 0:
        return 1.0

    adjusted = (count + 1) * Ncp1 / Nc

    return adjusted / count


def discounted_probability(
    count,
    context_count,
    counts_of_counts,
    cutoff=DEFAULT_CUTOFF
):
    """Return discounted conditional probability."""
    if count <= 0 or context_count <= 0:
        return 0.0

    if count > cutoff:
        return count / context_count

    discount = good_turing_discount(
        count,
        counts_of_counts,
        cutoff
    )

    return discount * count / context_count


def build_katz_table(
    model,
    lower_model,
    cutoff=DEFAULT_CUTOFF
):
    """
    Precompute all statistics required by Katz Backoff.

    Returns:
        {
            "discounts": {count: d(c)},
            "context_seen_sum": {context_key: sum P_discounted},
            "context_lower_seen_sum": {context_key: sum P_lower},
            "alpha": {context_key: alpha(context)}
        }
    """

    counts = model["counts"]
    lower_counts = lower_model["counts"]
    bits = model["bits"]

    counts_of_counts = frequency_of_frequencies(model)

    # ---------------------------------------------------------
    # 1. Precompute discounts d(c)
    # ---------------------------------------------------------

    discounts = {}

    for count in range(1, cutoff + 1):
        discounts[count] = good_turing_discount(
            count,
            counts_of_counts,
            cutoff
        )

    # ---------------------------------------------------------
    # 2. Calculate discounted probability mass per context
    # ---------------------------------------------------------

    context_seen_sum = defaultdict(float)

    for key, count in counts.items():

        context_key = key >> bits

        context_count = lower_counts.get(context_key, 0)

        if context_count == 0:
            continue

        if count > cutoff:
            probability = count / context_count
        else:
            probability = (
                discounts.get(count, 1.0)
                * count
                / context_count
            )

        context_seen_sum[context_key] += probability

    # ---------------------------------------------------------
    # 3. Calculate lower-order probability mass for seen words
    #
    # For Katz alpha:
    #
    # alpha(h) =
    #   (1 - discounted_seen_mass)
    #   -------------------------
    #   (1 - lower_seen_mass)
    #
    # For unigram lower probability:
    #
    # P(w) = count(w) / total
    # ---------------------------------------------------------

    lower_total = sum(lower_counts.values())

    context_lower_seen_sum = defaultdict(float)

    if lower_total > 0:

        for key in counts:

            context_key = key >> bits

            word_id = key & ((1 << bits) - 1)

            word_count = lower_counts.get(word_id, 0)

            context_lower_seen_sum[context_key] += (
                word_count / lower_total
            )

    # ---------------------------------------------------------
    # 4. Precompute alpha(context)
    # ---------------------------------------------------------

    alpha = {}

    all_contexts = set(context_seen_sum) | set(context_lower_seen_sum)

    for context_key in all_contexts:

        seen_mass = context_seen_sum.get(
            context_key,
            0.0
        )

        lower_seen_mass = context_lower_seen_sum.get(
            context_key,
            0.0
        )

        numerator = max(
            0.0,
            1.0 - seen_mass
        )

        denominator = max(
            EPSILON,
            1.0 - lower_seen_mass
        )

        alpha[context_key] = numerator / denominator

    return {
        "discounts": discounts,
        "context_seen_sum": dict(context_seen_sum),
        "context_lower_seen_sum": dict(context_lower_seen_sum),
        "alpha": alpha,
        "counts_of_counts": counts_of_counts,
    }


def katz_probability(
    ngram,
    model,
    lower_model,
    lower_probability,
    cutoff=DEFAULT_CUTOFF,
    probability_table=None
):
    """
    Calculate Katz Backoff probability.

    probability_table should be created once using:

        build_katz_table(model, lower_model)

    lower_probability must return the lower-order probability
    for the final word.
    """

    bits = model["bits"]

    key = pack(ngram, bits)

    count = model["counts"].get(
        key,
        0
    )

    # ---------------------------------------------------------
    # Seen n-gram
    # ---------------------------------------------------------

    if count > 0:

        context_key = pack(
            ngram[:-1],
            bits
        )

        context_count = lower_model["counts"].get(
            context_key,
            0
        )

        if context_count == 0:
            return EPSILON

        if probability_table is not None:
            discount = probability_table["discounts"].get(
                count,
                1.0
            )
        else:
            counts_of_counts = frequency_of_frequencies(model)

            discount = good_turing_discount(
                count,
                counts_of_counts,
                cutoff
            )

        probability = (
            discount
            * count
            / context_count
        )

        return max(
            probability,
            EPSILON
        )

    # ---------------------------------------------------------
    # Unseen n-gram -> backoff
    # ---------------------------------------------------------

    context = ngram[:-1]

    context_key = pack(
        context,
        bits
    )

    if probability_table is not None:

        alpha = probability_table["alpha"].get(
            context_key,
            1.0
        )

    else:

        alpha = 1.0

    lower_probability_value = lower_probability(
        (ngram[-1],)
    )

    return max(
        alpha * lower_probability_value,
        EPSILON
    )