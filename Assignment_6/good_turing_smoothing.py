"""
Good-Turing Smoothing for Bigram, Trigram and Quadgram Models.

For an n-gram with observed count c:

    c* = (c + 1) * N_{c+1} / N_c

For conditional n-gram probabilities:

    P(w | h) = c*(h,w) / sum_w c*(h,w)

where:
    N_c = number of n-grams occurring exactly c times.

Counts above the cutoff use their original counts.
"""

from collections import Counter, defaultdict


def pack(ids, bits):
    """Pack a sequence of token IDs into one integer."""
    key = 0

    for x in ids:
        key = (key << bits) | x

    return key


def frequency_of_frequencies(model):
    """
    Return frequency-of-frequencies.

    N_c = number of n-gram types whose count is exactly c.
    """
    return Counter(model["counts"].values())


def good_turing_count(count, counts_of_counts, cutoff=5):
    """
    Return the Good-Turing adjusted count c*.

        c* = (c + 1) * N_{c+1} / N_c
    """

    if count == 0:
        return counts_of_counts.get(1, 0)

    if count > cutoff:
        return float(count)

    Nc = counts_of_counts.get(count, 0)
    Ncp1 = counts_of_counts.get(count + 1, 0)

    if Nc == 0 or Ncp1 == 0:
        return float(count)

    return (count + 1) * Ncp1 / Nc


def build_good_turing_table(model, lower_model, cutoff=5):
    """
    Precompute all Good-Turing statistics.

    The statistics are calculated once before evaluation so that
    probability lookup during Dev/Test evaluation remains fast.
    """

    counts = model["counts"]
    bits = model["bits"]

    # ---------------------------------------------------------
    # 1. Frequency of frequencies
    # ---------------------------------------------------------

    counts_of_counts = frequency_of_frequencies(model)

    # ---------------------------------------------------------
    # 2. Good-Turing adjusted counts
    # ---------------------------------------------------------

    adjusted = {}

    for c in range(1, cutoff + 1):
        adjusted[c] = good_turing_count(
            c,
            counts_of_counts,
            cutoff
        )

    # ---------------------------------------------------------
    # 3. Calculate adjusted count total for every context
    # ---------------------------------------------------------

    context_total = defaultdict(float)

    for key, count in counts.items():

        # Remove the final word ID.
        context_key = key >> bits

        if count <= cutoff:
            adjusted_count = adjusted.get(
                count,
                float(count)
            )
        else:
            adjusted_count = float(count)

        context_total[context_key] += adjusted_count

    # ---------------------------------------------------------
    # 4. Find how many words have been observed after
    #    every context.
    # ---------------------------------------------------------

    context_seen = defaultdict(int)

    for key in counts:

        context_key = key >> bits

        context_seen[context_key] += 1

    # ---------------------------------------------------------
    # 5. Number of unseen words for every context.
    # ---------------------------------------------------------

    vocabulary_size = model["vocab_size"]

    context_unseen = {}

    for context_key, seen_count in context_seen.items():

        context_unseen[context_key] = max(
            vocabulary_size - seen_count,
            1
        )

    # ---------------------------------------------------------
    # 6. Store total token count.
    # ---------------------------------------------------------

    total = sum(counts.values())

    # ---------------------------------------------------------
    # 7. Probability mass associated with count = 1.
    #
    #    N1 / N is the total probability mass associated with
    #    unseen events in the basic Good-Turing formulation.
    # ---------------------------------------------------------

    N1 = counts_of_counts.get(1, 0)

    unseen_probability_mass = (
        N1 / total
        if total > 0
        else 0.0
    )

    return {
        "adjusted": adjusted,
        "context_total": dict(context_total),
        "context_seen": dict(context_seen),
        "context_unseen": context_unseen,
        "unseen_probability_mass": unseen_probability_mass,
        "counts_of_counts": counts_of_counts,
        "total": total,
    }


def good_turing_probability(
    ngram,
    model,
    probability_table,
    lower_model,
    cutoff=5
):
    """
    Calculate conditional Good-Turing probability:

        P(w | h)

    for a given n-gram.
    """

    bits = model["bits"]

    # ---------------------------------------------------------
    # N-gram key
    # ---------------------------------------------------------

    ngram_key = pack(ngram, bits)

    # ---------------------------------------------------------
    # Context key
    # ---------------------------------------------------------

    context_key = pack(
        ngram[:-1],
        bits
    )

    # ---------------------------------------------------------
    # Observed n-gram count
    # ---------------------------------------------------------

    count = model["counts"].get(
        ngram_key,
        0
    )

    # ---------------------------------------------------------
    # Context count
    # ---------------------------------------------------------

    context_count = lower_model["counts"].get(
        context_key,
        0
    )

    if context_count == 0:
        return 1e-12

    # ---------------------------------------------------------
    # SEEN N-GRAM
    # ---------------------------------------------------------

    if count > 0:

        if count <= cutoff:

            adjusted_count = probability_table["adjusted"].get(
                count,
                float(count)
            )

        else:

            adjusted_count = float(count)

        denominator = probability_table["context_total"].get(
            context_key,
            float(context_count)
        )

        if denominator <= 0:
            return 1e-12

        probability = adjusted_count / denominator

        return max(probability, 1e-12)

    # ---------------------------------------------------------
    # UNSEEN N-GRAM
    # ---------------------------------------------------------

    unseen_count = probability_table["context_unseen"].get(
        context_key,
        1
    )

    if unseen_count <= 0:
        return 1e-12

    # Basic Good-Turing unseen probability mass.
    unseen_mass = probability_table[
        "unseen_probability_mass"
    ]

    # Distribute the unseen probability mass across the
    # unseen word types for this context.
    probability = unseen_mass / unseen_count

    return max(probability, 1e-12)