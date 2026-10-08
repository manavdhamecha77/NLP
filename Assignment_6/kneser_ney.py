"""
Interpolated Kneser-Ney Smoothing.

For a bigram:

P_KN(w_i | w_{i-1}) =
    max(C(w_{i-1}, w_i) - D, 0) / C(w_{i-1})
    +
    lambda(w_{i-1}) * P_cont(w_i)

For higher-order models:

P_KN(w_i | h) =
    max(C(h, w_i) - D, 0) / C(h)
    +
    lambda(h) * P_KN(w_i | suffix(h))

where:

lambda(h) = D * N_1+(h, *) / C(h)

and:

P_cont(w) = N_1+(*, w) / N_1+(**)

A single discount D is used.
Default D = 0.75.

Statistics are precomputed once so probability()
does not repeatedly scan the complete n-gram models.
"""

from collections import defaultdict


EPSILON = 1e-12
DEFAULT_DISCOUNT = 0.75


def pack(ids, bits):
    """Pack token IDs into an integer key."""
    key = 0

    for x in ids:
        key = (key << bits) | x

    return key


def unpack(key, length, bits):
    """Recover token IDs from a packed n-gram key."""
    ids = [0] * length
    mask = (1 << bits) - 1

    for i in range(length - 1, -1, -1):
        ids[i] = key & mask
        key >>= bits

    return tuple(ids)


class KneserNey:

    def __init__(
        self,
        models,
        discount=DEFAULT_DISCOUNT
    ):
        """
        models:
            {
                1: unigram_model,
                2: bigram_model,
                3: trigram_model,
                4: quadgram_model
            }
        """

        self.models = models
        self.discount = discount
        self.bits = models[2]["bits"]

        # -----------------------------------------------------
        # Precomputed context statistics
        # -----------------------------------------------------

        self.context_counts = {}
        self.unique_followers = {}

        # -----------------------------------------------------
        # Continuation statistics
        # -----------------------------------------------------

        self.continuation_counts = {}
        self.total_continuation_types = 0

        # -----------------------------------------------------
        # Build everything once
        # -----------------------------------------------------

        self._build_context_statistics()
        self._build_continuation_statistics()

    # =========================================================
    # Context statistics
    # =========================================================

    def _build_context_statistics(self):

        for order in range(2, 5):

            if order not in self.models:
                continue

            model = self.models[order]

            context_counts = defaultdict(int)
            unique_followers = defaultdict(set)

            for key, count in model["counts"].items():

                ngram = unpack(
                    key,
                    order,
                    self.bits
                )

                context = ngram[:-1]
                word = ngram[-1]

                # C(h)
                context_counts[context] += count

                # N_1+(h,*)
                unique_followers[context].add(word)

            self.context_counts[order] = dict(
                context_counts
            )

            self.unique_followers[order] = {
                context: len(words)
                for context, words in unique_followers.items()
            }

    # =========================================================
    # Continuation statistics
    # =========================================================

    def _build_continuation_statistics(self):

        # word -> set of distinct preceding words
        predecessors = defaultdict(set)

        bigram_model = self.models[2]

        for key in bigram_model["counts"]:

            previous_word, word = unpack(
                key,
                2,
                self.bits
            )

            predecessors[word].add(
                previous_word
            )

        # N_1+(*, w)
        self.continuation_counts = {
            word: len(previous_words)
            for word, previous_words in predecessors.items()
        }

        # N_1+(**)
        self.total_continuation_types = sum(
            self.continuation_counts.values()
        )

    # =========================================================
    # Continuation probability
    # =========================================================

    def continuation_probability(self, word):

        if self.total_continuation_types == 0:
            return EPSILON

        probability = (
            self.continuation_counts.get(word, 0)
            / self.total_continuation_types
        )

        return max(
            probability,
            EPSILON
        )

    # =========================================================
    # Kneser-Ney probability
    # =========================================================

    def probability(self, ngram):

        ngram = tuple(ngram)

        order = len(ngram)

        if order == 0:
            return EPSILON

        # -----------------------------------------------------
        # Unigram
        # -----------------------------------------------------

        if order == 1:

            return self.continuation_probability(
                ngram[0]
            )

        if order > 4:
            raise ValueError(
                "Only up to quadgrams are supported."
            )

        model = self.models[order]

        context = ngram[:-1]
        word = ngram[-1]

        key = pack(
            ngram,
            self.bits
        )

        count = model["counts"].get(
            key,
            0
        )

        context_count = self.context_counts[order].get(
            context,
            0
        )

        # -----------------------------------------------------
        # Unseen context
        # -----------------------------------------------------

        if context_count == 0:

            return self.probability(
                ngram[1:]
            )

        # -----------------------------------------------------
        # First term
        #
        # max(C(h,w) - D, 0) / C(h)
        # -----------------------------------------------------

        discounted_count = max(
            count - self.discount,
            0.0
        )

        first_term = (
            discounted_count
            / context_count
        )

        # -----------------------------------------------------
        # Lambda(h)
        #
        # D * N_1+(h,*) / C(h)
        # -----------------------------------------------------

        distinct_followers = self.unique_followers[
            order
        ].get(
            context,
            0
        )

        lambda_h = (
            self.discount
            * distinct_followers
            / context_count
        )

        # -----------------------------------------------------
        # Lower-order probability
        # -----------------------------------------------------

        lower_probability = self.probability(
            ngram[1:]
        )

        probability = (
            first_term
            + lambda_h * lower_probability
        )

        return max(
            probability,
            EPSILON
        )


def kneser_ney_probability(
    ngram,
    models,
    discount=DEFAULT_DISCOUNT
):
    """
    Convenience wrapper.

    Prefer creating KneserNey(models) once and calling
    .probability() repeatedly when evaluating a dataset.
    """

    kn = KneserNey(
        models,
        discount=discount
    )

    return kn.probability(
        tuple(ngram)
    )