import math
import pickle
from tqdm import tqdm

from interpolated_smoothing import interpolated_probability

from good_turing_smoothing import (
    good_turing_probability,
    build_good_turing_table,
)

from katz_backoff import (
    build_katz_table,
    katz_probability,
)

from stupid_backoff import stupid_backoff_probability
from kneser_ney import KneserNey


GT_TABLES = {}
KATZ_TABLES = {}
KN_MODEL = None

MODEL_PATH = "../Assignment_4/models"


def load_model(name):
    with open(f"{MODEL_PATH}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_vocab():
    with open(f"{MODEL_PATH}/vocab.pkl", "rb") as f:
        return pickle.load(f)


def load_all_models():
    models = {
        i: load_model(name)
        for i, name in [
            (1, "unigram"),
            (2, "bigram"),
            (3, "trigram"),
            (4, "quadgram"),
        ]
    }

    # Store total count for every model.
    for model in models.values():
        model["_total"] = sum(model["counts"].values())

    # ---------------------------------------------------------
    # Good-Turing tables
    # ---------------------------------------------------------

    for order in (2, 3, 4):
        GT_TABLES[order] = build_good_turing_table(
            models[order],
            models[order - 1],
        )

    # ---------------------------------------------------------
    # Katz Backoff tables
    # ---------------------------------------------------------

    for order in (2, 3, 4):
        KATZ_TABLES[order] = build_katz_table(
            models[order],
            models[order - 1],
        )
        
    global KN_MODEL
    
    KN_MODEL = KneserNey(
        models,
        discount=0.75
    )

    return models


def encode_sentence(line, vocab):
    return (
        [vocab["<s>"]]
        + [vocab.get(w, vocab["<UNK>"]) for w in line.split()]
        + [vocab["</s>"]]
    )


def pack(ids, bits):
    key = 0

    for x in ids:
        key = (key << bits) | x

    return key


def unigram_probability(word_id, models):
    model = models[1]

    total = model["_total"]

    if total == 0:
        return 1e-12

    count = model["counts"].get(
        word_id,
        0,
    )

    return max(
        count / total,
        1e-12,
    )


def _probability(technique, ngram, models):
    order = len(ngram)

    # ---------------------------------------------------------
    # Interpolated Smoothing
    # ---------------------------------------------------------

    if technique == "interpolated":
        return interpolated_probability(
            ngram,
            models,
            [1.0 / order] * order,
        )

    # ---------------------------------------------------------
    # Good-Turing Smoothing
    # ---------------------------------------------------------

    if technique == "good_turing":

        model = models[order]

        return good_turing_probability(
            ngram,
            model,
            probability_table=GT_TABLES[order],
            lower_model=models[order - 1],
        )

    # ---------------------------------------------------------
    # Stupid Backoff
    # ---------------------------------------------------------

    if technique == "stupid_backoff":
        return stupid_backoff_probability(
            ngram,
            models,
        )

    # ---------------------------------------------------------
    # Kneser-Ney
    # ---------------------------------------------------------

    if technique == "kneser_ney":
        return KN_MODEL.probability(ngram)

    # ---------------------------------------------------------
    # Katz Backoff
    # ---------------------------------------------------------

    if technique == "katz":

        model = models[order]

        lower_model = models[order - 1]

        def lower_probability(word_tuple):
            word_id = word_tuple[-1]

            return unigram_probability(
                word_id,
                models,
            )

        return katz_probability(
            ngram,
            model,
            lower_model,
            lower_probability,
            probability_table=KATZ_TABLES[order],
        )

    raise ValueError(
        f"Unknown technique: {technique}"
    )


def calculate(
    technique,
    order,
    data_path,
    models=None,
    vocab=None,
):
    if models is None:
        models = load_all_models()

    if vocab is None:
        vocab = load_vocab()

    log_probability = 0.0
    N = 0

    with open(
        data_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            total=1000,
            desc=f"{technique} {order}-gram",
            unit="sent",
        ):

            ids = encode_sentence(
                line,
                vocab,
            )

            for i in range(
                len(ids) - order + 1
            ):

                ngram = tuple(
                    ids[i:i + order]
                )

                p = _probability(
                    technique,
                    ngram,
                    models,
                )

                p = max(
                    p,
                    1e-12,
                )

                log_probability += math.log(p)

                N += 1

    if N == 0:
        return float("inf")

    return math.exp(
        -log_probability / N
    )