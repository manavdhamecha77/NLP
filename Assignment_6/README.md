# Assignment 6 – Smoothing Techniques

This assignment applies different smoothing techniques to the Unigram, Bigram, Trigram, and Quadgram language models developed in Assignment 4.

The following smoothing techniques are implemented:

- Interpolated Smoothing
- Good-Turing Smoothing
- Katz Backoff Smoothing
- Stupid Backoff Smoothing
- Kneser-Ney Smoothing

## Dataset

The same dataset and trained models from Assignment 4 are reused.

- Total sentences: 2,444,262
- Training sentences: 2,442,262
- Development sentences: 1,000
- Test sentences: 1,000
- Vocabulary size: 157,670

## Models

- Unigram
- Bigram
- Trigram
- Quadgram

## Smoothing Techniques

### Interpolated Smoothing

Interpolated smoothing combines probabilities from different n-gram orders:

```text
P(w_n | context) =
λ₁P(w_n) +
λ₂P(w_n | w_{n-1}) +
... +
λₙP(w_n | w_{n-n+1}, ..., w_{n-1})
```

Equal interpolation weights are used.

### Good-Turing Smoothing

For an observed n-gram count `c`, the adjusted count is:

```text
c* = (c + 1) × N_{c+1} / N_c
```

where `N_c` is the number of n-grams occurring exactly `c` times.

The implementation uses context-specific probability calculations for Bigram, Trigram, and Quadgram models.

### Katz Backoff Smoothing

For observed n-grams:

```text
P_Katz = d(c) × P_ML
```

For unseen n-grams, probability is obtained by backing off to a lower-order model:

```text
P_Katz = α(context) × P_Katz(lower-order)
```

Good-Turing discounting is used to calculate the discount values.

### Stupid Backoff Smoothing

If an n-gram is observed, its maximum-likelihood probability is used. Otherwise, the model backs off to a lower-order model using a fixed backoff factor:

```text
P = α × P(lower-order)
```

The backoff factor used is `α = 0.4`.

### Kneser-Ney Smoothing

Kneser-Ney smoothing uses continuation probabilities based on the number of distinct contexts in which a word occurs.

For higher-order models:

```text
P_KN(w | h) =
max(C(h,w) - D, 0) / C(h)
+
λ(h) × P_KN(w | suffix(h))
```

The discount value used is:

```text
D = 0.75
```

## Files

- `interpolated_smoothing.py` – Implements Interpolated Smoothing.
- `good_turing_smoothing.py` – Implements Good-Turing Smoothing.
- `katz_backoff.py` – Implements Katz Backoff Smoothing.
- `stupid_backoff.py` – Implements Stupid Backoff Smoothing.
- `kneser_ney.py` – Implements Kneser-Ney Smoothing.
- `perplexity.py` – Calculates development and test perplexity for each smoothing technique.
- `main.py` – Runs the evaluation for the smoothing techniques and n-gram models.

## Results

| Technique | Model | Development | Test |
|---|---|---:|---:|
| Interpolated | Bigram | 37.6679 | 37.0970 |
| Interpolated | Trigram | 19.3709 | 19.0532 |
| Interpolated | Quadgram | 13.3970 | 13.0957 |
| Good-Turing | Bigram | 38.9526 | 38.2748 |
| Good-Turing | Trigram | 15.4264 | 14.9429 |
| Good-Turing | Quadgram | 15.1195 | 14.3212 |
| Katz Backoff | Bigram | 38.2416 | 37.7223 |
| Katz Backoff | Trigram | 13.6433 | 13.3575 |
| Katz Backoff | Quadgram | 9.5060 | 9.1827 |
| Stupid Backoff | Bigram | 38.0765 | 37.5630 |
| Stupid Backoff | Trigram | 13.1937 | 12.9150 |
| Stupid Backoff | Quadgram | 8.7514 | 8.4608 |
| Kneser-Ney | Bigram | 27.7463 | 27.2089 |
| Kneser-Ney | Trigram | 13.3868 | 13.1087 |
| Kneser-Ney | Quadgram | 9.0299 | 8.7186 |

## Running

Run from inside the `Assignment_6` directory:

```bash
python main.py
```