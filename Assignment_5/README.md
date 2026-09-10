# Assignment 5 – Add-K Smoothing

This assignment applies Add-K smoothing to Unigram, Bigram, Trigram, and Quadgram language models developed in Assignment 4.

The value of K used is 0.3.

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

## Add-K Smoothing

The Add-K probability is calculated as:

P(w_n | context) =
(C(context, w_n) + K) /
(C(context) + K × V)

where:

- K = 0.3
- C(context, w_n) = n-gram count
- C(context) = context count
- V = vocabulary size

## Files

- `add_k_smoothing.py` – Implements Add-K smoothing with K = 0.3.
- `perplexity.py` – Calculates development and test perplexity.
- `main.py` – Runs the evaluation for all four models.

## Running

Run from inside the Assignment_5 directory:

```bash
python main.py
```