# Assignment 5 – Add-K Smoothing

This project applies Add-K smoothing with `K = 0.3` to the four language models developed in Assignment 4 and evaluates them on the development and test sets.

## Dataset

- Gujarati tokenized corpus from Assignment 4
- Training sentences: 2,442,262
- Development sentences: 1,000
- Test sentences: 1,000

## Techniques Used

- Unigram Language Model
- Bigram Language Model
- Trigram Language Model
- Quadgram Language Model
- Add-K Smoothing
- Perplexity

## Files

- `add_k_smoothing.py` – Implements Add-K smoothing with `K = 0.3`.
- `perplexity.py` – Calculates perplexity using Add-K smoothed probabilities.
- `main.py` – Evaluates all four models on the development and test sets.

## Running the Project

The trained models and development/test sets from Assignment 4 are reused.

```bash
python main.py
```
