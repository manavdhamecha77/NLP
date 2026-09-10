# Assignment 4 – N-Gram Language Models

This project implements four statistical language models using tokenized Gujarati text. The models include Unigram, Bigram, Trigram, and Quadrigram models with Add-1/Laplace smoothing and perplexity evaluation.

## Dataset

- Gujarati tokenized corpus
- 2,444,262 sentences
- Training: 2,442,262 sentences
- Development: 1,000 sentences
- Test: 1,000 sentences

## Techniques Used

- Unigram Language Model
- Bigram Language Model
- Trigram Language Model
- Quadrigram Language Model
- Add-1/Laplace Smoothing
- Perplexity
- Train/Development/Test Split

## Files

- `split.py` – Splits the tokenized corpus into training, development, and test sets.
- `unigram.py` – Implements the Unigram language model.
- `bigram.py` – Implements the Bigram language model.
- `trigram.py` – Implements the Trigram language model.
- `quadgram.py` – Implements the Quadrigram language model.
- `laplace_smoothing.py` – Implements Add-1/Laplace smoothing.
- `perplexity.py` – Calculates perplexity for all four models.
- `main.py` – Executes the complete language modeling pipeline.

## Data

The input file is:

```text
data/indic_tokens.txt
```