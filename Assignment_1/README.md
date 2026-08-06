# Assignment 1 – Text Preprocessing

This project implements a complete text preprocessing pipeline using the IndicCorpV2 and OSCAR datasets. The implementation includes sentence tokenization, word tokenization, corpus analysis, and storage of the processed data.

## Datasets

- IndicCorpV2 (Gujarati – `guj_Gujr`)
- OSCAR-2301 (English)

## Techniques Used

- Sentence tokenization
- Word tokenization
- Regular expressions (Regex)
- Dataset streaming
- Corpus statistics
- Parquet file generation

## Files

- `download_indic.py` – Downloads and stores the IndicCorpV2 dataset.
- `download_oscar.py` – Downloads and stores the OSCAR dataset.
- `tokenizer.py` – Implements sentence and word tokenizers.
- `statistics.py` – Calculates corpus statistics and the Type-Token Ratio (TTR).
- `main.py` – Executes the complete preprocessing pipeline.

## Running the Project

```bash
pip install -r requirements.txt

python download_indic.py
python download_oscar.py

python main.py
```