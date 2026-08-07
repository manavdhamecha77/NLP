import re

def sentence_tokenizer(text):
    pattern = r'(?<=[.!?])\s+' # Everything before [.?!] (positive look-behind)

    sentences = re.split(pattern, text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def word_tokenizer(text):
    pattern = r"""
        https?://\S+|                                   # URL
        [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}| # Email
        \d{1,2}/\d{1,2}/\d{2,4}|                        # Date
        \d+\.\d+|                                       # Decimal
        \w+|                                            # Words
        [^\w\s]                                         # Special Characters/Punctuations
    """

    tokens = re.findall(pattern, text, re.VERBOSE)

    return tokens