import re

def sentence_tokenizer(text):
    pattern = r'(?<=[.!?])\s+'

    sentences = re.split(pattern, text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def word_tokenizer(text):
    pattern = r"""
        https?://\S+|
        [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|
        \d{1,2}/\d{1,2}/\d{2,4}|
        \d+\.\d+|
        \w+|
        [^\w\s]
    """

    tokens = re.findall(pattern, text, re.VERBOSE)

    return tokens