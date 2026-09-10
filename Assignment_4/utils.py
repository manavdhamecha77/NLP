import os
import pickle
from collections import Counter

BOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"


def build_vocabulary(path):
    vocab = {
        UNK: 0,
        BOS: 1,
        EOS: 2
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                if word not in vocab:
                    vocab[word] = len(vocab)

    return vocab


def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def encode_sentence(line, vocab):
    words = line.split()

    ids = [vocab[BOS]]

    for word in words:
        ids.append(vocab.get(word, vocab[UNK]))

    ids.append(vocab[EOS])

    return ids


def get_bits(vocab):
    return max(1, (len(vocab) - 1).bit_length())


def pack_ngram(ids, bits):
    key = 0

    for x in ids:
        key = (key << bits) | x

    return key


def save_model(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)