def corpus_statistics(tokenized_sentences):

    total_sentences = len(tokenized_sentences)

    all_words = []

    for sentence in tokenized_sentences:
        all_words.extend(sentence)

    total_words = len(all_words)

    total_characters = sum(
        len(word)
        for word in all_words
    )

    average_sentence_length = (
        total_words / total_sentences
    )

    average_word_length = (
        total_characters / total_words
    )

    ttr = len(set(all_words)) / total_words

    return {
        "Total Sentences": total_sentences,
        "Total Words": total_words,
        "Total Characters": total_characters,
        "Average Sentence Length": average_sentence_length,
        "Average Word Length": average_word_length,
        "TTR": ttr
    }