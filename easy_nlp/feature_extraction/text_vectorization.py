from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


class Vectorizer:

    def fit_countVectorizer(corpus, stopWords="english", ngram_range=(2, 2)):
        vectorizer = CountVectorizer(
            stop_words=stopWords, ngram_range=ngram_range)
        X = vectorizer.fit_transform(corpus)
        return X

    def fit_countVectorizer(corpus, stopWords="english", ngram_range):
        vectorizer = TfidfVectorizer(
            stop_words=stopWords, ngram_range=ngram_range)
        X = vectorizer.fit_transform(corpus)
        return X
