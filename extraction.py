from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import nltk
from nltk import pos_tag, ne_chunk
import re

nltk.download('popular')


class NounExtract():
    def __init__(self):
        # Initialising CountVectorizer and Tfidf Vectorizer
        self.count_vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')

    # Preprocessing the text data by removing digits and other expressions
    def preprocess(self, data):
        data = re.sub(r'[(.*?_)]+', '', data)
        data = re.sub(r'\d+', '', data)
        return data

    # Takes list of sentences and return the bigrams and trigrams in a list
    def ngrams(self, data):
        self.count_vectorizer.fit_transform(data)
        word_list = self.count_vectorizer.get_feature_names()
        return word_list

    # Takes a list of words and returns the noun chunks
    def noun_extract(self, word_list):
        noun_chunks = []
        word = ne_chunk(pos_tag(word_list))

        for elt in word:
            if isinstance(elt, nltk.Tree):
                for w, t in elt:
                    if t.startswith('NN'):
                        noun_chunks.append(w)
            elif elt[1].startswith('NN'):
                noun_chunks.append(elt[0])
        return noun_chunks

    # Takes the noun chunks and sorts the Top noun chunks in descending order
    def top_n_words(self, chunks):
        top_noun = []
        self.tfidf_vectorizer.fit_transform(chunks)
        sorted_features = np.argsort(self.tfidf_vectorizer.idf_)
        features = self.tfidf_vectorizer.get_feature_names()
        for f in sorted_features:
            if f in top_noun:
                continue
            else:
                top_noun.append(features[f])

        return top_noun

    # Takes the noun chunks and removes all duplications if present
    def final_processing(self, word_list):
        word_vocab = []
        final_words = []
        word_list=self.noun_extract(word_list)
        for w in word_list:
            c = 0
            split_words = w.split()
            for s in split_words:
                if s in word_vocab:
                    break
                else:
                    c += 1
                    word_vocab.append(s)
            if c == len(split_words):
                final_words.append(w)

        return final_words
