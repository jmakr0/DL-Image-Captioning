import sys

import numpy as np
from scipy import spatial

from src.settings.settings import Settings


class Glove:
    def __init__(self, dictionary_size=40000):
        self.dictionary_size = dictionary_size

        settings = Settings()
        self.embedding_path = settings.get_glove_embedding()
        self.word_embedding_size = settings.get_word_embedding_size()
        self.max_caption_length = settings.get_max_caption_length()

        self.word_numbers = {}
        self.embedding_vectors = np.zeros((dictionary_size + 1, self.word_embedding_size))
        self.words = {}

    def load_embedding(self):
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line_split = line.strip().split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype=float)

                self.word_numbers[word] = line_number
                self.embedding_vectors[line_number] = vector
                self.words[line_number] = word

                if line_number == self.dictionary_size:
                    break

    def get_word_number(self, word):
        if len(self.word_numbers) == 0:
            raise Exception('No embedding loaded.')

        word = word.lower()
        if word in self.word_numbers:
            return self.word_numbers[word]
        return 0

    def get_word_vector(self, word):
        return self.embedding_vectors[self.get_word_number(word)]

    def _text_to_word_sequence(self, text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, ' ') for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = text.split()
        return [word for word in seq if word]

    def text_to_word_indices(self, text, limit=None):
        vector = []

        if isinstance(text, list):
            seq = text
        else:
            seq = self._text_to_word_sequence(text)

        if limit:
            limit = min(len(seq), limit)
        else:
            limit = len(seq)

        for i in range(limit):
            word = seq[i]
            vector.append(self.get_word_number(word))

        return vector

    def embed_text(self, string):
        index_sequence = self.text_to_word_indices(string, limit=self.max_caption_length)

        embedded_words = self.embedding_vectors[index_sequence]
        zero_vector = np.zeros(shape=(self.max_caption_length, self.word_embedding_size))
        zero_vector[:len(embedded_words)] = embedded_words

        return zero_vector

    def wordembedding_to_most_similar_word(self, v_embedding):
        most_similar_word = None
        min_diff = sys.maxsize

        # if runtime too bad, see: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
        for word_number in self.word_numbers.values():
            v_compare_embedding = self.embedding_vectors[word_number]

            diff = spatial.distance.cosine(v_embedding, v_compare_embedding)
            if diff < min_diff:
                most_similar_word = self.words[word_number]
                min_diff = diff

        return most_similar_word
