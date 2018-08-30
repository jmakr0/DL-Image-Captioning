import sys

import numpy as np
from keras.utils import to_categorical
from scipy import spatial

from src.common.dataloader import matutils
from src.settings.settings import Settings


class Glove:
    def __init__(self, dictionary_size=400000):
        self.dictionary_size = dictionary_size

        settings = Settings()
        self.embedding_path = settings.get_glove_embedding()
        self.word_embedding_size = settings.get_word_embedding_size()
        self.max_caption_length = settings.get_max_caption_length()
        self.stop_word = settings.get_stop_word()

        self.word_numbers = {}
        self.embedding_vectors = np.zeros((dictionary_size, self.word_embedding_size))
        self.words = {}
        self.stop_word_vector = None
        self.one_hot_stop_word_vector = None
        self.vectors_norm = None

    def load_embedding(self):
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line_split = line.strip().split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype=float)

                self.word_numbers[word] = line_number-1
                self.embedding_vectors[line_number-1] = vector
                self.words[line_number-1] = word

                if line_number-1 == self.dictionary_size:
                    break

        self.stop_word_vector = self.get_word_vector(self.stop_word)
        self.one_hot_stop_word_vector = to_categorical(self.get_word_number(self.stop_word),
                                                       num_classes=self.dictionary_size)

        self.vectors_norm = None

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

    def one_hot_vector(self, string):
        index_sequence = self.text_to_word_indices(string, limit=self.max_caption_length)
        fix_len_indices = np.tile(self.one_hot_stop_word_vector, (self.max_caption_length, 1))
        fix_len_indices[:len(index_sequence)] = to_categorical(index_sequence, num_classes=self.dictionary_size)
        return fix_len_indices

    def embed_text(self, string):
        index_sequence = self.text_to_word_indices(string, limit=self.max_caption_length)

        embedded_words = self.embedding_vectors[index_sequence]
        fix_len_embeddings = np.full((self.max_caption_length, self.word_embedding_size), self.stop_word_vector)
        fix_len_embeddings[:len(embedded_words)] = embedded_words

        return fix_len_embeddings

    def wordembedding_to_most_similar_word(self, v_embedding):
        # use numpy-version from gensim instead:
        return self.most_similar_word(v_embedding)
        """
        most_similar_word = None
        min_diff = sys.maxsize

        # lookup for exact match (numpy vectorized operation)
        indices = np.flatnonzero((self.embedding_vectors == v_embedding).all(1))
        if len(indices) > 0:
            print("found early matches: {}".format(", ".join([self.words[i] for i in indices])))
            # just return first match
            return self.words[indices[0]]

        # if runtime too bad, see: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
        for word_number in self.word_numbers.values():
            v_compare_embedding = self.embedding_vectors[word_number]

            diff = spatial.distance.cosine(v_embedding, v_compare_embedding)
            if diff < min_diff:
                most_similar_word = self.words[word_number]
                min_diff = diff

        return most_similar_word
        """

    def init_sims(self):
        if getattr(self, 'vectors_norm', None) is None:
            self.vectors_norm = (
                    self.embedding_vectors / np.sqrt((self.embedding_vectors ** 2).sum(-1))[..., np.newaxis]
            ).astype(np.float32)

    def most_similar_word(self, embedding):
        """
        see: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
        """
        self.init_sims()

        unit_vector = matutils.unitvec(embedding).astype(np.float32)
        dists = np.dot(self.vectors_norm, unit_vector)
        nearest = np.argmax(dists)
        return self.words[nearest]

    def one_hot_to_word(self, one_hot_word):
        return self.words[np.argmax(one_hot_word)]
