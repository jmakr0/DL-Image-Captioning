import numpy as np

from src.common.dataloader import matutils
from src.settings.settings import Settings


class Glove:
    def __init__(self, dictionary_size=40000):
        self.dictionary_size = dictionary_size

        settings = Settings()
        self.embedding_path = settings.get_glove_embedding()
        self.word_embedding_size = settings.get_word_embedding_size()
        self.max_caption_length = settings.get_max_caption_length()

        self.word_numbers = {}
        self.embedding_vectors = np.zeros((dictionary_size, self.word_embedding_size))
        self.words = {}
        self.vectors_norm = None

    def load_embedding(self):
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line_split = line.strip().split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype='float32')

                # we skip the head but start the index with 0
                self.word_numbers[word] = line_number-1
                self.embedding_vectors[line_number-1] = vector
                self.words[line_number-1] = word

                if line_number-1 == self.dictionary_size:
                    break

        # vector norm is lazy (gets computed at first call to most_similar_word()
        self.vectors_norm = None

    def index_of_word(self, word):
        if len(self.word_numbers) == 0:
            raise Exception('No embedding loaded.')

        word = word.strip().lower()
        if word in self.word_numbers:
            return self.word_numbers[word]
        return 0

    def to_vector(self, word):
        return self.embedding_vectors[self.index_of_word(word)]

    def _text_to_word_sequence(self, text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, ' ') for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = text.split()
        return [word for word in seq if word]

    def text_to_word_indices(self, text, limit=None):
        indices = []

        if isinstance(text, list):
            seq = text
        else:
            seq = self._text_to_word_sequence(text)

        if limit:
            limit = min(len(seq), limit)
        else:
            limit = len(seq)

        for i in range(limit):
            indices.append(self.index_of_word(seq[i]))

        return indices

    def embed_text(self, string):
        index_sequence = self.text_to_word_indices(string, limit=self.max_caption_length)

        word_vectors = self.embedding_vectors[index_sequence]
        # add zero padding
        zero_vector = np.zeros(shape=(self.max_caption_length, self.word_embedding_size))
        zero_vector[:len(word_vectors)] = word_vectors

        return zero_vector

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
