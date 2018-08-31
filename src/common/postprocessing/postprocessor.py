from src.common.dataloader.glove import Glove
from src.settings.settings import Settings


class Postprocessor:
    def __init__(self, one_hot=False):
        settings = Settings()
        self.stop_word = settings.get_stop_word()

        print("loading embedding")
        glove = Glove()
        glove.load_embedding()
        self.one_hot = one_hot
        self.glove = glove

    def translate_word_vectors(self, capt_vectors):
        if self.one_hot:
            return [self.glove.one_hot_to_word(one_hot_word) for one_hot_word in capt_vectors]
        return [self.glove.most_similar_word(embd_word) for embd_word in capt_vectors]

    def end_with_stopword(self, caption):
        if self.stop_word in caption:
            first_occurrence = caption.index(self.stop_word)
            return caption[:first_occurrence + 1]
        else:
            return caption

    def concat_words(self, words):
        return ' '.join([word for word in words])

    def build_caption(self, word_vectors):
        words = self.translate_word_vectors(word_vectors)
        words = self.end_with_stopword(words)
        caption = self.concat_words(words)
        return caption

    def captions(self, predictions):
        return [self.build_caption(word_vectors) for word_vectors in predictions]
