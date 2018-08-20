from src.settings.settings import Settings


class Postprocessor:

    def __init__(self):
        settings = Settings()
        self.stop_word = settings.get_stop_word()

    def crop_caption(self, caption):
        if self.stop_word in caption:
            first_occurrence = caption.index(self.stop_word)
            return caption[:first_occurrence + 1]
        else:
            return caption