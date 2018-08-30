import yaml
from os import path


class Settings:
    FILE = path.join(path.dirname(__file__), 'settings.yml')

    def __init__(self):
        with open(self.FILE, 'r') as yaml_file:
            self.config = yaml.load(yaml_file)

    def get_glove_embedding(self):
        return self.config.get('embeddings')['glove']

    def get_path(self, name):
        return self.config.get('paths')[name]

    def get_image_dimensions(self):
        dim = self.config.get('image_dimensions')
        return int(dim[0]), int(dim[1]), int(dim[2])

    def get_word_embedding_size(self):
        return int(self.config.get('word_embedding_size'))

    def get_dictionary_size(self):
        return int(self.config.get('dictionary_size'))

    def get_max_caption_length(self):
        return int(self.config.get('max_caption_length'))

    def get_stop_word(self):
        return self.config.get('stop_word')

    def get_spice_dirs(self):
        return self.config.get('spice')
