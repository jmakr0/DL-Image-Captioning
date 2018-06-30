import yaml


class Settings:
    FILE = 'settings.yml'

    def __init__(self):
        with open(self.FILE, 'r') as yaml_file:
            self.config = yaml.load(yaml_file)

    def get_glove_embedding(self):
        return self.config.get('embeddings')['glove']
