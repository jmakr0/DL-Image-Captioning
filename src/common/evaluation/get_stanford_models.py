import os
import subprocess


def get_stanford_models():
    # get stanford nltk data
    download_script = os.path.join('.',
                                   os.path.dirname(__file__),
                                   'get_stanford_models.sh')
    return subprocess.call([download_script])


if __name__ == "__main__":
    get_stanford_models()
