# hack to make parent dir (`src` available) for import, when calling this file directly
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json

from argparse import ArgumentParser
from keras import backend as K
from keras.models import load_model

from src.common.dataloader.dataloader import TestSequence
from src.common.dataloader.glove import Glove

from src.settings.settings import Settings


def predict():
    K.set_learning_phase(0)
    config = Settings()

    print("loading embedding")
    glove = Glove(dictionary_size=40000)
    glove.load_embedding()

    print("loading model")
    model = load_model(args.model_path)

    print("beginning prediction on batches of size {}".format(args.batch_size))
    test_sequence = TestSequence(args.batch_size, input_caption=True)

    results = []
    for ids, images in test_sequence:
        predictions = model.predict_on_batch(images)

        for id_, capt in zip(ids, predictions):
            cap = ' '.join([glove.most_similar_word(embd_word) for embd_word in capt])  # delete end words
            results.append({
                "image_id": int(id_),
                "caption": cap
            })

    print("saving results to file")
    with open(args.output_path, "w") as fh:
        fh.write(json.dumps(results))


if __name__ == "__main__":
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--model_path', default='C:/repo/DL-Image-Captioning/model3_resnet50_64_55.model', type=str)
    arg_parse.add_argument('--output_path', default='C:/repo/DL-Image-Captioning/output_cacao_3.json', type=str)
    arg_parse.add_argument('--settings', default='C:/repo/DL-Image-Captioning/evaluation/settings_axel.yml', type=str)
    arg_parse.add_argument('--batch_size', type=int, default=3)
    args = arg_parse.parse_args()

    Settings.FILE = args.settings

    predict()
