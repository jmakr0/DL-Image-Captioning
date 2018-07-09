# hack to make parent dir (`src` available) for import, when calling this file directly
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json

from argparse import ArgumentParser
from keras import backend as K
from keras.models import load_model

from src.common.dataloader.dataloader import TestSequence
from src.common.dataloader.glove import Glove

from src.coca.modules.custom_lstm import CustomLSTM
from src.settings.settings import Settings


def predict():
    K.set_learning_phase(0)

    glove = Glove()
    glove.load_embedding()

    model = load_model(args.model_path, custom_objects={'CustomLSTM': CustomLSTM})
    test_sequence = TestSequence(args.batch_size)

    results = []
    for ids, images in test_sequence:
        predictions = model.predict_on_batch(images)

        for id_, capt in zip(ids, predictions):
            cap = ' '.join([glove.most_similar_word(embd_word) for embd_word in capt])  # delete end words
            results.append({
                "image_id": int(id_),
                "caption": cap
            })

    with open(args.output_path, "w") as fh:
        fh.write(json.dumps(results))


if __name__ == "__main__":
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--model_path', default='C:/repo/DL-Image-Captioning/coca_model_resnet50_64_30.model', type=str)
    arg_parse.add_argument('--output_path', default='C:/repo/DL-Image-Captioning/output.json', type=str)
    arg_parse.add_argument('--settings', default='C:/repo/DL-Image-Captioning/evaluation/settings_axel.yml', type=str)
    arg_parse.add_argument('--batch_size', type=int, default=64)
    args = arg_parse.parse_args()

    Settings.FILE = args.settings

    predict()
