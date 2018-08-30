# hack to make parent dir (`src` available) for import, when calling this file directly
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json

from argparse import ArgumentParser
from keras import backend as K
from keras.models import load_model

from src.common.dataloader.dataloader import TestSequence
from src.common.postprocessing.postprocessor import Postprocessor

from src.settings.settings import Settings


def predict(args):
    K.set_learning_phase(0)

    postprocessor = Postprocessor(dictionary_size=14831, one_hot=True)

    print("loading model")
    model = load_model(args.model_path)

    print("beginning prediction on batches of size {}".format(args.batch_size))
    test_sequence = TestSequence(args.batch_size, input_caption=True, dictionary_size=14831)

    results = []
    for i in range(len(test_sequence)):
        ids, images = test_sequence[i]
        predictions = model.predict_on_batch(images)

        for id_, word_vectors in zip(ids, predictions):
            caption_string = postprocessor.build_caption(word_vectors)
            results.append({
                "image_id": int(id_),
                "caption": caption_string
            })

    print("saving results to file")
    with open(args.output_path, "w") as fh:
        fh.write(json.dumps(results))


if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Uses a saved model to generate captions for pictures in a batch. " +
                                           "Writes the results into a JSON-file.")
    arg_parse.add_argument('--batch_size',
                           type=int,
                           default=8,
                           help="batch size to make the predictions")
    arg_parse.add_argument('model_path',
                           type=str,
                           help="filepath of the saved `cacao` model to use for generating captions")
    arg_parse.add_argument('output_path',
                           type=str,
                           help="filepath, where the output JSON should be written to")
    arg_parse.add_argument('settings',
                           type=str,
                           help="filepath to the configuration file, that was also used for training!")
    arguments = arg_parse.parse_args()

    if not os.path.isfile(arguments.settings):
        raise FileNotFoundError('Settings under {} do not exist.'.format(arguments.settings))
    Settings.FILE = arguments.settings

    predict(arguments)
