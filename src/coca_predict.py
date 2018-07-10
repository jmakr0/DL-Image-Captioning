# hack to make parent dir (`src` available) for import, when calling this file directly
import json
import os
import sys

from keras.engine.saving import load_model

from src.common.dataloader.glove import Glove

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os

from src.common.dataloader.dataloader import TestSequence

from src.settings.settings import Settings

if __name__ == "__main__":

    settings = Settings()
    batch_size = 2
    model_dir = settings.get_path('models')
    model_path = os.path.join(model_dir, 'model.h5') # TODO: use correct filename

    output_dir = settings.get_path('output')
    output_path = os.path.join(output_dir, 'results.json')

    glove = Glove()
    glove.load_embedding()

    model = load_model(model_path)

    test_sequence = TestSequence(50)

    results = []
    for ids, images in test_sequence:
        predictions = model.predict_on_batch(images)

        for id_, capt in zip(ids, predictions):
            cap = ' '.join([glove.most_similar_word(embd_word) for embd_word in capt])  # delete end words
            results.append({
                "image_id": int(id_),
                "caption": cap
            })

    with open(output_path, "w") as fh:
        fh.write(json.dumps(results))
