# hack to make parent dir (`src` available) for import, when calling this file directly
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os

from src.common.dataloader.dataloader import TestSequence
from src.coca.model import create_model

from src.settings.settings import Settings

if __name__ == "__main__":

    settings = Settings()
    batch_size = 2
    weights_dir = settings.get_path('weights')
    weights_path = os.path.join(weights_dir, 'weights.h5')

    output_dir = settings.get_path('output')
    output_path = os.path.join(output_dir, 'results.json')

    model = create_model('resnet50', weights_path=weights_path)

    test_sequence = TestSequence(batch_size)

    test_sequence = test_sequence[:10]

    prediction = model.predict(test_sequence)

    with open(weights_path, 'w') as f:
        f.write(str(prediction))
