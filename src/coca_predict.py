import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras.engine.saving import load_model

from src.coca.modules.custom_lstm import CustomLSTM
from src.common.dataloader.glove import Glove
from src.common.dataloader.dataloader import TestSequence, TrainSequence
from src.settings.settings import Settings

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    settings = Settings()
    model_dir = settings.get_path('models')
    model_path = os.path.join(model_dir, 'model_resnet50_64_50.model')

    output_dir = settings.get_path('output')
    output_path = os.path.join(output_dir, 'results.json')

    glove = Glove()
    glove.load_embedding()

    custom_objects = {'CustomLSTM': CustomLSTM}

    model = load_model(model_path, custom_objects=custom_objects)

    test_sequence = TestSequence(64)

    results = []
    for i in range(len(test_sequence)):
        images, ids = test_sequence[i]
        predictions = model.predict_on_batch(images)

        for prediction, image_id in zip(predictions, ids):
            results.append({
                'image_id': image_id,
                'caption': ' '.join([glove.most_similar_word(embedding) for embedding in prediction])
            })

    with open(output_path, "w") as fh:
        fh.write(json.dumps(results, indent=2))
