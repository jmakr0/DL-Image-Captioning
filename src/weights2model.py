import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cacao.model import image_captioning_model
from src.settings.settings import Settings

from keras import backend as K
from keras.utils import plot_model


if __name__ == "__main__":
    weights_file = "/dl_data/run5cacao/381model_checkpoints/weights.07-0.0048.h5"
    model_file = "cacao_model.pkl"
    Settings.FILE = "settings/settings-sebastian.yml"
    config = Settings()

    K.set_learning_phase(1)
    model = image_captioning_model(
        lr=1e-3,
        cnn='resnet50',
        gpus=1,
        img_shape=config.get_image_dimensions(),
        embedding_dim=config.get_word_embedding_size(),
        max_caption_length=config.get_max_caption_length()
    )
    model.load_weights(weights_file)
    model.summary()
    plot_model(model, "cacao_model.png")
    model.save(model_file)
