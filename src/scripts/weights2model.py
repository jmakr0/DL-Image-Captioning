import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from argparse import ArgumentParser

from src.cacao.model import image_captioning_model
from src.settings.settings import Settings

from keras import backend as K


def main(weights_file, model_file, cnn, save_plot):
    config = Settings()

    print("loading `cacao` model weights from file {}".format(weights_file))
    K.set_learning_phase(1)
    model = image_captioning_model(
        lr=1e-3,
        cnn=cnn,
        gpus=1,
        img_shape=config.get_image_dimensions(),
        embedding_dim=config.get_word_embedding_size(),
        max_caption_length=config.get_max_caption_length()
    )
    model.load_weights(weights_file)
    model.summary()

    if save_plot:
        print("Option `-plot")
        from keras.utils import plot_model
        plot_model(model, "cacao_model.png")

    print("saving model to {}".format(model_file))
    model.save(model_file)


if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Converts a weights file to a model file for " +
                                           "a `cacao` model.")
    arg_parse.add_argument('-p', '--plot',
                           action='store_true',
                           help="plots the model's architecture as picture to `cacao_model.png`")
    arg_parse.add_argument('--cnn',
                           type=str,
                           default='resnet50',
                           choices=['resnet50', 'resnet152'],
                           help="cnn used in the model that produced the weights file, default = `resnet50`")
    arg_parse.add_argument('input',
                           type=str,
                           help="filepath to the weights file")
    arg_parse.add_argument('output',
                           type=str,
                           help="filepath, where the output model file should be saved to")
    arg_parse.add_argument('settings',
                           type=str,
                           help="filepath to the settings file to use")
    args = arg_parse.parse_args()

    if not os.path.isfile(args.settings):
        raise FileNotFoundError('Settings under {} do not exist.'.format(arguments.settings))
    Settings.FILE = args.settings

    main(args.input, args.output, args.cnn, args.plot)
