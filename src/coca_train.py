# hack to make parent dir (`src` available) for import, when calling this file directly
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
from argparse import ArgumentParser

from src.common.dataloader.dataloader import TrainSequence, ValSequence
from src.coca.model import create_model, load_coca_model

from src.common.callbacks import common_callbacks

from src.settings.settings import Settings


def train(cnn, batch_size, epochs, devices=None, workers=4, mw_path=None):
    # get train and val dataset loader
    train_sequence = TrainSequence(batch_size)
    val_sequence = ValSequence(batch_size)

    gpus = 0
    if devices is not None:  # this lets one specify the device "0" (would lead to `false` with `if devices:`)
        devices = [str(d) for d in devices]
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(devices)
        gpus = len(devices)
        print("CUDA devices specified: {}, using {} gpus".format(devices, gpus))

    import tensorflow as tf
    from tensorflow.python import debug as tf_debug
    import keras

    keras.backend.set_session(
        tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:1212"))

    # if weights_path is a pickled model --> load it
    try:
        model = load_coca_model(mw_path)
    # otherwise: create model and if weights_path is a weights file load the weights into it
    except (ValueError, TypeError):
        model = create_model(cnn, gpus=gpus, weights_path=mw_path)

    model.summary()

    callbacks = common_callbacks(batch_size=batch_size)

    # steps_per_epoch, validation_steps derived from Sequence
    model.fit_generator(train_sequence,
                        epochs=epochs,
                        validation_data=val_sequence,
                        callbacks=callbacks,
                        verbose=1,
                        workers=workers,                # workers consuming the sequence
                        use_multiprocessing=True,       # uses multiprocessing instead of multithreading
                        max_queue_size=20)              # generator queue size

    settings = Settings()
    model_dir = settings.get_path('models')
    model_path = os.path.join(model_dir, 'model_{}_{}_{}.model'.format(cnn, batch_size, epochs))

    model.save(model_path)


if __name__ == "__main__":

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--cnn', type=str, default='resnet50', choices=['resnet50', 'resnet152'])
    arg_parse.add_argument('--batch_size', type=int, default=64)
    arg_parse.add_argument('--epochs', type=int, default=50)
    arg_parse.add_argument('--devices', type=int, nargs='*')
    arg_parse.add_argument('--workers', type=int, default=4)
    arg_parse.add_argument('--settings_yml', type=str, default=None)
    arg_parse.add_argument('--continue_from', type=str, default=None)

    arguments = arg_parse.parse_args()

    if arguments.settings_yml:
        # overwrite settings file path
        # hacky!!
        yml_file = arguments.settings_yml
        print("Using settings file path: {}".format(yml_file))
        if not os.path.isfile(yml_file):
            raise FileNotFoundError("{} is not a file".format(yml_file))
        Settings.FILE = yml_file

    if arguments.continue_from:
        # check for file existence early
        continue_from = arguments.continue_from
        print("Using weights from {} to continue training".format(continue_from))
        if not os.path.isfile(continue_from):
            raise FileNotFoundError("{} is not a file".format(continue_from))

    train(arguments.cnn, arguments.batch_size, arguments.epochs, arguments.devices, arguments.workers, arguments.continue_from)
