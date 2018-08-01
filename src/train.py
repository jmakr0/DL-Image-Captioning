import argparse
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras import backend as K

from src.cacao.model import image_captioning_model
from src.common.callbacks import common_callbacks
from src.common.dataloader.dataloader import TrainSequence, ValSequence
from src.settings.settings import Settings


def train(args):
    """
    ToDo:
     * Refine model: BachNorm Layer, different LossFunctions, tune parameters/optimizers.
     * Print out loss and create diagrams.
     * More models with different behaviors. maybe one parametrized model definition
     * Experiments on all models.
     * Think about project structure.

     * max caption len in train data -> 37
     * Review: clip attention
    """
    K.set_learning_phase(1)

    train_sequence = TrainSequence(args.batch_size, input_caption=True)
    val_sequence = ValSequence(args.batch_size, input_caption=True)

    config = Settings()
    callbacks = common_callbacks(batch_size=args.batch_size, exp_id=args.exp_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(map(str, args.devices))

    model = image_captioning_model(lr=args.lr, cnn=args.cnn, gpus=len(args.devices),
                                   img_shape=config.get_image_dimensions(),
                                   embedding_dim=config.get_word_embedding_size(),
                                   max_caption_length=config.get_max_caption_length())
    model.summary()
    model.fit_generator(train_sequence,
                        epochs=args.epochs,
                        validation_data=val_sequence,
                        callbacks=callbacks,
                        verbose=1,
                        workers=args.workers,
                        use_multiprocessing=False,
                        max_queue_size=20)

    model_dir = config.get_path('models')
    model_path = os.path.join(model_dir, f'model{args.exp_id}_{args.cnn}_{args.batch_size}_{args.epochs}.model')

    model.save_weights(model_path + '.weights')
    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts the training of the `cacao` model')

    parser.add_argument('--exp_id',
                        type=str,
                        default=str(np.random.randint(20, 1000)),
                        help="experiment ID for saving logs")
    parser.add_argument('--devices',
                        type=int, nargs='*',
                        default=[],
                        help='IDs of the GPUs to use, starting from 0')
    parser.add_argument('--cnn',
                        type=str, choices=['resnet50', 'resnet152'],
                        default='resnet50',
                        help="type of CNN to use as image feature extractor")
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help="trainings stops after this number of epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="batch size as power of 2; if multiple GPUs are used the batch is divided between them")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="learning rate for the optimization algorithm")
    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help="number of worker-threads to use for data preprocessing and loading")
    parser.add_argument('settings',
                        type=str,
                        help="filepath to the configuration file")

    # parser.add_argument('--final_submission', default='False', choices=['True', 'False'])
    arguments = parser.parse_args()

    if not os.path.isfile(arguments.settings):
        raise FileNotFoundError(f'Settings under {arguments.settings} do not exist.')
    Settings.FILE = arguments.settings

    train(arguments)
