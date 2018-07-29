import argparse
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras import backend as K

from src.cacao.model import image_captioning_model
from src.common.callbacks import common_callbacks
from src.common.dataloader.dataloader import TrainSequence, ValSequence
from src.settings.settings import Settings


def train():
    """
    ToDo:
     * test resnet152
     * Train: final_submission. Use flickr30k to train on.
     * Create shell scripts.
     * Test net on my machine. Let it train.
     * Test the output.
     * Make it work with nvidia-docker
     * Test on server.
     * Make predictions on test set provided by the chair.
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

    model = image_captioning_model(lr=args.lr, cnn=args.cnn, gpus=len(args.devices), img_shape=config.get_image_dimensions(),
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

    model.save_weights(model_path + '_weights')
    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--exp_id', default=str(np.random.randint(20, 1000)), type=str)
    parser.add_argument('--settings', default=None, type=str)
    parser.add_argument('--devices', default=[], type=int, nargs='*', help='GPUs to use')
    parser.add_argument('--cnn', default='resnet50', choices=['resnet50', 'resnet152'])
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--workers', default=4, type=int)

    # parser.add_argument('--final_submission', default='False', choices=['True', 'False'])
    args = parser.parse_args()

    if not os.path.isfile(args.settings):
        raise FileNotFoundError(f'Settings under {args.settings} do not exist.')
    Settings.FILE = args.settings

    train()
