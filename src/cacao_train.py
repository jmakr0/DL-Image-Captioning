import argparse
import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from keras import backend as K

from src.cacao.model import image_captioning_model
from src.common.callbacks import common_callbacks
from src.common.dataloader.dataloader import DataLoader
from src.settings.settings import Settings


def train():
    """
    ToDo:
     * DataLoader also with proper Settings integration
     * DataLoader proper generator
     * Model: settings Glove.DIMENSIONS
     * Model: MAX_LEN_CAPTION -> use settings
     * Train: final_submission. Use validation set to train on.
     * Model: Save functionality
     * Look into callback functions
     * Model: include multi-gpu support, if not already given by Tensorflow.
     * DataGenerator Validation: look into train-flag
     * Adjust argument lines.
     * Create shell scripts.
     * Test net on my machine. Let it train.
     * Test the output.
     * Make it work with nvidia-docker
     * Test on server.
     * Make predictions on test set provided by the chair.
     * Refine model: BachNorm Layer, different LossFunctions, tune parameters/optimizers.
     * Print out loss and create diagrams.
     * More models with different behaviors. maybe one parametrized model definition
     * Experiments on all models. Also include the CustomLSTM Layer variant.
     * Think about project structure.

     * max caption len in train data
     * consider allowing training of last layers
     * Review: clip attention
    """
    K.set_learning_phase(1)

    dataloader = DataLoader()
    train_dataset_size = dataloader.get_dataset_size('train')
    validation_dataset_size = dataloader.get_dataset_size('val')
    train_generator = dataloader.generator('train', args.batch_size)
    validation_generator = dataloader.generator('val', args.batch_size, train_flag=False)

    config = Settings()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices)
    model = image_captioning_model(lr=args.lr, cnn=args.cnn, gpus=args.devices, img_shape=config.get_image_dimensions(),
                                   embedding_dim=config.get_word_embedding_size(),
                                   max_caption_length=config.get_max_caption_length())
    model.summary()

    callbacks = common_callbacks(batch_size=args.batch_size)

    model.fit_generator(train_generator,
                        epochs=args.epochs,
                        validation_data=validation_generator,
                        steps_per_epoch=train_dataset_size / args.batch_size,
                        validation_steps=validation_dataset_size / args.batch_size,
                        callbacks=callbacks,
                        verbose=1,
                        workers=2)

    model_dir = config.get_path('models')
    model_path = os.path.join(model_dir, f'model{args.exp_id}_{args.cnn}_{args.batch_size}_{args.epochs}.model')

    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--exp_id', default=np.random.randint(20, 1000), type=int)
    parser.add_argument('--settings', default='C:/repo/DL-Image-Captioning/evaluation/cacao/settings_axel.yml')
    parser.add_argument('--devices', default=[0], type=int, nargs='*', help='GPUs to use')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--cnn', default='resnet50', choices=['resnet50', 'resnet152'])
    parser.add_argument('--lr', default=3e-3, type=float)

    parser.add_argument('--final_submission', default='False', choices=['True', 'False'])
    args = parser.parse_args()

    if not os.path.isfile(args.settings):
        raise FileNotFoundError(f'Settings under {args.settings} do not exist.')
    Settings.FILE = args.settings

    train()
