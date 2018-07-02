import os
from argparse import ArgumentParser

from dataloader.dataloader import DataLoader
from model import create_model

from utils import common_callbacks


def train(cnn, batch_size, epochs, devices=None):
    dataloader = DataLoader()

    train_dataset_size = dataloader.get_dataset_size('train')
    validation_dataset_size = dataloader.get_dataset_size('val')

    train_generator = dataloader.generator('train', batch_size)
    validation_generator = dataloader.generator('val', batch_size, train_flag=False)

    if devices:
        devices = [str(d) for d in devices]
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(devices)

    model = create_model(cnn, gpus=len(devices))
    model.summary()

    callbacks = common_callbacks(batch_size=batch_size)

    model.fit_generator(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        steps_per_epoch=train_dataset_size / batch_size,
                        validation_steps=validation_dataset_size / batch_size,
                        callbacks=callbacks,
                        verbose=1,
                        workers=2)


if __name__ == "__main__":

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--cnn', type=str, default='resnet50', choices=['resnet50', 'resnet152'])
    arg_parse.add_argument('--batch_size', type=int, default=64)
    arg_parse.add_argument('--epochs', type=int, default=50)
    arg_parse.add_argument('--devices', type=int, nargs='*')

    arguments = arg_parse.parse_args()

    train(arguments.cnn, arguments.batch_size, arguments.epochs, arguments.devices)
