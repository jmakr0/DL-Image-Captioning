import os

from keras.utils import multi_gpu_model

from model import create_model
from dataloader.dataloader import DataLoader


def train(batch_size=64, multi_gpu=None):
    max_caption_length = 25

    dataloader = DataLoader(max_caption_length)

    train_dataset_size = dataloader.get_dataset_size('train')
    validation_dataset_size = dataloader.get_dataset_size('val')

    train_generator = dataloader.generator('train', batch_size)
    validation_generator = dataloader.generator('val', batch_size, train_flag=False)
    model = create_model(max_caption_length, gpus=multi_gpu)

    model.summary()

    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=train_dataset_size / batch_size,
                        validation_steps=validation_dataset_size / batch_size,
                        verbose=1,
                        workers=2)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    train(batch_size=64, multi_gpu=2)
