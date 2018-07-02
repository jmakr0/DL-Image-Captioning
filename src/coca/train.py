from keras.utils import multi_gpu_model

from model import create_model
from dataloader.dataloader import DataLoader

from utils import common_callbacks


def train(log_dir, batch_size=64, multi_gpu=None):
    root_dir = '/data'

    args = {
        'capture_dir': root_dir + '/annotations',
        'train_images_dir': root_dir + '/train2014',
        'val_images_dir': root_dir + '/val2014'
    }

    dataloader = DataLoader(args)
    N_train = dataloader.get_dataset_size('train')
    N_val = dataloader.get_dataset_size('val')
    train_gen = dataloader.generator('train', batch_size)
    val_gen = dataloader.generator('val', batch_size)
    model = create_model()

    if multi_gpu:
        model = multi_gpu_model(model, gpus=multi_gpu, cpu_merge=True, cpu_relocation=False)

    model.summary()

    callbacks = common_callbacks(log_dir, batch_size=batch_size)

    model.fit_generator(train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=N_train/batch_size,
                        validation_steps=N_val/batch_size,
                        callbacks=callbacks,
                        verbose=1,
                        workers=2)


if __name__ == "__main__":
    train('./data/log', batch_size=64)
