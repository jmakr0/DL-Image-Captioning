import os
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint

from model import create_model
from dataloader.dataloader import DataLoader


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

    # create log folder
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    checkpoints_dir = os.path.join(log_dir, 'model_checkpoints')
    csv_log_file = os.path.join(log_dir, 'metrics_log.csv')

    try:
        os.makedirs(tensorboard_dir)
    except OSError:
        pass
    try:
        os.makedirs(checkpoints_dir)
    except OSError:
        pass

    callbacks = [
        TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False),
        CSVLogger(csv_log_file),
        EarlyStopping(monitor='val_loss', patience=5, mode='auto', baseline=None),
        ModelCheckpoint(os.path.join(
            checkpoints_dir,
            "weights.{epoch:02d}-{val_loss:2.f}.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True)
    ]

    model.fit_generator(train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=N_train/batch_size,
                        validation_steps=N_val/batch_size,
                        callbacks=callbacks,
                        verbose=1,
                        workers=2)


if __name__ == "__main__":
    train(batch_size=64)
