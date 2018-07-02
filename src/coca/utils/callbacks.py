import os

from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint


def common_callbacks(log_dir, batch_size=64, make_dirs=True):
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    checkpoints_dir = os.path.join(log_dir, 'model_checkpoints')
    csv_log_file = os.path.join(log_dir, 'metrics_log.csv')

    # create log folder
    if make_dirs:
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
            batch_size=batch_size,
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
    return callbacks
