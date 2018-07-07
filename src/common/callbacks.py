import os

from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint

from src.settings.settings import Settings


def common_callbacks(batch_size=64, make_dirs=True):
    settings = Settings()
    log_dir = settings.get_path('logs')

    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    checkpoints_dir = os.path.join(log_dir, 'model_checkpoints')
    csv_log_file = os.path.join(log_dir, 'metrics_log.csv')

    # create log folders
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
            "weights.{epoch:02d}-{val_loss:.4f}.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True),
        ModelCheckpoint(os.path.join(
            checkpoints_dir,
            "model.{epoch:02d}-{val_loss:.4f}.pkl"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False)
    ]
    return callbacks
