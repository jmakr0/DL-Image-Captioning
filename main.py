from keras import backend as K

from src.coca.model import image_captioning_model
from src.coca.dataloader.dataloader import DataLoader


def main():

    K.set_learning_phase(0)
    batch_size = 2
    model = image_captioning_model(batch_size)
    root_dir = '/data'
    dl_args = {'capture_dir': root_dir + '/annotations',
               'train_images_dir': root_dir + '/train2014',
               'val_images_dir': root_dir + '/val2014'}

    dataloader = DataLoader(dl_args)
    N_train = dataloader.get_dataset_size('train')
    N_val = dataloader.get_dataset_size('val')
    train_gen = dataloader.generator('train', batch_size)
    val_gen = dataloader.generator('val', batch_size)

    # model.fit_generator(train_gen, validation_data=val_gen,
    #                     steps_per_epoch=N_train/batch_size,
    #                     validation_steps=N_val/batch_size,
    #                     verbose=1,
    #                     workers=2)
    # epochs etc.

    for batch in train_gen:
        print(batch)
        break

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Say hello')
    # parser.add_argument('X_path', help='path to images')
    # parser.add_argument('y_path', help='path to captions')
    # parser.add_argument('X_path_val', help='path to val images')
    # parser.add_argument('y_path_val', help='path to val captions')
    #
    # parser.add_argument('stuff, lr, optimizer, etc.')
    # args = parser.parse_args()
    main()
