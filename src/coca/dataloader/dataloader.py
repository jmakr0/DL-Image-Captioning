import json
import os
import random
import threading

import numpy as np
import keras
from keras_preprocessing.image import load_img

from src.coca.settings.settings import Settings
from .glove import Glove
from .image_loader import load_image


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)  # .next()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class DataLoader(object):
    def __init__(self):
        settings = Settings()
        self.annotations_dir = settings.get_path('annotations')
        self.train_images_dir = settings.get_path('train_images')
        self.validatoin_images_dir = settings.get_path('validation_images')

        self.word_embedding_size = settings.get_word_embedding_size()
        self.image_dimensions = settings.get_image_dimensions()
        self.max_caption_length = settings.get_max_caption_length()

        self.glove = Glove()
        self.glove.load_embedding()

        self.train_metadata = self._load_metadata('train')
        self.validation_metadata = self._load_metadata('val')

    def get_dataset_size(self, partition):
        if partition == 'train':
            return len(self.train_metadata)
        elif partition == 'val':
            return len(self.validation_metadata)
        else:
            raise ValueError

    def _load_metadata(self, partition):
        if partition != 'train' and partition != 'val':
            raise ValueError

        captions_filepath = os.path.join(self.annotations_dir, 'captions_{}2014.json'.format(partition))

        file = open(captions_filepath, 'r')
        data = json.load(file)

        annotations_raw = data['annotations']
        images_raw = data['images']

        images_metadata = self._images_metadata(images_raw)
        annotations = self._annotations(annotations_raw)

        result = []
        for id in images_metadata.keys():
            for annotation in annotations[id]:
                result.append((images_metadata[id], annotation))
        return result

    def _images_metadata(self, images_raw):
        images_metadata = {}
        for image in images_raw:
            img_data = {
                'filename': image['file_name'],
                'width': image['width'],
                'height': image['height']
            }
            images_metadata[image['id']] = img_data
        return images_metadata

    def _annotations(self, annotations_raw):
        annotations = {}
        for annotation in annotations_raw:
            if not annotation['image_id'] in annotations:
                annotations[annotation['image_id']] = [annotation['caption']]
            else:
                annotations[annotation['image_id']].append(annotation['caption'])
        return annotations

    @threadsafe_generator
    def generator(self, partition, batch_size, train_flag=True):
        if partition == 'train':
            images_dir = self.train_images_dir
            metadata = self.train_metadata
        elif partition == 'val':
            images_dir = self.validatoin_images_dir
            metadata = self.val_metadata
        else:
            raise ValueError

        batch_count = int(np.floor(len(metadata) / batch_size))

        while True:
            if train_flag:
                random.shuffle(metadata)

            for batch_number in range(batch_count):
                batch = metadata[batch_number * batch_size:batch_number * batch_size + batch_size]

                images = np.zeros(shape=(batch_size,) + self.image_dimensions)
                captions = np.zeros(shape=(batch_size, self.max_caption_length, self.word_embedding_size))

                for i, (image_metadata, caption) in enumerate(batch):
                    image_path = os.path.join(images_dir, image_metadata['filename'])

                    images[i] = load_image(image_path)
                    captions[i] = self.glove.embed_text(caption)

                yield (images, captions)

    def _load_image(self, file_path):
        if len(self.image_dimensions) == 2:
            image = load_img(file_path, target_size=(self.image_dimensions[0], self.image_dimensions[1]), grayscale=True)
        elif len(self.image_dimensions) == 3:
            image = load_img(file_path, target_size=(self.image_dimensions[0], self.image_dimensions[1]))
        else:
            raise ValueError('Image shape has to be 2 or 3 dimensional.')
        result = np.array(image, dtype=np.float)
        # normalize
        result = result / 255 * 2
        result = result - 1

        return result
