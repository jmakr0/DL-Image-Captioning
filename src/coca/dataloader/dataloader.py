import json
import os
import random
import threading

import numpy as np

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

    def __init__(self, args_dict):

        self.capture_dir = args_dict['capture_dir']
        self.train_images_dir = args_dict['train_images_dir']
        self.val_images_dir = args_dict['val_images_dir']

        self.glove = Glove()
        self.glove.load_embedding()

    def _load_metadata(self, partition):
        result = []

        captions_filepath = os.path.join(self.capture_dir, 'captions_' + partition + '2014.json')

        file = open(captions_filepath, 'r')
        data = json.load(file)

        annotations_raw = data['annotations']
        images_raw = data['images']

        images_metadata = {}

        for image in images_raw:
            img_data = {}
            img_data['filename'] = image['file_name']
            img_data['width'] = image['width']
            img_data['height'] = image['height']

            images_metadata[image['id']] = img_data

        annotations = {}

        for annotation in annotations_raw:
            if not annotation['image_id'] in annotations:
                annotations[annotation['image_id']] = [annotation['caption']]
            else:
                annotations[annotation['image_id']].append(annotation['caption'])

        for id in list(images_metadata.keys()):
            img_meta = images_metadata[id]
            captions = annotations[id]

            result.append((img_meta, captions))

        return result

    def _create_image_caption_pairs(self, metadata):
        '''
        takes metadata in form of (image_info, [caption_1, caption_2, ...])
        and returns list of tupels of form (image_info, caption_n)
        '''
        result = []

        image_info, captions = metadata

        for caption in captions:
            result.append((image_info, caption))

        return result

    def _transform_metadata(self, metadata_list):
        '''
        takes a list of metainfos [(img_info_1, [cap_1_1, cap_1_2,...]), (img_info_2, [cap_2_1, cap_2_2,...]), ...]
        and returns [(img_info_1, cap_1_1), (img_info_1, cap_1_2), (img_info_2, cap_2_1)]
        '''

        result = []

        for meta in metadata_list:
            transformed_meta = self._create_image_caption_pairs(meta)
            result += transformed_meta

        return result

    def get_dataset_size(self, partition):
        metadata = self._load_metadata(partition)
        metadata = self._transform_metadata(metadata)
        return len(metadata)


    @threadsafe_generator
    def generator(self, partition, batch_size, train_flag=True):
        if partition == "train":
            images_dir = self.train_images_dir
        else:
            images_dir = self.val_images_dir

        metadata = self._load_metadata('train')
        metadata = self._transform_metadata(metadata)

        # image_filenames = [f for f in listdir(images_dir) if isfile(join(images_dir, f)) and f.endswith('.jpg')]
        rng = int(np.floor(len(metadata) / batch_size))

        while True:

            if train_flag:
                random.shuffle(metadata)

            for i in range(rng):
                batch = metadata[i * batch_size:i * batch_size + batch_size]

                transformed_batch = []

                for image_meta, caption in batch:
                    image_path = os.path.join(images_dir, image_meta['filename'])

                    image = load_image(image_path)

                    # create list with word embeddings
                    embedded_caption = self.glove.embedd_string_sequence(caption)

                    transformed_batch.append([image, embedded_caption])

                yield transformed_batch
