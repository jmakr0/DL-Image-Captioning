import json
import os
import random

import numpy as np
from keras.utils import Sequence
from keras_preprocessing.image import load_img

from src.common.dataloader.glove import Glove
from src.settings.settings import Settings


class DataLoadingSequence(Sequence):

    def __init__(self, partition, batch_size, input_caption=False, shuffle=False):
        if partition != 'train' and partition != 'val' and partition != 'test':
            raise ValueError("partition `{}` is not valid. Either specify `train` or `val`".format(partition))

        self.partition = partition
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_caption = input_caption

        settings = Settings()
        self.word_embedding_size = settings.get_word_embedding_size()
        self.max_caption_length = settings.get_max_caption_length()

        if self.test_mode:
            self.annotations_dir = settings.get_path('test_metadata')
        else:
            self.annotations_dir = settings.get_path('annotations')

            self.glove = Glove()
            self.glove.load_embedding()

        self.image_dimensions = settings.get_image_dimensions()
        self.images_dir = settings.get_path("{}_images".format(partition))

        self._load_metadata()

    @property
    def test_mode(self):
        return self.partition == 'test'

    def _load_metadata(self):
        if self.test_mode:
            metadata_filepath = os.path.join(self.annotations_dir, 'input.json')
        else:
            metadata_filepath = os.path.join(self.annotations_dir, 'captions_{}2014.json'.format(self.partition))

        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)

        images = {}
        for image in metadata['images']:
            images[image['id']] = {'id': image['id'], 'file_name': image['file_name']}

        if not self.test_mode:
            for annotation in metadata['annotations']:
                images[annotation['image_id']]['caption'] = annotation['caption']

        self.metadata = [value for _, value in images.items()]

        if self.shuffle:
            random.shuffle(self.metadata)

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, index):
        bs = self.batch_size
        batch = self.metadata[index * bs:(index + 1) * bs]

        ids = []
        images = np.zeros(shape=(bs,) + self.image_dimensions)
        captions = np.zeros(shape=(bs, self.max_caption_length, self.word_embedding_size))

        for i, metadata in enumerate(batch):
            ids.append(metadata['id'])
            image_path = os.path.join(self.images_dir, metadata['file_name'])
            try:
                images[i] = self._get_image(image_path)
            except FileNotFoundError:
                print("image not found: {}".format(metadata['id']))
                print("  use `python src/common/filter_metadata.py --input IN_JSON --output OUT_JSON --negative_image_ids {}`".format(metadata['id']))
                print("  to clean your metadata file")
                print("missing images lead to batches with different sizes!")
                break

            if not self.test_mode:
                captions[i] = self.glove.embed_text(metadata['caption'])

        if self.test_mode:
            return (images, ids) if self.input_caption is False else (ids, [images, captions])
        else:
            return (images, captions) if self.input_caption is False else ([images, captions], captions)

    def _get_image(self, image_path):
        if len(self.image_dimensions) == 2:
            image = load_img(image_path, target_size=(self.image_dimensions[0], self.image_dimensions[1]),
                             grayscale=True)
        elif len(self.image_dimensions) == 3:
            image = load_img(image_path, target_size=(self.image_dimensions[0], self.image_dimensions[1]))
        else:
            raise ValueError('Image shape has to be 2 or 3 dimensional.')

        image = np.array(image, dtype=np.float)
        # normalize
        image = image / 255 * 2
        image = image - 1

        return image


class TrainSequence(DataLoadingSequence):
    def __init__(self, batch_size, input_caption=False):
        super().__init__('train', batch_size, input_caption=input_caption, shuffle=True)


class ValSequence(DataLoadingSequence):
    def __init__(self, batch_size, input_caption=False):
        super().__init__('val', batch_size, input_caption=input_caption, shuffle=True)


class TestSequence(DataLoadingSequence):
    def __init__(self, batch_size, input_caption=False):
        super().__init__('test', batch_size, input_caption=input_caption, shuffle=False)
