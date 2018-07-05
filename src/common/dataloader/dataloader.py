import json
import os
import random

import numpy as np
from keras.utils import Sequence
from keras_preprocessing.image import load_img

from src.common.dataloader.glove import Glove
from src.settings.settings import Settings


class DataLoadingSequence(Sequence):

    def __init__(self, partition, batch_size, shuffle=False):
        if partition != 'train' and partition != 'val':
            raise ValueError("partition `{}` is not valid. Either specify `train` or `val`".format(partition))

        settings = Settings()
        self.annotations_dir = settings.get_path('annotations')
        self.images_dir = settings.get_path("{}_images".format(partition))

        self.word_embedding_size = settings.get_word_embedding_size()
        self.image_dimensions = settings.get_image_dimensions()
        self.max_caption_length = settings.get_max_caption_length()

        self.glove = Glove()
        self.glove.load_embedding()

        self.metadata = self._load_metadata(partition)
        if partition == 'train' and shuffle:
            random.shuffle(self.metadata)

        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.metadata) / float(self.batch_size)))

    def __getitem__(self, index):
        bs = self.batch_size
        batch = self.metadata[index * bs:(index+1) * bs]

        images = np.zeros(shape=(bs,) + self.image_dimensions)
        captions = np.zeros(shape=(bs, self.max_caption_length, self.word_embedding_size))

        for i, (image_metadata, caption) in enumerate(batch):
            image_path = os.path.join(self.images_dir, image_metadata['filename'])
            images[i] = self._load_image(image_path)
            captions[i] = self.glove.embed_text(caption)

        return images, captions

    def _load_metadata(self, partition):
        captions_filepath = os.path.join(self.annotations_dir, 'captions_{}2014.json'.format(partition))

        with open(captions_filepath, 'r') as file:
            data = json.load(file)

        annotations_raw = data['annotations']
        images_raw = data['images']

        images_metadata = self._images_metadata(images_raw)
        annotations = self._annotations(annotations_raw)

        result = []
        for annotation_id in images_metadata.keys():
            for annotation in annotations[annotation_id]:
                result.append((images_metadata[annotation_id], annotation))
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


class TrainSequence(DataLoadingSequence):

    def __init__(self, batch_size):
        super().__init__('train', batch_size)


class ValSequence(DataLoadingSequence):

    def __init__(self, batch_size):
        super().__init__('val', batch_size, shuffle=False)
