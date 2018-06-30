import numpy as np

from keras.preprocessing.image import load_img


def load_image(file_path):
    image = load_img(file_path, target_size=(224, 224))
    result = np.array(image, dtype=np.float)
    # normalize
    result = result / 255 * 2
    result = result - 1

    return result
