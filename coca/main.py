import cv2
import numpy as np
from keras.optimizers import SGD

from coca.modules import ResNet152


def main():

    import sys
    sys.setrecursionlimit(3000)

    im = cv2.resize(cv2.imread('../cat.jpg'), (224, 224)).astype(np.float32)

    # Remove train image mean
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    # Use pre-trained weights for Tensorflow backend
    weights_path = '../models/resnet152/resnet152_weights_tf.h5'

    # Insert a new dimension for the batch_size
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = ResNet152(weights_path)
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    preds = model.predict(im)
    print(np.argmax(preds))


if __name__ == "__main__":
    main()
