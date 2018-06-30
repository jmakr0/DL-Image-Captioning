from keras.layers import Input
from keras.optimizers import SGD

from modules.resnet import ResNet152Embed


def main():
    # Input tensor
    img_input = Input((224, 224, 3), name='img_input')

    # Compile cnn model
    model = ResNet152Embed(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input
    )
    model.trainable = False

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()


if __name__ == "__main__":
    main()
