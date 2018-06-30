from keras.layers import Input, Flatten
from keras.models import Model

def imageEmbedding(input_tensor, input_shape, cnn='resnet152'):
    """
    Loads the specified convnet with pretrained weights.
    Currently does not support finetuning.

    :param input_tensor: Inbound keras tensor
    :param input_shape: Inbound shape
    :param cnn: string specifying convnet to be loaded
    :return: the loaded and configured model
    """

    if cnn == 'resnet152':
        from modules.resnet import ResNet152Embed as base_model_c
    elif cnn == 'resnet50':
        from keras.applications.resnet50 import resnet50 as base_model_c
    else:
        raise ValueError("{} is not a supported cnn".format(cnn))

    base_model = base_model_c(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=input_shape
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten(name='im_flatten')(base_model.output)
    return Model(input=input_tensor, output=x, name="conv_{}".format(cnn))


def languageModel():
    pass


def model():
    img_shape = (224, 224, 3)
    img_input = Input(shape=img_shape, name='img_input')

    convnet = imageEmbedding(img_input, img_shape, cnn='resnet152')
    return convnet


if __name__ == "__main__":
    m = model()
    m.summary()
