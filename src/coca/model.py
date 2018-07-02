from keras.layers import Input, Flatten, RepeatVector, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from modules.custom_lstm import CustomLSTM


def image_embedding(input_tensor, input_shape, cnn='resnet152'):
    """
    Loads the specified convnet with pretrained weights.
    Currently does not support finetuning.

    :param input_tensor: Inbound keras tensor
    :param input_shape: Inbound shape
    :param cnn: string specifying convnet to be loaded
    :return: the loaded and configured model
    """

    if cnn == 'resnet152':
        from modules.resnet import ResNet152Embed as resnet
    elif cnn == 'resnet50':
        from keras.applications.resnet50 import ResNet50 as resnet
    else:
        raise ValueError("{} is not a supported cnn".format(cnn))

    base_model = resnet(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=input_shape
    )

    for layer in base_model.layers:
        layer.trainable = False

    if cnn == 'resnet50':
        x = GlobalAveragePooling2D()(base_model.output)
        # x = Flatten(name='im_flatten')(x)
    else:
        x = Flatten(name='im_flatten')(base_model.output)
    return x


def language_model(input_tensor, max_caption_len, word_embedding_size):
    x = RepeatVector(max_caption_len)(input_tensor)
    x = CustomLSTM(1024)(x)
    x = Dense(word_embedding_size)(x)

    return x


def create_model(max_caption_len, gpus=None):
    img_shape = (224, 224, 3)
    img_input = Input(shape=img_shape, name='img_input')

    # x = image_embedding(img_input, img_shape, cnn='resnet152')
    x = image_embedding(img_input, img_shape, cnn='resnet50')
    x = language_model(x, max_caption_len, 50)

    model = Model(input=img_input, output=x, name='img_cap')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
    if gpus:
        model = multi_gpu_model(model, gpus=gpus, cpu_merge=True, cpu_relocation=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    return model
