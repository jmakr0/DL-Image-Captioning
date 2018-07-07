from keras.engine.saving import load_model
from keras.layers import Input, Flatten, RepeatVector, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from keras.utils import multi_gpu_model

from src.coca.modules.custom_lstm import CustomLSTM
from src.settings.settings import Settings


def image_embedding(input_tensor, cnn='resnet152'):
    """
    Loads the specified convnet with pretrained weights.
    Currently does not support finetuning.

    :param input_tensor: Inbound keras tensor
    :param input_shape: Inbound shape
    :param cnn: string specifying convnet to be loaded
    :return: the loaded and configured model
    """

    if cnn == 'resnet152':
        from src.common.modules.resnet import ResNet152Embed as resnet
    elif cnn == 'resnet50':
        from keras.applications.resnet50 import ResNet50 as resnet
    else:
        raise ValueError("{} is not a supported cnn".format(cnn))

    base_model = resnet(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor
    )

    for layer in base_model.layers:
        layer.trainable = False

    if cnn == 'resnet50':
        x = GlobalAveragePooling2D()(base_model.output)
    else:
        x = Flatten(name='im_flatten')(base_model.output)

    return x


def language_model(input_tensor):
    settings = Settings()
    max_caption_len = settings.get_max_caption_length()
    word_embedding_size = settings.get_word_embedding_size()

    x = RepeatVector(max_caption_len)(input_tensor)
    x = CustomLSTM(1024)(x)
    x = Dense(word_embedding_size)(x)

    return x


def create_model(cnn, gpus=None, weights_path=None):
    settings = Settings()
    img_shape = settings.get_image_dimensions()

    img_input = Input(shape=img_shape, name='img_input')
    img_embedding = image_embedding(img_input, cnn=cnn)
    x = language_model(img_embedding)

    model = Model(inputs=img_input, outputs=x, name='img_cap')

    # load model weights to continue training
    if weights_path:
        model.load_weights(weights_path)

    if gpus and gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus, cpu_merge=True, cpu_relocation=False)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    return model


def load_coca_model(filename):
    return load_model(filename, custom_objects={
        "CustomLSTM": CustomLSTM
    })
