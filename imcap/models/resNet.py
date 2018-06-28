from keras import backend as K
from keras.layers import Convolution2D, BatchNormalization, Activation, ZeroPadding2D, Input, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense
from keras.layers import merge as Merge
from keras.models import Model

from imcap.layers.bnScale import BNScale


def identity_block(input_tensor, kernel_size, filters, bn_axis, stage, block):
    """The identity_block is the block that has no conv layer at shortcut

    Adopted from https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = Merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, bn_axis, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    Adopted from https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = BNScale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = Merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def resnet152_conv_layers(eps):
    """Creates the ResNet152 architecture without the fully connected layers at the end.

    Adopted from https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6

    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    """

    # Handle Dimension Ordering for different backends
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = BNScale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], bn_axis, stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], bn_axis, stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], bn_axis, stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], bn_axis, stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='c')

    return img_input, x


class ResNet152(Model):
    """Initiates ResNet152 architecture
    # Arguments
            weights_path: path to pretrained weight file
            eps: fuzzy parameter for batch normalization
    """

    def __init__(self, weights_path, eps=1.1e-5):
        self.eps = eps

        img_input, x = resnet152_conv_layers(self.eps)
        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

        super().__init__(img_input, x_fc)

        self.load_weights(weights_path)


class ResNet152Finetune(Model):
    """Initiates ResNet152 architecture
    # Arguments
            weights_path: path to pretrained weight file
            num_classes: number of output neurons / classes
            eps: fuzzy parameter for batch normalization
    """

    def __init__(self, weights_path, num_classes, eps=1.1e-5):
        self.eps = eps

        img_input, x = resnet152_conv_layers(self.eps)
        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

        model = Model(img_input, x_fc)

        # load weights
        if weights_path:
            model.load_weights(weights_path, by_name=True)

        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_newfc = Flatten()(x_newfc)
        x_newfc = Dense(num_classes, activation='softmax', name='fc_new_{}'.format(num_classes))(x_newfc)

        super().__init__(img_input, x_newfc)
