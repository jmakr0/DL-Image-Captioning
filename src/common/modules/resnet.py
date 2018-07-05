from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, Input, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense, add, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model

from src.common.modules.layers.bnScale import BNScale
from src.common.modules.weights_downloader import download_ResNet152_weights_tf, download_ResNet152_weights_th


def _identity_block(input_tensor, kernel_size, filters, bn_axis, stage, block):
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

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def _conv_block(input_tensor, kernel_size, filters, bn_axis, stage, block, strides=(2, 2)):
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

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BNScale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = BNScale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def _resnet152_conv_layers(img_input, eps):
    """Creates the ResNet152 architecture without the fully connected layers at the end.
    Adopted from https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6
    # Arguments
        img_input: keras tensor as image input for the model
    # Returns
        A Keras model instance.
    """

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), name="conv1", strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = BNScale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = _conv_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='a', strides=(1, 1))
    x = _identity_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='b')
    x = _identity_block(x, 3, [64, 64, 256], bn_axis, stage=2, block='c')

    x = _conv_block(x, 3, [128, 128, 512], bn_axis, stage=3, block='a')
    for i in range(1, 8):
        x = _identity_block(x, 3, [128, 128, 512], bn_axis, stage=3, block='b' + str(i))

    x = _conv_block(x, 3, [256, 256, 1024], bn_axis, stage=4, block='a')
    for i in range(1, 36):
        x = _identity_block(x, 3, [256, 256, 1024], bn_axis, stage=4, block='b' + str(i))

    x = _conv_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='a')
    x = _identity_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='b')
    x = _identity_block(x, 3, [512, 512, 2048], bn_axis, stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    return x


def ResNet152(weights_path, eps=1.1e-5):
    """Initiates ResNet152 architecture
    # Arguments
            weights_path: path to pretrained weight file
            eps: fuzzy parameter for batch normalization
    """

    eps = eps

    # Handle Dimension Ordering for different backends
    if K.image_dim_ordering() == 'tf':
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        img_input = Input(shape=(3, 224, 224), name='data')

    x = _resnet152_conv_layers(img_input, eps)
    x_fc = Flatten()(x)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc, name='resnet152')
    model.load_weights(weights_path)

    return model


def ResNet152Finetune(weights_path, num_classes, eps=1.1e-5):
    """Initiates ResNet152 architecture
    # Arguments
            weights_path: path to pretrained weight file
            num_classes: number of output neurons / classes
            eps: fuzzy parameter for batch normalization
    """

    eps = eps

    # Handle Dimension Ordering for different backends
    if K.image_dim_ordering() == 'tf':
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        img_input = Input(shape=(3, 224, 224), name='data')

    x = _resnet152_conv_layers(img_input, eps)
    x_fc = Flatten()(x)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = Flatten()(x)
    x_newfc = Dense(num_classes, activation='softmax', name='fc_new_{}'.format(num_classes))(x_newfc)

    return Model(img_input, x_newfc, name='resnet152')


def ResNet152Embed(include_top=True, weights='imagenet',
                   input_tensor=None, input_shape=None,
                   pooling=None, classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    eps = 1.1e-5

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _resnet152_conv_layers(img_input, eps)
    x_fc = Flatten()(x)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if weights == 'imagenet':
        model = Model(inputs, x_fc)

        # load weights
        if K.backend() == 'theano':
            weights_path = download_ResNet152_weights_th()
        else:
            weights_path = download_ResNet152_weights_tf()

        model.load_weights(weights_path, by_name=True)

    if include_top:
        # We only have weights for the whole model, so we have to drop layers if we want to exclude top.
        # Cannot use model.layers.pop() since model is not of Sequential() type.
        # The method below works since pre-trained weights are stored in layers but not in the model.
        x_newfc = x_fc
    else:
        if pooling == 'avg':
            x_newfc = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x_newfc = GlobalMaxPooling2D()(x)
        else:
            x_newfc = x

    return Model(inputs, x_newfc, name='resnet152')
