import keras.backend as K
from keras import Model
from keras.applications import ResNet50
from keras.layers import Flatten, AveragePooling2D, Dense, LSTM, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model

from src.common.modules.resnet import ResNet152Embed


def image_embedding(cnn_input, cnn_name='resnet50', img_shape=(224, 224, 3)):
    cnn_switcher = {
        'resnet50': ResNet50,
        'resnet152': ResNet152Embed
    }

    # Definition of CNN
    cnn = cnn_switcher[cnn_name](
        include_top=False,
        weights='imagenet',
        input_tensor=cnn_input,
        input_shape=img_shape
    )
    for layer in cnn.layers:
        layer.trainable = False

    if cnn_name == 'resnet152':
        cnn_output = Flatten(name="final_cnn_flatten")(cnn.output)
    else:
        cnn_output = Flatten(name="final_cnn_flatten")(AveragePooling2D((7, 7), name="final_cnn_pool")(cnn.output))
    cnn_output_len = int(cnn.output.shape[-1])
    return cnn_output, cnn_output_len


def constant_ones_tensor(size):
    def constant(input_batch):
        batch_size = K.shape(input_batch)[0]
        return K.tile(K.ones((1, size)), (batch_size, 1))
    return constant


def dense_attention_layer(nodes, l2_reg=1e-8):
    return Dense(nodes, activation='sigmoid', kernel_regularizer=l2(l2_reg), name="attention")


def lstm_generator(nodes, dropout=.0, l2_reg=1e-8):
    return LSTM(nodes,
                name="word_embd_rnn",
                return_sequences=False, return_state=True,
                dropout=dropout, recurrent_dropout=dropout,
                recurrent_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg),
                kernel_regularizer=l2(l2_reg))


def three_layer_word_embedding(nodes1, nodes2, nodes3, l2_reg=1e-8):
    embedding_layer1 = Dense(nodes1, activation='relu', kernel_regularizer=l2(l2_reg), name="word_embd_1")
    embedding_layer2 = Dense(nodes2, activation='relu', kernel_regularizer=l2(l2_reg), name="word_embd_2")
    embedding_layer3 = Dense(nodes3, activation='tanh', kernel_regularizer=l2(l2_reg), name="word_embd_3")
    return lambda input_tensor: embedding_layer3(embedding_layer2(embedding_layer1(input_tensor)))


def single_layer_word_embedding(nodes, l2_reg=1e-8):
    return Dense(nodes, activation='tanh', kernel_regularizer=l2(l2_reg), name="word_embd")


def replace_embedding_word_during_training(index):
    def get_word_with_index(x):
        return x[:, index]
    return Lambda(get_word_with_index, name="train_word_{}".format(index))


def create_compile_model(inputs, outputs, gpus=0, lr=1e-3, loss='mean_squared_error', name="image_captioning_model"):
    model = Model(inputs=inputs, outputs=outputs, name=name)
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['mae'])
    return model

