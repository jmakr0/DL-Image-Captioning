from keras import backend as K
from keras import Model, Input
from keras.applications import ResNet50
from keras.layers import Dense, LSTM, Multiply, Concatenate, Reshape, Lambda
from keras.optimizers import Adam


# from src.coca.modules.resnet import ResNet152Embed
from src.common.dataloader.glove import Glove

MAX_LEN_CAPTION = 15


def image_captioning_model(lr=3e-3):
    # Definition of CNN
    cnn = ResNet50(weights='imagenet')
    cnn.layers.pop()  # remove classification layer
    cnn_input = cnn.input

    # image_shape = (224, 224, 3)
    # cnn_input = Input(shape=image_shape, name='img_input')
    # cnn = ResNet152Embed(
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=cnn_input,
    #     input_shape=image_shape
    # )

    # BatchNorm, Flatten, etc?
    # cnn_output = cnn.output
    # cnn_output = BatchNormalization(axis=-1)(cnn_output)
    # cnn_output = Flatten(name='im_flatten')(cnn_output)

    # consider allowing training of last layers
    for layer in cnn.layers:
        layer.trainable = False

    cnn_output_len = cnn.layers[-1].output_shape[-1]
    cnn_output = cnn.layers[-1].output
    batch_size = K.shape(cnn.input)[0]

    # Caption Input. Consider supporting all five of them.
    caption_input = Input((MAX_LEN_CAPTION, Glove.DIMENSIONS))

    # Definition of RNN
    rnn = LSTM(666, return_sequences=False, return_state=True)
    attention_layer = Dense(cnn_output_len, activation='relu')
    embedding_layer = Dense(Glove.DIMENSIONS, activation='relu')

    emd_word_start = Input(tensor=K.zeros((1, Glove.DIMENSIONS)))
    emd_word = Lambda(lambda x: K.tile(x, (batch_size, 1)))(emd_word_start)
    attention_start = Input(tensor=K.ones((1, cnn_output_len)))
    attention = Lambda(lambda x: K.tile(x, (batch_size, 1)))(attention_start)
    state = None

    caption = []
    for i in range(MAX_LEN_CAPTION):
        attention_image = Multiply()([cnn_output, attention])  # Review: clip attention
        rnn_in = Concatenate()([emd_word, attention_image])

        rnn_in = Reshape((1, Glove.DIMENSIONS + cnn_output_len))(rnn_in)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)
        state = (hidden_state, cell_state)

        emd_word = embedding_layer(rnn_out)
        attention = attention_layer(rnn_out)

        caption.append(emd_word)
        if K.learning_phase():
            emd_word = Lambda(lambda x: x[:, i], arguments={'i': i})(caption_input)
    caption = Concatenate(axis=0)(caption)

    model = Model(inputs=[cnn_input, caption_input, attention_start, emd_word_start], outputs=caption)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['accuracy'])
