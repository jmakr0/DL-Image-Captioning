from keras import backend as K
from keras import Model, Input
from keras.applications import ResNet50 as resnet50
from keras.layers import Dense, LSTM, Multiply, Concatenate, Reshape, Lambda, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from src.common.modules.resnet import ResNet152Embed as resnet152


def image_captioning_model(img_shape=(224, 224, 3), cnn='resnet152', embedding_dim=50, max_caption_length=15,
                           gpus=None, lr=3e-3):

    # Definition of CNN
    cnn_input = Input(shape=img_shape)
    cnn = eval(cnn)(
            include_top=False,
            weights='imagenet',
            input_tensor=cnn_input,
            input_shape=img_shape
    )
    for layer in cnn.layers:
        layer.trainable = False
    cnn_output = (Flatten() if cnn == 'resnet152' else GlobalAveragePooling2D())(cnn.output)
    cnn_output_len = int(cnn_output.shape[-1])

    # Caption Input
    caption_input = Input((max_caption_length, embedding_dim))

    # Definition of RNN
    rnn = LSTM(666, return_sequences=False, return_state=True)
    attention_layer = Dense(cnn_output_len, activation='relu')
    embedding_layer = Dense(embedding_dim, activation='relu')

    # Start vars
    def constant(input_batch, size):
        batch_size = K.shape(input_batch)[0]
        return K.tile(K.ones((1, size)), (batch_size, 1))
    embd_word = Lambda(constant, arguments={'size': embedding_dim})(cnn_input)
    attention = Lambda(constant, arguments={'size': cnn_output_len})(cnn_input)
    state = None

    words = []
    for i in range(max_caption_length):
        attention_image = Multiply()([cnn_output, attention])
        rnn_in = Concatenate()([embd_word, attention_image])

        rnn_in = Reshape((1, embedding_dim + cnn_output_len))(rnn_in)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)
        state = (hidden_state, cell_state)

        embd_word = embedding_layer(rnn_out)
        attention = attention_layer(rnn_out)

        embd_word_concat = Reshape((1, embedding_dim))(embd_word)
        words.append(embd_word_concat)
        if K.learning_phase():
            embd_word = Lambda(lambda x, ii: x[:, ii], arguments={'ii': i})(caption_input)

    caption = Concatenate(axis=1)(words)

    # Assemble Model
    model = Model(inputs=[cnn_input, caption_input], outputs=caption)
    if len(gpus) >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['mae', 'acc'])
    return model
