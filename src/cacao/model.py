from keras import backend as K
from keras import Model, Input
from keras.applications import ResNet50 as resnet50
from keras.layers import Dense, LSTM, Multiply, Concatenate, Reshape, Lambda, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from src.common.modules.resnet import ResNet152Embed as resnet152


def image_captioning_model(lr=3e-3, cnn='resnet152', gpus=None, img_shape=(224, 224, 3),
                           embedding_dim=50,
                           max_caption_length=15):

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

    # Caption Input.
    caption_input = Input((max_caption_length, embedding_dim))

    # Vars
    cnn_output_len = int(cnn_output.shape[-1])
    batch_size = K.shape(cnn_input)[0]

    # Definition of RNN
    rnn = LSTM(666, return_sequences=False, return_state=True)
    attention_layer = Dense(cnn_output_len, activation='relu')
    embedding_layer = Dense(embedding_dim, activation='relu')

    emd_word_start = Input(tensor=K.zeros((1, embedding_dim)))
    emd_word = Lambda(lambda x: K.tile(x, (batch_size, 1)))(emd_word_start)
    attention_start = Input(tensor=K.ones((1, cnn_output_len)))
    attention = Lambda(lambda x: K.tile(x, (batch_size, 1)))(attention_start)
    state = None

    caption = []
    for i in range(max_caption_length):
        attention_image = Multiply()([cnn_output, attention])
        rnn_in = Concatenate()([emd_word, attention_image])

        rnn_in = Reshape((1, embedding_dim + cnn_output_len))(rnn_in)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)
        state = (hidden_state, cell_state)

        emd_word = embedding_layer(rnn_out)
        attention = attention_layer(rnn_out)

        caption.append(emd_word)
        if K.learning_phase():
            emd_word = Lambda(lambda x, ii: x[:, ii], arguments={'ii': i})(caption_input)
    caption = Concatenate(axis=0)(caption)

    # Assemble Model
    model = Model(inputs=[cnn_input, caption_input, attention_start, emd_word_start], outputs=caption)
    if len(gpus) >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['mae', 'acc'])
    return model
