from keras import Input
from keras import backend as K
from keras.layers import Dense, Concatenate, Reshape

import src.cacao.model_commons as C


def image_captioning_model_raw(img_shape=(224, 224, 3), cnn='resnet152', embedding_dim=50, max_caption_length=15,
                               gpus=0, lr=1e-3, regularizer=1e-8, dropout=0.2):
    # Definition of CNN
    cnn_input = Input(shape=img_shape)
    cnn_output, cnn_output_len = C.image_embedding(cnn_input, cnn, img_shape)

    # Caption Input
    caption_input = Input((max_caption_length, embedding_dim))

    # Definition of RNN
    rnn = C.lstm_generator(50, dropout=dropout, l2_reg=regularizer)

    # Auxiliary layers
    reshape_rnn_in = Reshape((1, embedding_dim))
    reshape_embd_word_for_concat = Reshape((1, embedding_dim))

    # Start vars
    embd_word = Dense(embedding_dim, activation='relu')(cnn_output)  # learn first word

    state = None
    words = []

    for i in range(max_caption_length):
        rnn_in = reshape_rnn_in(embd_word)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)

        # update vars
        embd_word = rnn_out

        state = [hidden_state, cell_state]
        words.append(reshape_embd_word_for_concat(embd_word))

        if K.learning_phase():
            embd_word = C.replace_embedding_word_during_training(i)(caption_input)

    caption = Concatenate(axis=1)(words)

    # Assemble and Return Model
    return C.create_compile_model([cnn_input, caption_input], caption, gpus=gpus, lr=lr, loss='mean_squared_error')
