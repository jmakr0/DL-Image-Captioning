from keras import Input
from keras import backend as K
from keras.layers import Multiply, Concatenate, Reshape, Lambda, Dropout

import src.cacao.model_commons as C


def image_captioning_model(img_shape=(224, 224, 3), cnn='resnet152', embedding_dim=50, max_caption_length=15,
                           gpus=0, lr=1e-3, regularizer=1e-8, dropout_rate=0.2):
    # Definition of CNN
    cnn_input = Input(shape=img_shape)
    cnn_output, cnn_output_len = C.image_embedding(cnn_input, cnn, img_shape)

    # Caption Input
    caption_input = Input((max_caption_length, embedding_dim))

    # Definition of RNN
    rnn = C.lstm_generator(150, dropout=dropout_rate, l2_reg=regularizer)
    attention_layer = C.dense_attention_layer(cnn_output_len, l2_reg=regularizer)
    embedding = C.x_layer_word_embedding([100, embedding_dim], l2_reg=regularizer)

    # Auxiliary layers
    multiply = Multiply()
    dropout = Dropout(dropout_rate)
    concatenate = Concatenate()
    reshape_rnn_in = Reshape((1, embedding_dim + cnn_output_len))
    reshape_embd_word_for_concat = Reshape((1, embedding_dim))

    # Start vars
    embd_word = Lambda(C.constant_ones_tensor(embedding_dim))(cnn_input)
    attention = Lambda(C.constant_ones_tensor(cnn_output_len))(cnn_input)

    state = None
    words = []

    for i in range(max_caption_length):
        attention_image = dropout(multiply([cnn_output, attention]))
        rnn_in = concatenate([embd_word, attention_image])

        rnn_in = reshape_rnn_in(rnn_in)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)

        # update vars
        embd_word = embedding(rnn_out)
        attention = attention_layer(rnn_out)

        state = [hidden_state, cell_state]
        words.append(reshape_embd_word_for_concat(embd_word))

        if K.learning_phase():
            embd_word = C.replace_embedding_word_during_training(i)(caption_input)

    caption = Concatenate(axis=1)(words)

    # Assemble and Return Model
    return C.create_compile_model([cnn_input, caption_input], caption, gpus=gpus, lr=lr, loss='mean_squared_error')
