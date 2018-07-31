from keras import backend as K
from keras import Model, Input
from keras.layers import Dense, LSTM, Concatenate, Reshape, Lambda, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model

# with eval()
from keras.applications import ResNet50 as resnet50
from src.common.modules.resnet import ResNet152Embed as resnet152


def image_captioning_model_raw(img_shape=(224, 224, 3), cnn='resnet152', embedding_dim=50, max_caption_length=15,
                               gpus=0, lr=1e-3, regularizer=1e-8, dropout=0.2):
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
    cnn_output = Flatten()(cnn.output) if cnn == 'resnet152' else Flatten()(AveragePooling2D((7, 7))(cnn.output))

    # Caption Input
    caption_input = Input((max_caption_length, embedding_dim))

    # Start vars
    embd_word = Dense(50, activation='relu')(cnn_output)
    state = None
    words = []

    # Definition of RNN
    rnn = LSTM(50, return_sequences=False, return_state=True,
               dropout=dropout, recurrent_dropout=dropout,
               recurrent_regularizer=l2(regularizer),
               bias_regularizer=l2(regularizer),
               kernel_regularizer=l2(regularizer))

    # Auxiliary layers
    reshape_rnn_in = Reshape((1, embedding_dim))
    reshape_embd_word_for_concat = Reshape((1, embedding_dim))

    for i in range(max_caption_length):
        rnn_in = reshape_rnn_in(embd_word)
        rnn_out, hidden_state, cell_state = rnn(rnn_in, initial_state=state)
        state = [hidden_state, cell_state]

        embd_word = reshape_embd_word_for_concat(rnn_out)
        words.append(rnn_in)
        if K.learning_phase():
            embd_word = Lambda(lambda x, ii: x[:, ii], arguments={'ii': i})(caption_input)

    caption = Concatenate(axis=1)(words)

    # Assemble Model
    model = Model(inputs=[cnn_input, caption_input], outputs=caption)
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['mae'])
    return model
