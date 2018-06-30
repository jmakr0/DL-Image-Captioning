from keras import Input, Model
from keras.engine import InputSpec
from keras.layers import LSTM, activations, initializers, K, TimeDistributed, Dense, GlobalAveragePooling2D, \
    RepeatVector


class AttentionLSTM(LSTM):
    """LSTM with attention mechanism
    This is an LSTM incorporating an attention mechanism into its hidden states.
    Currently, the context vector calculated from the attended vector is fed
    into the model's internal states, closely following the model by Xu et al.
    (2016, Sec. 3.1.2), using a soft attention model following
    Bahdanau et al. (2014).
    The layer expects two inputs instead of the usual one:
        1. the "normal" layer input; and
        2. a 3D vector to attend.
    Args:
        attn_activation: Activation function for attentional components
        attn_init: Initialization function for attention weights
        output_alpha (boolean): If true, outputs the alpha values, i.e.,
            what parts of the attention vector the layer attends to at each
            timestep.
    References:
        * Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly
          Learning to Align and Translate", <https://arxiv.org/pdf/1409.0473.pdf>
        * Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel & Bengio (2016),
          "Show, Attend and Tell: Neural Image Caption Generation with Visual
          Attention", <http://arxiv.org/pdf/1502.03044.pdf>
    See Also:
        `LSTM`_ in the Keras documentation.
        .. _LSTM: http://keras.io/layers/recurrent/#lstm
    """
    def __init__(self, *args, attn_activation='tanh', attn_initializer='orthogonal',
                 output_alpha=False, **kwargs):
        self.attn_activation = activations.get(attn_activation)
        self.attn_initializer = initializers.get(attn_initializer)
        #self.attn_init = initializations.get(attn_init)
        self.output_alpha = output_alpha
        super().__init__(*args, **kwargs)
        self.output_dim = self.units

    def inner_init(self, shape, name=None):
        return self.add_weight(name, shape, initializer=self.recurrent_initializer)

    def attn_init(self, shape, name=None):
        return self.add_weight(name, shape, initializer=self.attn_initializer)

    def init(self, shape, name=None):
        return self.add_weight(name, shape, initializer=self.kernel_initializer)

    def inner_activation(self, x):
        return self.recurrent_activation(x)

    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise Exception('Input to AttentionLSTM must be a list of '
                            'two tensors [lstm_input, attn_input].')

        input_shape, attn_input_shape = input_shape
        super().build(input_shape)
        self.input_spec.append(InputSpec(shape=attn_input_shape))

        # weights for attention model
        self.U_att = self.inner_init((self.output_dim, self.output_dim), name='{}_U_att'.format(self.name))
        self.W_att = self.attn_init((attn_input_shape[-1], self.output_dim), name='{}_W_att'.format(self.name))
        self.v_att = self.init((self.output_dim, 1), name='{}_v_att'.format(self.name))
        self.b_att = K.zeros((self.output_dim,), name='{}_b_att'.format(self.name))
        #self.trainable_weights += [self.U_att, self.W_att, self.v_att, self.b_att]

        # weights for incorporating attention into hidden states
        if hasattr(self, 'consume_less') and self.consume_less == 'gpu':
        #if self.consume_less == 'gpu':
            self.Z = self.init((attn_input_shape[-1], 4 * self.output_dim),
                               name='{}_Z'.format(self.name))
            #self.trainable_weights += [self.Z]
        else:
            self.Z_i = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_i'.format(self.name))
            self.Z_f = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_f'.format(self.name))
            self.Z_c = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_c'.format(self.name))
            self.Z_o = self.attn_init((attn_input_shape[-1], self.output_dim),
                                      name='{}_Z_o'.format(self.name))
            #self.trainable_weights += [self.Z_i, self.Z_f, self.Z_c, self.Z_o]
            self.Z = K.concatenate([self.Z_i, self.Z_f, self.Z_c, self.Z_o])

        # weights for initializing states based on attention vector
        if not self.stateful:
            self.W_init_c = self.attn_init((attn_input_shape[-1], self.output_dim),
                                           name='{}_W_init_c'.format(self.name))
            self.W_init_h = self.attn_init((attn_input_shape[-1], self.output_dim),
                                           name='{}_W_init_h'.format(self.name))
            self.b_init_c = K.zeros((self.output_dim,),
                                    name='{}_b_init_c'.format(self.name))
            self.b_init_h = K.zeros((self.output_dim,),
                                    name='{}_b_init_h'.format(self.name))
            #self.trainable_weights += [self.W_init_c, self.b_init_c, self.W_init_h, self.b_init_h]

        if hasattr(self, 'initial_weights') and self.initial_weights is not None:
        #if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return super().compute_output_shape(input_shape[0])
        #return super().get_output_shape_for(input_shape[0])

    def compute_mask(self, input, input_mask=None):
        if input_mask is not None:
            input_mask = input_mask[0]
        return super().compute_mask(input, input_mask)
        #return super().compute_mask(input, input_mask=input_mask)

    def get_initial_states(self, x_input, x_attn, mask_attn):
        # set initial states from mean attention vector fed through a dense
        # activation
        mean_attn = K.mean(x_attn * K.expand_dims(mask_attn), axis=1)
        h0 = K.dot(mean_attn, self.W_init_h) + self.b_init_h
        c0 = K.dot(mean_attn, self.W_init_c) + self.b_init_c
        return [self.attn_activation(h0), self.attn_activation(c0)]

    def call(self, x, mask=None, training=None, initial_state=None):
    #def call(self, x, mask=None):
        assert isinstance(x, list) and len(x) == 2
        x_input, x_attn = x
        if mask is not None:
            mask_input, mask_attn = mask
        else:
            mask_input, mask_attn = None, None
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if initial_state is not None:
            initial_states = initial_state
        elif self.stateful:
        #if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x_input, x_attn, mask_attn)
        constants = self.get_constants(x_input, x_attn, mask_attn)
        preprocessed_input = x_input
        #preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask_input,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            #self.updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
                #self.updates.append((self.states[i], states[i]))
            self.add_update(updates, x_input)

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        x_attn = states[4]
        mask_attn = states[5]
        attn_shape = self.input_spec[1].shape

        #### attentional component
        # alignment model
        # -- keeping weight matrices for x_attn and h_s separate has the advantage
        # that the feature dimensions of the vectors can be different
        h_att = K.repeat(h_tm1, attn_shape[1])
        att = time_distributed_dense(x_attn, self.W_att, self.b_att)
        energy = self.attn_activation(K.dot(h_att, self.U_att) + att)
        energy = K.squeeze(K.dot(energy, self.v_att), 2)
        # make probability tensor
        alpha = K.exp(energy)
        if mask_attn is not None:
            alpha *= mask_attn
        alpha /= K.sum(alpha, axis=1, keepdims=True)
        alpha_r = K.repeat(alpha, attn_shape[2])
        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))
        # make context vector -- soft attention after Bahdanau et al.
        z_hat = x_attn * alpha_r
        z_hat = K.sum(z_hat, axis=1)

        if True:
        #if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) \
                + K.dot(z_hat, self.Z) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]
        else:
            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            z0 = x_i + K.dot(h_tm1 * B_U[0], self.U_i) + K.dot(z_hat, self.Z_i)
            z1 = x_f + K.dot(h_tm1 * B_U[1], self.U_f) + K.dot(z_hat, self.Z_f)
            z2 = x_c + K.dot(h_tm1 * B_U[2], self.U_c) + K.dot(z_hat, self.Z_c)
            z3 = x_o + K.dot(h_tm1 * B_U[3], self.U_o) + K.dot(z_hat, self.Z_o)

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)

        h = o * self.activation(c)
        if self.output_alpha:
            return alpha, [h, c]
        else:
            return h, [h, c]

    def super_get_constants(self, x):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_constants(self, x_input, x_attn, mask_attn):
        constants = self.super_get_constants(x_input)
        #constants = super().get_constants(x_input)
        attn_shape = self.input_spec[1].shape
        if mask_attn is not None:
            if K.ndim(mask_attn) == 3:
                mask_attn = K.all(mask_attn, axis=-1)
        constants.append(x_attn)
        constants.append(mask_attn)
        return constants

    def get_config(self):
        cfg = super().get_config()
        cfg['output_alpha'] = self.output_alpha
        cfg['attn_activation'] = self.attn_activation.__name__
        return cfg

    @classmethod
    def from_config(cls, config):
        instance = super(AttentionLSTM, cls).from_config(config)
        if 'output_alpha' in config:
            instance.output_alpha = config['output_alpha']
        if 'attn_activation' in config:
            instance.attn_activation = activations.get(config['attn_activation'])
        return instance


def time_distributed_dense(x, w, b=None, dropout=None,
                           input_dim=None, output_dim=None, timesteps=None):
    '''Apply y.w + b for every temporal slice y of x.
    '''
    if not input_dim:
        # won't work with TensorFlow
        input_dim = K.shape(x)[2]
    if not timesteps:
        # won't work with TensorFlow
        timesteps = K.shape(x)[1]
    if not output_dim:
        # won't work with TensorFlow
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))

    x = K.dot(x, w)
    if b:
        x = x + b
    # reshape to 3D tensor
    x = K.reshape(x, (-1, timesteps, output_dim))
    return x


if __name__ == '__main__':
    units = 1024

    feature_vector = Input(shape=(14, 14, 512))
    feature_vector_flat = GlobalAveragePooling2D()(feature_vector)
    feature_vector_flat = RepeatVector(25)(feature_vector_flat)

    attn_lstm = AttentionLSTM(units=units)([feature_vector_flat, feature_vector])

    model = Model(inputs=feature_vector, outputs=attn_lstm)
    model.compile('adam', loss='kullback_leibler_divergence', metrics=['accuracy'])
    model.summary()
