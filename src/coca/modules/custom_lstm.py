from keras.engine import Layer
from keras.layers import RNN, K, initializers, regularizers, constraints


class CustomLSTMCell(Layer):
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='uniform',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attention_regularizer = regularizers.get(attention_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attention_constraint = constraints.get(attention_constraint)

        self.state_size = (self.units, self.units)

    def build(self, input_shape):
        self.u_f = self.add_weight(shape=(input_shape[-1], self.units), name='U_f', initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        self.u_c = self.add_weight(shape=(input_shape[-1], self.units), name='U_c', initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        self.u_i = self.add_weight(shape=(input_shape[-1], self.units), name='U_i', initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        self.u_o = self.add_weight(shape=(input_shape[-1], self.units), name='U_o', initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        self.w_f = self.add_weight(shape=(self.units, self.units), name='W_f', initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)
        self.w_c = self.add_weight(shape=(self.units, self.units), name='W_c', initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)
        self.w_i = self.add_weight(shape=(self.units, self.units), name='W_i', initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)
        self.w_o = self.add_weight(shape=(self.units, self.units), name='W_o', initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.attention_weights = self.add_weight(shape=(self.units, input_shape[-1]), name='attention_weights',
                                                 initializer=self.attention_initializer,
                                                 regularizer=self.attention_regularizer,
                                                 constraint=self.attention_constraint)

        self.built = True

    def call(self, inputs, states, training=None):
        prev_c, prev_h = states

        attention = K.sigmoid(K.dot(prev_h, self.attention_weights))
        inputs *= attention

        forget_gate = K.hard_sigmoid(K.dot(inputs, self.u_f) + K.dot(prev_h, self.w_f))
        c_change = K.tanh(K.dot(inputs, self.u_c) + K.dot(prev_h, self.w_c)) * K.hard_sigmoid(
            K.dot(inputs, self.u_i) + K.dot(prev_h, self.w_i))
        output_gate = K.hard_sigmoid(K.dot(inputs, self.u_o) + K.dot(prev_h, self.w_o))

        c = prev_c * forget_gate + c_change
        h = output_gate * K.tanh(c)
        output = K.sigmoid(h)

        return output, [c, h]

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'attention_initializer': initializers.serialize(self.attention_regularizer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'attention_regularizer': regularizers.serialize(self.attention_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'attention_constraint': constraints.serialize(self.attention_constraint)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CustomLSTM(RNN):
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='uniform',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_constraint=None,
                 return_sequences=True,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = CustomLSTMCell(units,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              attention_initializer=attention_initializer,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              attention_regularizer=attention_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              attention_constraint=attention_constraint, )
        super().__init__(cell,
                         return_sequences=return_sequences,
                         return_state=return_state,
                         go_backwards=go_backwards,
                         stateful=stateful,
                         unroll=unroll,
                         **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        return super().call(inputs, mask=mask, training=training, initial_state=initial_state,
                            constants=constants)

    @property
    def units(self):
        return self.cell.units

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def attention_initializer(self):
        return self.cell.attention_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def attention_regularizer(self):
        return self.cell.attention_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def attention_constraint(self):
        return self.cell.attention_constraint

    def get_config(self):
        config = {'units': self.units,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'attention_initializer': initializers.serialize(self.attention_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'attention_regularizer': regularizers.serialize(self.attention_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'attention_constraint': constraints.serialize(self.attention_constraint)}
        base_config = super().get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
