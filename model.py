import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
# ILSTM CELL
# ─────────────────────────────────────────────

class ILSTMCell(keras.layers.Layer):
    """
    ILSTM Cell — equations (2)-(7) from the paper.

    Differences from LSTM:
      - No output gate
      - Input gate includes c_{t-1}
      - CIM = tanh(it) replaces candidate gate
      - ht = tanh(ct) directly

    Weights: 4 matrices, 2 biases (vs LSTM's 8 and 4)
    """

    def __init__(self, units, kernel_initializer='he_normal', **kwargs):
        super().__init__(**kwargs)
        self.units              = units
        self.kernel_initializer = kernel_initializer

    @property
    def state_size(self):
        return [self.units, self.units]   # [h, c]

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        init      = self.kernel_initializer

        # Forget gate weights  [eq. 2]
        self.Wfx = self.add_weight(name='Wfx', shape=(input_dim, self.units), initializer=init)
        self.Wfh = self.add_weight(name='Wfh', shape=(self.units, self.units), initializer=init)
        self.bf  = self.add_weight(name='bf',  shape=(self.units,), initializer='zeros')

        # Input gate weights  [eq. 4]
        self.Wix = self.add_weight(name='Wix', shape=(input_dim, self.units), initializer=init)
        self.Wih = self.add_weight(name='Wih', shape=(self.units, self.units), initializer=init)
        self.bi  = self.add_weight(name='bi',  shape=(self.units,), initializer='zeros')

        self.built = True

    def call(self, x_t, states):
        h_prev, c_prev = states[0], states[1]

        # Forget gate — eq. (2)
        ft = tf.sigmoid(x_t @ self.Wfx + h_prev @ self.Wfh + self.bf)

        # Mainline forgetting — eq. (3)
        kt = ft * c_prev

        # Input gate — eq. (4): c_prev included for memory effect
        it = tf.sigmoid(x_t @ self.Wix + h_prev @ self.Wih + c_prev + self.bi)

        # CIM: anti-supersaturation — eq. (5)
        CIM = tf.tanh(it)

        # Cell state — eq. (6)
        ct = kt + CIM

        # Hidden state — eq. (7): no output gate
        ht = tf.tanh(ct)

        return ht, [ht, ct]

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': self.kernel_initializer
        })
        return config


# ─────────────────────────────────────────────
# ILSTM LAYER
# ─────────────────────────────────────────────

class ILSTMLayer(keras.layers.Layer):
    """
    Wraps ILSTMCell in layers.RNN.
    return_sequences=False → returns only final hidden state.
    """

    def __init__(self, units, kernel_initializer='he_normal', **kwargs):
        super().__init__(**kwargs)
        self.units              = units
        self.kernel_initializer = kernel_initializer
        self.rnn = layers.RNN(
            ILSTMCell(units, kernel_initializer=kernel_initializer),
            return_sequences=False
        )

    def call(self, x):
        return self.rnn(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': self.kernel_initializer
        })
        return config


# ─────────────────────────────────────────────
# CNN-ILSTM MODEL
# ─────────────────────────────────────────────

def build_cnn_ilstm(window=23, n_features=19):
    """
    Builds CNN-ILSTM as per Fig. 9 and Table 5 of the paper.

    Input  : (batch, 23, 19)
    Conv1D : filters=16, kernel_size=1, padding='valid', activation='relu'
    MaxPool: pool_size=1, padding='valid'
    ILSTM  : units=16, kernel_initializer='he_normal'
    Dense  : 1  (AQI prediction)
    """
    inputs = keras.Input(shape=(window, n_features), name='input')

    # CNN block — feature extraction
    x = layers.Conv1D(
        filters=19, kernel_size=1,
        padding='valid', activation='relu',
        name='conv1d'
    )(inputs)
    x = layers.MaxPooling1D(
        pool_size=1, padding='valid',
        name='maxpool'
    )(x)

    # ILSTM block — temporal modelling
    x = ILSTMLayer(units=32, kernel_initializer='he_normal', name='ilstm')(x)

    # Output
    outputs = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_ILSTM')
    return model


# ─────────────────────────────────────────────
# BUILD AND VERIFY
# ─────────────────────────────────────────────

model = build_cnn_ilstm(window=23, n_features=19)
model.summary()