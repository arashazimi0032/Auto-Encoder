import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from CustomLayers import BooleanMask


class AutoEncoder:
    def __init__(self, encoder_dims):
        self.encoder_dims = encoder_dims
        self.decoder_dims = np.array(self.encoder_dims)[:-1][::-1]
        self.model = Model()
        self.history = None
        self.no_data_value = None

    def build_model(self, input_shape, no_data_value=None):
        self.no_data_value = no_data_value
        inputs = Input(shape=input_shape)
        features = Dense(self.encoder_dims[0], activation='relu')(inputs)
        for d in self.encoder_dims[1:]:
            features = Dense(d, activation='relu')(features)

        for d in self.decoder_dims:
            features = Dense(d, activation='relu')(features)

        outputs = Dense(input_shape[0])(features)
        outputs = BooleanMask(no_data_value=no_data_value)([outputs, inputs])
        self.model = Model(inputs=inputs, outputs=outputs)

    def summary(self):
        self.model.summary()

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        self.history = self.model.history

    def predict(self, *args, **kwargs):
        if self.no_data_value is not None:
            layers_output = self.model.layers[-2].output
            prediction_model = Model(inputs=self.model.input, outputs=layers_output)
        else:
            prediction_model = self.model
        return prediction_model.predict(*args, **kwargs)
