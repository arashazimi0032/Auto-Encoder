import keras
import tensorflow as tf


class BooleanMask(keras.layers.Layer):
    def __init__(self, no_data_value=None):
        super().__init__()
        self.no_data_value = no_data_value

    def call(self, inputs):
        org_input, input_sample = inputs
        if self.no_data_value is not None:
            boolean_mask = tf.cast(tf.not_equal(input_sample, self.no_data_value), input_sample.dtype)
            no_data_vec = tf.cast(tf.equal(input_sample, self.no_data_value), input_sample.dtype) * self.no_data_value
            matmul = org_input * boolean_mask + no_data_vec
        else:
            return org_input
        return matmul
