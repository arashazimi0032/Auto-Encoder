import tensorflow as tf


def custom_mse_loss(y_true, y_pred):
    boolean_mask = tf.not_equal(y_true, 0)
    squared_difference = tf.square(y_true[boolean_mask] - y_pred[boolean_mask])
    return tf.reduce_mean(squared_difference, axis=-1)
