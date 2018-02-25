import tensorflow as tf


def pearson_correlation(x1, x2):
    x1 = tf.convert_to_tensor(x1)
    x2 = tf.convert_to_tensor(x2)
    m1, std1 = tf.nn.moments(x1, axes=0)
    m2, std2 = tf.nn.moments(x2, axes=0)
    l = tf.reduce_sum((x1 - m1) * (x2 - m2))
    i = tf.reduce_sum((x1 - m1) ** 2) * tf.reduce_sum((x2 - m2) ** 2)
    p = tf.sqrt(i)
    return l / p


def f_test():
    pass
