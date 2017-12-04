import tensorflow as tf


def fisher(data, num_instances: list, top_k=10):
    class1, class2 = tf.split(data, num_instances)
    mean1, std1 = tf.nn.moments(class1, axes=0)
    mean2, std2 = tf.nn.moments(class2, axes=0)
    fisher_coeffs = (mean1 - mean2) / (std1 + std2)
    return tf.nn.top_k(fisher_coeffs, k=top_k)

