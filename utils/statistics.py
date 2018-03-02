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


def f_test(data, num_instances):
    """
    Performs F-statistic between the genes and the classification variable h
    as the score of maximum relevance.

    :param data:
    :param num_instances:
    :return:
    """

    data = tf.convert_to_tensor(data)
    class1, class2 = tf.split(data, num_instances)
    K = 2
    with tf.name_scope('f_statistic'):
        mean1, var1 = tf.nn.moments(class1, axes=0)
        mean2, var2 = tf.nn.moments(class2, axes=0)
        mean, var = tf.nn.moments(data, axes=0)

        pooled_var = pooled_variance(data, num_instances)
        tf.reduce_sum(((mean1 - mean) + (mean2 - mean))/(K-1))/pooled_var


def pooled_variance(data, num_instances):
    K = len(num_instances)
    n = sum(num_instances)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    split_classes = tf.split(data, num_instances)
    vars = []
    for i in range(len(split_classes)):
        _, var = tf.nn.moments(split_classes[i], axes=0)
        vars.append(var)

    n_k = tf.to_float(tf.reshape(num_instances, [K, -1]))
    stacked_var = tf.stack(vars)
    pooled_var = tf.reduce_sum(stacked_var * (n_k - 1), axis=0) / (n - K)
    return pooled_var