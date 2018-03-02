import tensorflow as tf


def fisher(data, num_instances: list, top_k_features=2):
    """
    Performs Fisher feature selection method according to the following formula:
    D(f) = (m1(f) - m2(f) / (std1(f) - std2(f))

    :param data:
    :param num_instances:
    :param top_k_features:
    :return: the list of most significant features.
    """
    assert len(num_instances) == 2, "Fisher selection method can be performed for two-class problems."

    data = tf.convert_to_tensor(data)
    num_features = data.get_shape().as_list()[-1]
    if top_k_features > num_features:
        top_k_features = num_features
    class1, class2 = tf.split(data, num_instances)

    with tf.name_scope('fisher_selection'):
        mean1, std1 = tf.nn.moments(class1, axes=0)
        mean2, std2 = tf.nn.moments(class2, axes=0)
        fisher_coeffs = tf.abs(mean1 - mean2) / (std1 + std2)
        selected_features = tf.nn.top_k(fisher_coeffs, k=top_k_features)

    return selected_features


def feature_correlation_with_class(data, num_instances: list, top_k_features=10):
    """
    Makes feature correlation with class selection according to the following formula:
    D(f) = [(m1(f) - m(f))^2 + (m2(f) - m(f))^2] / 2*sigma(f)^2
    :return: the list of most significant features.
    """
    data = tf.convert_to_tensor(data)
    num_features = data.get_shape().as_list()[-1]
    if top_k_features > num_features:
        top_k_features = num_features
    class1, class2 = tf.split(data, num_instances)

    with tf.name_scope('corr_selection'):
        mean1, std1 = tf.nn.moments(class1, axes=0)
        mean2, std2 = tf.nn.moments(class2, axes=0)
        mean, std = tf.nn.moments(data, axes=0)
        corr_coeffs = (tf.square(mean1 - mean) + tf.square(mean2 - mean)) / 2 * tf.square(std)
        selected_features = tf.nn.top_k(corr_coeffs, k=top_k_features)

    return selected_features


def t_test(data, num_instances: list, top_k_features=10):
    """
    Makes feature correlation with class selection according to the following formula:
    D(f) = [(m1(f) - m(f))^2 + (m2(f) - m(f))^2] / 2*sigma(f)^2
    :return: the list of most significant features.
    """
    data = tf.convert_to_tensor(data)
    num_features = data.get_shape().as_list()[-1]
    if top_k_features > num_features:
        top_k_features = num_features
    class1, class2 = tf.split(data, num_instances)

    with tf.name_scope('t_test_selection'):
        mean1, std1 = tf.nn.moments(class1, axes=0)
        mean2, std2 = tf.nn.moments(class2, axes=0)
        t_test_coeffs = tf.abs(mean1 - mean2) / tf.sqrt(
            tf.square(std1) / num_instances[0] + tf.square(std2) / num_instances[1])
        selected_features = tf.nn.top_k(t_test_coeffs, k=top_k_features)

    return selected_features


def random(data, num_instances: list, top_k_features=10):
    data = tf.convert_to_tensor(data)
    num_features = data.get_shape().as_list()[-1]
    if top_k_features > num_features:
        top_k_features = num_features
    class1, class2 = tf.split(data, num_instances)

    with tf.name_scope('random_selection'):
        pass