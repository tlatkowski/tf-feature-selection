import tensorflow as tf


def ff_neural_network(inputs, units):
    layer = tf.layers.dense(inputs, units=units, activation=tf.nn.tanh)
    output = tf.layers.dense(layer, units=1)
    return output


class NeuralNetworkClassifier:

    def __init__(self, num_features, units):
        self.x = tf.placeholder(dtype=tf.float64, shape=[None, num_features], name='inputs')
        self.y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='labels')

        output = ff_neural_network(self.x, units=units)

        with tf.name_scope('loss'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.y, output)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

        with tf.name_scope('metrics'):
            self.prediction = tf.nn.sigmoid(output)

            self.correct_predictions = tf.equal(self.prediction, self.y)
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()
