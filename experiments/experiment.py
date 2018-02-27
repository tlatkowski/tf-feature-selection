import tensorflow as tf

from experiments.classifier import NeuralNetworkClassifier
from methods.selection_wrapper import SelectionWrapper
from methods.selection import fisher, feature_correlation_with_class, t_test, random

methods = {
    'fisher': fisher,
    'corr': feature_correlation_with_class,
    'ttest': t_test,
    'random': random
}


class Experiment:

    def __init__(self, experiment_config, num_features, num_instances, classifier, dataset):

        selection_method = methods[experiment_config['SELECTION']['method']]
        num_features = experiment_config['SELECTION']['num_features']

        with tf.name_scope('selection'):
            self.selection_wrapper = SelectionWrapper(dataset,
                                                      num_instances=num_instances,
                                                      selection_method=selection_method,
                                                      num_features=num_features)

        with tf.name_scope('classifier'):
            self.clf = NeuralNetworkClassifier(num_features, 20)