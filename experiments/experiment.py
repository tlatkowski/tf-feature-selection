import tensorflow as tf

from experiments.classifier import NeuralNetworkClassifier
from methods.selection_wrapper import SelectionWrapper


class ExperimentModel:

    def __init__(self, selection_method, num_features, num_instances, classifier, dataset):

        with tf.name_scope('selection'):
            self.selection_wrapper = SelectionWrapper(dataset,
                                                     num_instances=num_instances,
                                                     selection_method=selection_method,
                                                     num_features=num_features)

        with tf.name_scope('classifier'):
            self.clf = NeuralNetworkClassifier(num_features, 20)
