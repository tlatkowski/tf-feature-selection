import tensorflow as tf

from methods.selection import fisher, feature_correlation_with_class, t_test, random
from methods.selection_wrapper import SelectionWrapper

methods = {
    'fisher': fisher,
    'corr': feature_correlation_with_class,
    'ttest': t_test,
    'random': random
}


class Experiment:

    def __init__(self, experiment_config, num_instances, classifier, dataset):
        selection_method = methods[experiment_config['SELECTION']['method']]
        num_features = int(experiment_config['SELECTION']['num_features'])
        hidden_sizes = int(experiment_config['CLASSIFIER']['hidden_sizes'])

        with tf.name_scope('selection'):
            self.selection_wrapper = SelectionWrapper(dataset,
                                                      num_instances=num_instances,
                                                      selection_method=selection_method,
                                                      num_features=num_features)

        with tf.name_scope('classifier'):
            self.clf = classifier(num_features, hidden_sizes)
