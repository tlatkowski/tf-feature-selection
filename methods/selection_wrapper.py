import tensorflow as tf


class SelectionWrapper:

    def __init__(self, data, num_instances, selection_method=None, num_features=None):
        if data is None:
            raise ValueError('Provide data to make selection.')

        if selection_method is None:
            raise ValueError('Provide selection method.')

        if num_features is None:
            data = tf.convert_to_tensor(data)
            num_features = data.get_shape().as_list()[-1]

        self.values, self.indices = selection_method(data, num_instances, num_features)
        self.selected_data = tf.gather(data, self.indices, axis=1)

    def select(self, data):
        return tf.gather(data, self.indices, axis=1)
