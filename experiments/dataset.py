import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.data_reader import read


class Dataset:

    def __init__(self, data_fn):
        self.data = read(data_fn)
        # FIXME
        self.labels = np.concatenate([np.ones(82, dtype=np.float64), np.zeros(64, dtype=np.float64)])
        self.labels = np.reshape(self.labels, (-1, 1))

        self.skf = StratifiedKFold(n_splits=10)

    def cross_validation(self):
        return enumerate(self.skf.split(self.data, self.labels.reshape(146)))

    def get_data(self, indices):
        return self.data[indices, :]

    def get_labels(self, indices):
        selected_labels = self.labels[indices]
        num_instances = [int(sum(selected_labels == 0)), int(sum(selected_labels == 1))]
        return num_instances, selected_labels
