import numpy as np
import tensorflow as tf

from methods.selection import fisher
from methods.selection_wrapper import SelectionWrapper


class TestFisherSelection(tf.test.TestCase):

    def testFisherCorrectScore(self):
        with self.test_session() as test_session:
            data = np.array([[2, 2],
                             [4, 4],
                             [3, 6],
                             [5, 6]])
            num_instances = [2, 2]
            top_k = 2
            actual_most_significant_features, _ = test_session.run(fisher(data, num_instances, top_k))
            correct_most_significant_features = [3., .5]

            self.assertAllEqual(actual_most_significant_features, correct_most_significant_features)

    def testFisherPickFirstSignificantFeature(self):
        with self.test_session() as test_session:
            data = np.array([[2, 2],
                             [4, 4],
                             [3, 6],
                             [5, 6]])

            num_instances = [2, 2]
            top_k = 1
            selection_wrapper = SelectionWrapper(data,
                                                 num_instances,
                                                 fisher,
                                                 num_features=top_k)
            actual_most_significant_features = test_session.run(selection_wrapper.selected_data)
            correct_most_significant_features = [[2.], [4.], [6.], [6.]]

            self.assertAllEqual(actual_most_significant_features, correct_most_significant_features)

    def testFisherCorrectOrderOfFeatures(self):
        with self.test_session() as test_session:
            data = np.array([[2, 2],
                             [4, 4],
                             [3, 6],
                             [5, 6]])
            num_instances = [2, 2]
            top_k = 2
            _, actual_most_significant_features = test_session.run(fisher(data, num_instances, top_k))
            correct_most_significant_features = [1., 0.]

            self.assertAllEqual(actual_most_significant_features, correct_most_significant_features)

    def testMoreThan2ClassesIsNotAllowed(self):
        with self.test_session() as test_session:
            data = np.array([[2, 2],
                             [4, 4],
                             [3, 6],
                             [5, 6]])
            num_instances = [2, 2, 2]
            top_k = 2
            with self.assertRaises(AssertionError):
                _, actual_most_significant_features = test_session.run(fisher(data, num_instances, top_k))


if __name__ == '__main__':
    tf.test.main()
