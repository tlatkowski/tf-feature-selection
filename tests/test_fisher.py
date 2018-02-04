import numpy as np
import tensorflow as tf

from methods.selection import fisher


class TestFisherSelection(tf.test.TestCase):

    def testFisherCorrectScore(self):
        with self.test_session() as test_session:
            data = np.array([[2, 2],
                             [4, 4],
                             [3, 6],
                             [5, 6]])
            num_instances = [2, 2]
            top_k = 2
            actual_most_significant_features = test_session.run(fisher(data, num_instances, top_k))

            correct_most_significant_features = tf.constant([3., .5])
            self.assertAllEqual(actual_most_significant_features.values, correct_most_significant_features.eval())

    def testFisherCorrectOrderOfFeatures(self):
        raise NotImplementedError

    def testMoreThan2ClassesIsNotAllowed(self):
        raise NotImplementedError

if __name__ == '__main__':
    tf.test.main()
