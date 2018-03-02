import numpy as np
import tensorflow as tf

from utils.statistics import pooled_variance


class TestStatistics(tf.test.TestCase):

    def testPooledVariance(self):
        with self.test_session() as test_session:
            data = np.array([[2., 3., 4., 5.],
                             [2., 3., 4., 5.],
                             [2., 3., 4., 5.],
                             [2., 3., 4., 5.]])
            num_instances = [2, 2]
            actual_pooled_variance = test_session.run(pooled_variance(data, num_instances))
            correct_pooled_variance = [.0, .0, .0, .0]

            self.assertAllEqual(actual_pooled_variance, correct_pooled_variance)


if __name__ == '__main__':
    tf.test.main()
