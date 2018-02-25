import numpy as np
import tensorflow as tf

from utils.statistics import pearson_correlation


class TestPearson(tf.test.TestCase):

    def testPearsonCoefficientValueForTwoVectors(self):
        with self.test_session() as test_session:
            x1 = np.array([2., 3., 4.])
            x2 = np.array([3., 1., 5.])

            actual_pearson_coefficient = test_session.run(pearson_correlation(x1, x2))
            correct_pearson_coefficient = [.5]

            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient)

    def testNegativePearsonCoefficientValueForTwoVectors(self):
        with self.test_session() as test_session:
            x1 = np.array([1., 2., 3.])
            x2 = np.array([-1., -2., -3.])

            actual_pearson_coefficient = test_session.run(pearson_correlation(x1, x2))
            correct_pearson_coefficient = [-1.]

            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient)

    def testPositivePearsonCoefficientValueForTwoVectors(self):
        with self.test_session() as test_session:
            x1 = np.array([1., 2., 3.])
            x2 = np.array([1., 2., 3.])

            actual_pearson_coefficient = test_session.run(pearson_correlation(x1, x2))
            correct_pearson_coefficient = [1.]

            self.assertEqual(actual_pearson_coefficient, correct_pearson_coefficient)


if __name__ == '__main__':
    tf.test.main()
