import configparser
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from experiments.classifier import NeuralNetworkClassifier
from experiments.dataset import Dataset
from experiments.experiment import Experiment
from utils.log_saver import LogSaver


def run_experiment(experiment_config):
    dataset = Dataset('data/autism.tsv')
    num_epochs = 1000
    eval_every = 10

    for fold_id, (train_idxs, test_idxs) in dataset.cross_validation():

        data_train_fold = dataset.get_data(train_idxs)
        num_instances, labels_train_fold = dataset.get_labels(train_idxs)

        data_test_fold = dataset.get_data(test_idxs)
        _, labels_test_fold = dataset.get_labels(test_idxs)

        with tf.Graph().as_default() as graph:

            experiment = Experiment(experiment_config, num_instances, NeuralNetworkClassifier, data_train_fold)

            with tf.Session() as session:

                global_step = 0
                session.run(tf.global_variables_initializer())

                log_saver = LogSaver('logs', 'fisher_fold{}'.format(fold_id), session.graph)

                train_selected_data = session.run(experiment.selection_wrapper.selected_data)
                test_selected_data = session.run(experiment.selection_wrapper.select(data_test_fold))

                tqdm_iter = tqdm(range(num_epochs), desc='Epochs')

                for epoch in tqdm_iter:
                    feed_dict = {experiment.clf.x: train_selected_data, experiment.clf.y: labels_train_fold}
                    loss, _ = session.run([experiment.clf.loss, experiment.clf.opt],
                                          feed_dict=feed_dict)

                    if epoch % eval_every == 0:
                        summary = session.run(experiment.clf.summary_op, feed_dict=feed_dict)
                        log_saver.log_train(summary, epoch)

                        feed_dict = {experiment.clf.x: test_selected_data, experiment.clf.y: labels_test_fold}
                        summary = session.run(experiment.clf.summary_op, feed_dict=feed_dict)
                        log_saver.log_test(summary, epoch)

                    tqdm_iter.set_postfix(loss='{:.2f}'.format(float(loss)), epoch=epoch)


def main():
    parser = ArgumentParser()
    parser.add_argument('experiment',
                        default='simple_experiment',
                        choices=['simple_experiment'],
                        help='model used during training (default: %(default))')

    args = parser.parse_args()
    experiment_config = configparser.ConfigParser()
    experiment_config.read('config/experiments/{}.ini'.format(args.experiment))

    run_experiment(experiment_config)


if __name__ == '__main__':
    main()
