import numpy as np
import tensorflow as tf
from utils.log_saver import LogSaver
from experiments.experiment import ExperimentModel
from methods.selection import fisher
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils.data_reader import read


data_fn = 'data/autism.tsv'
data = read(data_fn)

num_features = 100
num_epochs = 1000
eval_every = 10

labels = np.concatenate([np.ones(82, dtype=np.float64), np.zeros(64, dtype=np.float64)])
labels = np.reshape(labels, (-1, 1))


skf = StratifiedKFold(n_splits=10)

for fold_id, (train_idxs, test_idxs) in enumerate(skf.split(data, labels.reshape(146))):

    data_train_fold = data[train_idxs, :]
    labels_train_fold = labels[train_idxs]
    num_instances = [int(sum(labels_train_fold == 0)), int(sum(labels_train_fold == 1))]

    data_test_fold = data[test_idxs, :]
    labels_test_fold = labels[test_idxs]

    with tf.Graph().as_default() as graph:

        model = ExperimentModel(fisher, num_features, num_instances, None, data_train_fold)

        with tf.Session() as session:

            global_step = 0
            session.run(tf.global_variables_initializer())

            log_saver = LogSaver('logs', 'fisher_fold{}'.format(fold_id), session.graph)

            train_selected_data = session.run(model.selection_wrapper.selected_data)
            test_selected_data = session.run(model.selection_wrapper.select(data_test_fold))

            tqdm_iter = tqdm(range(num_epochs), desc='Epochs')

            for epoch in tqdm_iter:
                feed_dict = {model.clf.x: train_selected_data, model.clf.y: labels_train_fold}
                loss, _, summary = session.run([model.clf.loss, model.clf.opt, model.clf.summary_op], feed_dict=feed_dict)

                if epoch % eval_every == 0:
                    summary = session.run(model.clf.summary_op, feed_dict=feed_dict)
                    log_saver.log_train(summary, epoch)

                    feed_dict = {model.clf.x: test_selected_data, model.clf.y: labels_test_fold}
                    summary = session.run(model.clf.summary_op, feed_dict=feed_dict)
                    log_saver.log_test(summary, epoch)

                tqdm_iter.set_postfix(loss='{:.2f}'.format(float(loss)), epoch=epoch)