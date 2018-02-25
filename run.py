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

labels = np.concatenate([np.ones(82, dtype=np.float64), np.zeros(64, dtype=np.float64)])
labels = np.reshape(labels, (-1, 1))


skf = StratifiedKFold(n_splits=10)

for fold_id, (train_idxs, test_idxs) in enumerate(skf.split(data, labels.reshape(146))):

    data_fold = data[train_idxs, :]
    labels_fold = labels[train_idxs]
    num_instances = [int(sum(labels_fold == 0)), int(sum(labels_fold == 1))]

    with tf.Graph().as_default() as graph:

        model = ExperimentModel(fisher, num_features, num_instances, None, data_fold)

        with tf.Session() as session:

            global_step = 0
            session.run(tf.global_variables_initializer())

            log_saver = LogSaver('logs', 'fisher_fold{}'.format(fold_id), session.graph)

            selected_data = session.run(model.selection_wrapper.selected_features)

            tqdm_iter = tqdm(range(num_epochs), desc='Epochs')

            for epoch in tqdm_iter:
                feed_dict = {model.clf.x: selected_data, model.clf.y: labels_fold}
                loss, _, summary = session.run([model.clf.loss, model.clf.opt, model.clf.summary_op], feed_dict=feed_dict)
                log_saver.log_train(summary, epoch)
                tqdm_iter.set_postfix(loss='{:.2f}'.format(float(loss)), epoch=epoch)