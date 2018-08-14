import os
import sys
import json
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from cnn_embeddings import CNNEmbeddings
import sys
sys.path.append('../')
import data_helper

logging.getLogger().setLevel(logging.INFO)


def load_trained_params(trained_dir):
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())

    with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
        fetched_embedding = pickle.load(input_file)
    embedding_mat = np.array(fetched_embedding, dtype=np.float32)
    return params, words_index, labels, embedding_mat


def load_test_data(test_file, labels):
    df = pd.read_csv(test_file, sep=',')
    select = ['text']

    df = df.dropna(axis=0, how='any', subset=select)
    test_examples = df[select[0]].apply(lambda x: data_helper.clean_text(x, True, False)).tolist()

    # test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_ = None
    if 'label' in df.columns:
        select.append('label')
        y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

    not_select = list(set(df.columns) - set(select))
    df = df.drop(not_select, axis=1)
    return test_examples, y_, df


def map_word_to_index(examples, words_index):
    x_ = []
    for example in examples:
        temp = []
        for word in example:
            if word in words_index:
                temp.append(words_index[word])
            else:
                temp.append(0)
        x_.append(temp)
    return x_


def load_graph(graph_dir):
    graph_def = tf.GraphDef()
    with tf.gfile.Open(graph_dir, 'rb') as f:
        data = f.read()
        graph_def.ParseFromString(data)

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def predict_unseen_data(trained_dir ,test_file):
    # trained_dir = sys.argv[1]
    # if not trained_dir.endswith('/'):
    #     trained_dir += '/'
    # test_file = sys.argv[2]

    params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
    x_, y_, voc, df = data_helper.load(test_file)
    # x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
    x_ = map_word_to_index(x_, words_index)

    x_test, y_test = np.asarray(x_), None
    if y_ is not None:
        y_test = np.asarray(y_)

    timestamp = trained_dir.split('/')[-2].split('_')[-1]
    predicted_dir = './predicted_results_' + timestamp + '/'
    if os.path.exists(predicted_dir):
        shutil.rmtree(predicted_dir)
    os.makedirs(predicted_dir)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CNNEmbeddings(
                vocab_size=7813,
                sequence_length=len(x_test[0]),
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                num_classes=len(labels),
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            # def real_len(batches):
            #     return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            def predict_step(x_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.dropout_keep_prob: 1.0,
                }
                predictions = sess.run(cnn.predictions, feed_dict)
                return predictions

            print(os.curdir)

            checkpoint_file = trained_dir + 'checkpoints/model-800'
            saver = tf.train.Saver(tf.global_variables())
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            logging.critical('{} has been loaded'.format(checkpoint_file))

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

            predictions = []
            predict_labels = []
            for x_batch in batches:
                batch_predictions = predict_step(x_batch)
                for batch_prediction in batch_predictions:
                    predictions.append(batch_prediction)
                    predict_labels.append(labels[batch_prediction])

            # Save the predictions back to file
            df['NEW_PREDICTED'] = predict_labels
            columns = sorted(df.columns, reverse=True)
            df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

            if y_test is not None:
                y_test = np.array(np.argmax(y_test, axis=1))
                accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
                logging.critical('The prediction accuracy is: {}'.format(accuracy))

            logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))


if __name__ == '__main__':
    # pythonw predict.py ./trained_model_1534159683/ ../data/isear_test.csv

    dir = './trained_model_1534163991/'
    test_file = '../data/isear_test.csv'
    predict_unseen_data(dir, test_file)