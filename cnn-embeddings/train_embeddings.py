import json
import logging
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import sys
sys.path.append('../')
import data_helper
from cnn_embeddings import TextCNN
from sklearn.metrics import classification_report, confusion_matrix

logging.getLogger().setLevel(logging.INFO)


def train_cnn():
    """Step 0: load sentences, labels, and training parameters"""
    train_file = '../data/iseardataset.csv'
    x_raw, y_raw, df, labels, embedding_mat = data_helper.load_data_and_labels(train_file)

    parameter_file = '../training_config.json'
    params = json.loads(open(parameter_file).read())

    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_raw)))
    y = np.array(y_raw)

    # print x.shape
    """Step 2: split the original dataset into train and test sets"""
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """Step 5: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=9000,
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'], embedding_mat=embedding_mat,
                l2_reg_lambda=params['l2_reg_lambda'])

            # Optimizing our loss function using Adam's optimizer
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summary for predictions
            # predictions_summary = tf.summary.scalar("predictions", cnn.predictions)


            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step, summaries, loss, acc = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries, step)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, acc, num_correct, predictions = \
                    sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions],
                             feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return num_correct, predictions

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            print "Loading Embeddings !"

            embedding_dimension = 200
            embedding_dir = '../embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
            # embedding_dir = '../GoogleNews-vectors-negative300.bin'

            initW = data_helper.load_embedding_vectors_glove(vocab_processor.vocabulary_, embedding_dir, embedding_dimension)
            # initW = data_helper.load_embedding_vectors_word2vec(vocab_processor.vocabulary_, embedding_dir, embedding_dimension)
            sess.run(cnn.W.assign(initW))

            print "Loaded Embeddings !"

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        if len(dev_batch) == 0:
                            continue
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct, y_pred_tre = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy of the dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                if len(test_batch) == 0:
                    continue
                print "Non Zero Length"
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, y_pred = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

            test_accuracy = (float(total_test_correct) / len(y_test))*100

            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], 1)

            total_train_correct = 0
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                print "Non Zero Length"
                x_train_batch, y_train_batch = zip(*train_batch)
                num_test_correct, y_ = dev_step(x_train_batch, y_train_batch)
                total_train_correct += num_test_correct

            train_accuracy = (float(total_train_correct) / len(y_train))*100

        print 'Accuracy on test set is {} based on the best model'.format(test_accuracy)
        print 'Accuracy on train set is {} based on the best model'.format(train_accuracy)
        # logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))

        print(len(y_test_batch))
        print(y_test_batch[0])
        print(len(y_pred))
        print(y_pred[0])
        # Y_test = np.argmax(y_test_batch, axis=1)
        # y_pred_class = np.argmax(y_pred, axis=1)

        print(classification_report(y_test_batch, y_pred, target_names=classes))

        # # Create confusion matrix
        # cnf_matrix = confusion_matrix(Y_test, y_pred_class)
        # plt.figure(figsize=(20, 10))
        # data_helper.plot_confusion_matrix(cnf_matrix, labels=classes)

        logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train_cnn.py
    train_cnn()
