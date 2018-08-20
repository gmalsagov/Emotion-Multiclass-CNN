#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import data_helper
import itertools


# Start a graph session
sess = tf.Session()

"""Step 1: load test and train data and training parameters"""

# Relative path to datasets
# train_file = '../data/iseardataset.csv'
test_set = '../data/isear_test.csv'
train_set = '../data/isear_train.csv'

# Classes to be predicted
classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
# classes = ["fear", "anger", "sadness", "disgust", "shame", "guilt"]

# Load data
print("Loading data...")

# Load train/validation data
x_raw, y_raw, vocabulary, df = data_helper.load(train_set)

# Load test data
x_test, y_test, vocabulary1, df1 = data_helper.load(test_set)

# Choose max text word length at 40 to cover most data
sentence_size = 40
min_word_freq = 3

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
vocab_processor.transform(x_raw)
transformed_texts = np.array([x for x in vocab_processor.transform(x_raw)])
embedding_size = len(np.unique(transformed_texts))
print("Total words: " + str(embedding_size))

# Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1, y_raw.shape[1]]))

# Initialize placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32, name="inputs_x")
y_target = tf.placeholder(shape=[1, 1, y_raw.shape[1]], dtype=tf.float32, name="predictions")

# Text-Vocab Embedding
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# Declare model operations
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Prediction operation
prediction = tf.sigmoid(model_output)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
print('Starting Training Over {} Sentences.'.format(len(x_raw)))
loss_vec = []
train_acc_all = []
train_acc_avg = []

"""Train Step"""
for ix, t in enumerate(vocab_processor.fit_transform(x_raw)):
    y_data = [[y_raw[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix + 1) % 10 == 0:
        print('Training Observation #' + str(ix + 1) + ': Loss = ' + str(temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    y_pred = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # y_pred = list(itertools.chain(*y_pred))

    y_pred = [item for sublist in y_pred.tolist() for item in sublist]
    # print(y_pred)
    # Convert prediction and true vectors into an integer index
    pred_val = np.argmax(y_pred, axis=1)
    # print(pred_val)
    true_val = np.argmax(y_raw[ix])

    # Get True/False if prediction is accurate
    train_acc_temp = true_val == pred_val

    # Append accuracies to list
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))


"""Testing Step"""
print('Getting Test Set Accuracy For {} Sentences.'.format(len(x_test)))
test_acc_all = []
predictions = []
true_vals = []
for ix, t in enumerate(vocab_processor.fit_transform(x_test)):
    y_data = [[y_test[ix]]]
    # print("Y data: " + str(y_data))

    if (ix + 1) % 50 == 0:
        print('Test Observation #' + str(ix + 1))

        # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    y_pred = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})

    y_pred = [item for sublist in y_pred.tolist() for item in sublist]

    # Convert prediction and true vectors into an integer index
    pred_val = np.argmax(y_pred, axis=1)
    true_val = np.argmax(y_test[ix])

    # Get True/False if prediction is accurate
    test_acc_temp = true_val == pred_val

    # Append accuracies to list
    predictions.append(pred_val)
    true_vals.append(true_val)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

# Print classification report
print(classification_report(true_vals, predictions, target_names=classes))

# Create confusion matrix
cnf_matrix = confusion_matrix(true_vals, predictions)

plt.figure()
# Plot and store confusion matrix in a png file
data_helper.plot_confusion_matrix(cnf_matrix, labels=classes)
plt.show()

# Plot training accuracy over time
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
plt.title('Avg Training Acc Over Past 50 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.show()