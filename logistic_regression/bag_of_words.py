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
from data_helper import loadData, randomize_data, plot_confusion_matrix, load


# Start a graph session
sess = tf.Session()

file_name = 'iseardataset.csv'
test_set = './data/isear_test.csv'
train_set = './data/isear_train.csv'

# Load data
print("Loading data...")

# Load train/validation data
x_text, y = load(train_set)

# Load test data
x_test, y_test = load(test_set)

# Choose max text word length at 40 to cover most data
sentence_size = 40
min_word_freq = 3

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
vocab_processor.transform(x_text)
transformed_texts = np.array([x for x in vocab_processor.transform(x_text)])
embedding_size = len(np.unique(transformed_texts))
print("Total words: " + str(embedding_size))

# Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1, 1, 7]))

# Initialize placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1, 7], dtype=tf.float32)

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
print('Starting Training Over {} Sentences.'.format(len(x_text)))
loss_vec = []
train_acc_all = []
train_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(x_text)):
    y_data = [[y[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix + 1) % 10 == 0:
        print('Training Observation #' + str(ix + 1) + ': Loss = ' + str(temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = y[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# Get test set accuracy
print('Getting Test Set Accuracy For {} Sentences.'.format(len(x_text)))
test_acc_all = []
for ix, t in enumerate(vocab_processor.fit_transform(x_test)):
    y_data = [[y_test[ix]]]
    print("Y data: " + str(y_data))

    if (ix + 1) % 50 == 0:
        print('Test Observation #' + str(ix + 1))

        # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    print("Temp_pred: " + str(np.round(temp_pred)))
    # print("Y_pred: " + str(y_pred))
    test_acc_temp = y_test[ix] == np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

# classes = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]

# # Create confusion matrix
# cnf_matrix = confusion_matrix(y_data, y_pred)
# plt.figure(figsize=(20, 10))
# plot_confusion_matrix(cnf_matrix, labels=classes)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

# Plot training accuracy over time
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
plt.title('Avg Training Acc Over Past 50 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.show()