#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_helper
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

batch_size = 200
# the maximum number of tf-idf textual words
max_features = 1000

# Relative path to train/test datasets
# train_file = '../data/iseardataset.csv'
test_set = '../data/isear_test.csv'
train_set = '../data/isear_train.csv'

# Classes to be predicted
classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

# Load train/validation data
x_train, y_train, _, _ = data_helper.load(train_set)
x_train = np.array(x_train)

# Load test data
x_test, y_test, _, _ = data_helper.load(test_set)
x_test = np.array(x_test)


# Define tokenizer function from NLTK
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


# Create TF-IDF vectors of texts
tf_idf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
x_train = tf_idf.fit_transform(x_train)
x_test = tf_idf.fit_transform(x_test)

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features, y_train.shape[1]]), name="weights")
b = tf.Variable(tf.random_normal(shape=[1, 1, y_train.shape[1]]), name="bias")

# Initialize placeholders
inputs_x = tf.placeholder(shape=[None, max_features], dtype=tf.float32, name='inputs')
target_y = tf.placeholder(shape=[1, None, y_train.shape[1]], dtype=tf.float32, name='predictions')
print()
# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(inputs_x, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=target_y))

# Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, target_y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

# Initialize Global Variables
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
y_preds = []
i_data = []

for i in range(10000):

    rand_index = np.random.choice(x_train.shape[0], size=batch_size)
    rand_x = x_train[rand_index].todense()
    rand_y = [y_train[rand_index]]
    sess.run(train_step, feed_dict={inputs_x: rand_x, target_y: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)

        train_loss_temp = sess.run(loss, feed_dict={inputs_x: rand_x, target_y: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={inputs_x: x_test.todense(), target_y: ([y_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={inputs_x: rand_x, target_y: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={inputs_x: x_test.todense(), target_y: ([y_test])})
        test_acc.append(test_acc_temp)

        prediction_temp = sess.run(prediction, feed_dict={inputs_x: x_test.todense(), target_y: ([y_test])})
        # print(prediction_temp)
        y_preds.append(prediction_temp)

    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

# # Plot loss over time
# plt.plot(i_data, train_loss, 'k-', label='Train Loss')
# plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
# plt.title('Cross Entropy Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Cross Entropy Loss')
# plt.legend(loc='upper right')
# plt.show()
#
# # Plot train and test accuracy
# plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
# plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
# plt.title('Train and Test Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()