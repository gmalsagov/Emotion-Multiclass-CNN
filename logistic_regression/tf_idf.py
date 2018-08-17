#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import os
import string
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops
ops.reset_default_graph()

from data_helper import loadData, randomize_data, tokenizer

# Start a graph session
sess = tf.Session()


batch_size = 200
# the maximum number of tf-idf textual words
max_features = 1000

file_name = 'iseardataset.csv'
label_dict = {'joy': 4, 'shame': 6, 'sadness': 5, 'guilt': 3, 'disgust': 1, 'anger': 0, 'fear': 2}
dictionary, categories = loadData(file_name)

# Randomly create training and testing datasets with uniform distribution of each category samples
train_data, test_data = randomize_data(dictionary, categories)


texts = train_data.keys()
texts = texts + test_data.keys()
targets = train_data.values()
targets = targets + test_data.values()

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)

text_train = tfidf.fit_transform(train_data.keys())
text_test = tfidf.fit_transform(test_data.keys())
target_train = train_data.values()
target_train = np.array(pd.get_dummies(target_train, columns=label_dict).values.tolist())
target_test = test_data.values()
target_test = pd.get_dummies(target_test, columns=label_dict).values.tolist()


# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features, 7]), name="weights")
b = tf.Variable(tf.random_normal(shape=[1, 1, 7]), name="bias")

# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None, 7], dtype=tf.float32)

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_train.shape[0], size=batch_size)
    rand_x = text_train[rand_index].todense()
    # print(target_train[rand_index])
    rand_y = ([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: text_test.todense(), y_target: ([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={x_data: text_test.todense(), y_target: ([target_test])})
        test_acc.append(test_acc_temp)
    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()