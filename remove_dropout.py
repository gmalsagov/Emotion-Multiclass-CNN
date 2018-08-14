#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from pandas import json
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np
import data_helper
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split


def preprocess(inputs, wordid):
    """Pre-process input data by cleaning text and mapping words onto integer id's."""

    # Load pre-trained vocabulary
    words_index = learn.preprocessing.VocabularyProcessor.restore(wordid)

    # Load inputs to be classified
    x_text, y, vocab, df = data_helper.load(inputs)

    # Map sentences onto exitsting vocabulary
    x = np.array(list(words_index.fit_transform(x_text)))

    return x, y


def open_graph(graph_dir):
    """Open TensorFlow graph from directory."""

    with tf.gfile.Open(graph_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def save_graph(graph_nodes, name):
    """Store TensorFlow graph."""
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(graph_nodes)
    with tf.gfile.GFile(name + '.pb', 'w') as f:
        f.write(output_graph.SerializeToString())


def display_nodes(graph_def):
    """Visualize TensorFlow graph."""
    # Check if graph has attribute 'as_graph_def'
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    for i, node in enumerate(graph_def.node):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]


def accuracy(predictions, labels):
    """Calculate Accuracy of TensorFlow graph."""
    return 100.0 * np.sum(predictions == np.argmax(labels, axis=1)) / predictions.shape[0]


def test_graph(graph_path, use_dropout):
    """Run script to test the graphs on given inputs."""

    tf.reset_default_graph()
    graph_def = tf.GraphDef()

    inputs = "data/isear_test.csv"

    vocabulary = graph_path
    if vocabulary.endswith('checkpoints'):
        vocabulary = vocabulary[:-11]
    vocabulary = vocabulary + 'vocabulary'

    sentences, labels = preprocess(vocabulary, inputs)

    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map = None,
                            return_elements = None,
                            name = ""
                            )
    sess = tf.Session(graph=graph)
    prediction_tensor = sess.graph.get_tensor_by_name('output/predictions:0')
    x = graph.get_tensor_by_name("input_x:0")

    feed_dict = {x: sentences[:len(sentences)]}
    if use_dropout:
        feed_dict['dropout_keep_prob:0'] = 1.0

    predictions = sess.run(prediction_tensor, feed_dict)
    # print(np.argmax(labels[:100], axis=1))
    # print(predictions)
    result = accuracy(predictions, labels[:len(sentences)])
    return result

def remove_dropout(graph_dir):

    # read frozen graph and display nodes
    # graph_dir = './cnn-embeddings/trained_model_1534172737/checkpoints/frozen_model.pb'

    graph = open_graph(graph_dir + 'frozen_model.pb')

    display_nodes(graph)

    if hasattr(graph, 'as_graph_def'):
        graph = graph.as_graph_def()

    # Connect #52 'output/scores/Matmul' node to output of 'Reshape' node #43
    graph.node[52].input[0] = 'prefix/Reshape'

    # Remove dropout nodes
    nodes = graph.node[:44] + graph.node[52:] # 44 -> output/scores

    del nodes[5] # 5 dropout
    del nodes[5] # 6 dropout
    del nodes[5] # 7 dpopout
    del nodes[23] # 23 -> keep_prob

    save_graph(nodes, graph_dir + 'frozen_model_no_dropout')

    processed_graph = open_graph(graph_dir + 'frozen_model_no_dropout.pb')

    print("New Graph:")
    print("")
    display_nodes(processed_graph)


graph_dir = './cnn-embeddings/trained_model_1534172737/checkpoints/'
#
# remove_dropout(graph_dir)

# Check model accuracy
result_1 = test_graph(graph_dir + 'frozen_model.pb', use_dropout=True)
result_2 = test_graph(graph_dir + 'frozen_embeddings_no_dropout.pb', use_dropout=False)
print('Accuracy with dropout: %f' % result_1)
print('Accuracy without dropout: %f' % result_2)

