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

    # Map sentences onto existing pre-trained vocabulary
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
        tf.import_graph_def(graph_def, name="")

    return graph


def save_graph(graph_nodes, name):
    """Store TensorFlow graph."""

    # Create default graph
    output_graph = graph_pb2.GraphDef()

    # Add nodes to default graph
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

    # Converts one-hot vector predictions into integers and compares with original labels
    return 100.0 * np.sum(predictions == np.argmax(labels, axis=1)) / predictions.shape[0]


def test_graph(model_path, graph_path, use_dropout):
    """Run script to test the graphs' accuracy on given inputs."""

    tf.reset_default_graph()
    graph_def = tf.GraphDef()

    # Specify test dataset
    inputs = "data/isear_test.csv"

    # Modify path to vocabulary
    vocabulary = model_path
    if vocabulary.endswith('checkpoints/'):
        vocabulary = vocabulary[:-12]
    vocabulary = vocabulary + 'vocabulary.pickle'

    # Preprocess inputs and labels
    sentences, labels = preprocess(inputs, vocabulary)

    # Read the graph
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map = None,
                            return_elements = None,
                            name = ""
                            )
    # Create TF session and specify input nodes
    sess = tf.Session(graph=graph)
    prediction_tensor = sess.graph.get_tensor_by_name('output/predictions:0')
    x = graph.get_tensor_by_name("input_x:0")

    # Create input dictionary
    feed_dict = {x: sentences[:len(sentences)]}
    if use_dropout:
        feed_dict['dropout_keep_prob:0'] = 1.0

    # Extract predictions from the graph
    predictions = sess.run(prediction_tensor, feed_dict)

    # Calculate accuracy of predictions
    result = accuracy(predictions, labels[:len(sentences)])

    return result


def remove_dropout(graph_dir):
    """Function removes dropout node from frozen model

    Note: this procedure is not automatic since model may have
    different structure with dropout layer in several places.

    Therefore, one must be careful when using this function"""


    # read frozen graph and display nodes
    graph = open_graph(graph_dir + 'frozen_model.pb')
    display_nodes(graph)

    if hasattr(graph, 'as_graph_def'):
        graph = graph.as_graph_def()

    # Connect #52 'output/scores/Matmul' node to output of 'Reshape' node #43
    graph.node[51].input[0] = 'Reshape'

    # Remove dropout nodes
    nodes = graph.node[:36] + graph.node[47:] # 46 -> output/scores

    # Delete dropout placeholder node
    del nodes[1]

    # Save graph in the same directory
    save_graph(nodes, graph_dir + 'frozen_model_no_dropout')

    # Load newly created graph
    processed_graph = open_graph(graph_dir + 'frozen_model_no_dropout.pb')

    if hasattr(processed_graph, 'as_graph_def'):
        processed_graph = processed_graph.as_graph_def()

    print("\nNew Graph:")
    print("")

    # Visualize new graph
    display_nodes(processed_graph)

# Specify paths to model, frozen graph and stripped graph
model_dir = './cnn-embeddings/trained_model_1534255535/checkpoints/'
graph_dir1 = model_dir + 'frozen_model.pb'
graph_dir2 = model_dir + 'frozen_model_no_dropout.pb'

# Remove dropout layer from the graph (might have to do it manually)
# remove_dropout(model_dir)

# Check model accuracy
result_1 = test_graph(model_dir, graph_dir1, use_dropout=True)
result_2 = test_graph(model_dir, graph_dir2, use_dropout=False)
print('Accuracy with dropout: %f' % result_1)
print('Accuracy without dropout: %f' % result_2)

