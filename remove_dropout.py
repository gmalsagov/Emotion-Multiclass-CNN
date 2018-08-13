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


def preprocess(wordid):
    file = "data/isear_test.csv"

    words_index = learn.preprocessing.VocabularyProcessor.restore(wordid)


    x_text, y, vocab, df = data_helper.load(file)

    # x_text = data_helper.pad_sentences(x_text, forced_sequence_length=179)

    # Build vocabulary
    max_document_length = max([len(x) for x in x_text])
    # vocab_processor = learn.preprocessing.VocabularyProcessor(179)
    x = np.array(list(words_index.fit_transform(x_text)))

    return x, y


def open_graph(graph_dir):

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

def save_graph(graph_nodes, name):
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(graph_nodes)
    with tf.gfile.GFile(name + '.pb', 'w') as f:
        f.write(output_graph.SerializeToString())


def display_nodes(graph):

    for i, node in enumerate(graph.node):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]


def accuracy(predictions, labels):
    return 100.0 * np.sum(predictions == np.argmax(labels, axis=1)) / predictions.shape[0]


def test_graph(graph_path, use_dropout):
    tf.reset_default_graph()
    graph_def = tf.GraphDef()

    sentences, labels = preprocess('cnn-embeddings/trained_model_1534172737/vocabulary')

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


# read frozen graph and display nodes
graph_dir = './cnn-embeddings/trained_model_1534172737/checkpoints/frozen_model.pb'

graph = open_graph(graph_dir)

display_nodes(graph)
#
# # Connect #52 'output/scores/Matmul' node to output of 'Reshape' node #43
# graph.node[52].input[0] = 'Reshape'
#
# # Remove dropout nodes
# nodes = graph.node[:44] + graph.node[52:] # 44 -> output/scores
#
# del nodes[5] # 5 dropout
# del nodes[5] # 6 dropout
# del nodes[5] # 7 dpopout
# del nodes[23] # 23 -> keep_prob
#
# save_graph(nodes, 'cnn-embeddings/64%_trained_model/checkpoints/frozen_embeddings_no_dropout')
#
# processed_graph = open_graph('./cnn-embeddings/64%_trained_model/checkpoints/frozen_embeddings_no_dropout.pb')
#
# print("New Graph:")
# print("")
# display_nodes(processed_graph)

# # Check model accuracy
# result_1 = test_graph('./cnn-embeddings/trained_model_1534172737/checkpoints/frozen_model.pb', use_dropout=True)
# result_2 = test_graph('./cnn-embeddings/trained_model_1534163991/checkpoints/frozen_embeddings_no_dropout.pb', use_dropout=False)
# print('Accuracy with dropout: %f' % result_1)
# # print('Accuracy without dropout: %f' % result_2)
#
