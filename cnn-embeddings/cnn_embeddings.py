import tensorflow as tf
import numpy as np

class CNNEmbeddings(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda):
        # Placeholders for input, output and dropout
        # print sequence_length
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # self.predictions = tf.placeholder(tf.float32, name='predictions')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer tf.device('/cpu:0'),
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                   -1/embedding_size, 1/embedding_size), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # This is redundant but kept for reference later
            # self.embedded_chars_exp = tf.expand_dims(self.embedded_chars, -1)

            # Determine shape of a tensor as list
            shape = self.embedded_chars.get_shape().as_list()

            # Remove None from list since reshape doesn't support NoneType as input
            dim = shape[1:]

            # Add -1 to provide non-fixed size and append 1 to produce a 4d list of ints
            dim = [-1] + dim + [1]

            print(dim)

            self.embedded_chars_expanded = tf.reshape(self.embedded_chars, dim)

            shape = self.embedded_chars_expanded.get_shape().as_list()
            print(shape)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                # Apply nonlinearity
                # h = tf.nn.tanh(tf.nn.bias_add(conv, b), name = 'tanh')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            # self.precision = precision_score(tf.argmax(self.input_y, 1), self.predictions)
            # self.recall = recall_score(tf.argmax(self.input_y, 1), self.predictions)
            # self.f1_score = f1_score(tf.argmax(self.input_y, 1), self.predictions)

        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
