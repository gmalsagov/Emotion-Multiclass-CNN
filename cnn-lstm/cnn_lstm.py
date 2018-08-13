import tensorflow as tf
import numpy as np


def f1(cell, dropout_keep_prob): return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)


def f2(cell): return cell

class TextCNNLSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda, max_pool_size, hidden_unit):
        # Placeholders for input, output and dropout
        # print sequence_length
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding laywier
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            # self.W = tf.Variable(embedding_mat, name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        conv_outputs = []
        reduced = np.int32(np.ceil(sequence_length * 1.0 / max_pool_size))
        print(reduced)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):

                # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, self.embedded_chars_expanded, pad_post], 1)

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    emb_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')

                # Apply nonlinearity
                # h = tf.nn.tanh(tf.nn.bias_add(conv, b), name = 'tanh')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                """
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_pool_size, 1, 1],
                    strides=[1, max_pool_size, 1, 1],
                    padding='VALID',
                    name='pool')
                """
                h = tf.reshape(h, [-1, sequence_length, num_filters])
                # out = tf.reduce_max(h, 2)  # [-1, L, d]
                conv_outputs.append(h)

        conv_outputs = tf.concat(conv_outputs, 2)

        # pooled_outputs = tf.nn.dropout(pooled_outputs, self.dropout_keep_prob)

        lstm_cell = tf.contrib.rnn.GRUCell(num_units=hidden_unit)

        # def f1(cell, dropout_keep_prob): return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        #
        # def f2(cell): return cell
        #
        # if self.is_training:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
        # lstm_cell = tf.cond(self.is_training, lambda: f1(lstm_cell, self.dropout_keep_prob), lambda: f2(lstm_cell))
        # print('hello')

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

        inputs = [tf.squeeze(input_, [1]) for input_ in
                  tf.split(conv_outputs, num_or_size_splits=int(sequence_length), axis=1)]

        outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state,
                                                   sequence_length=self.real_len)

        output = outputs[0]

        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[hidden_unit, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, W, b, name='scores')
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
