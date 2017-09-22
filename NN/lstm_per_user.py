import os
import json
import time
import numpy as np
import tensorflow as tf

from collections import namedtuple
from Util.log import get_logger

log = get_logger()


def save_log(folder_to_save, config_name, trained_user_name,
             num_auxiliary, num_characters, num_batches, num_char,
             batch_size, lstm_size, num_layers, learning_rate, keep_prob,
             epochs, train_loss, test_loss, ep_time, total_time):
    results = {
        'epochs': epochs,
        'trained_user_name': trained_user_name,
        'number of auxiliary': num_auxiliary,
        'number of characters': num_characters,
        'number of batchs': num_batches,
        'number of characters per train': num_char,
        'batch_size': batch_size,
        'lstm_size': lstm_size,
        'num_layers': num_layers,
        'learning rate': learning_rate,
        'keep probability': keep_prob,
        'training loss': str(train_loss),
        'test loss': str(test_loss),
        'epochs time': str(ep_time),
        'total_time': str(total_time)
    }
    re = []

    log.info('folder_to_save: {}', folder_to_save)
    if not os.path.exists('{}.json'.format(folder_to_save + '/' + str(config_name))):
        re.append(results)
        # os.makedirs(folder_to_save)
    else:
        with open('{}.json'.format(folder_to_save + '/' + str(config_name))) as data_input:
            re = json.load(data_input)
        re.append(results)

    with open('{}.json'.format(folder_to_save + '/' + str(config_name)), 'wt') as output:
        output.write(json.dumps(re, indent=4))


def save_checkpoint(folder_to_save, sess, epochs):
    tf.train.Saver().save(sess, folder_to_save + '/' + 'e{}.ckpt'.format(epochs))
    return folder_to_save + '/' + 'e{}.ckpt'.format(epochs)


class LSTM:
    def __init__(self, vector_length, num_auxiliary, num_item_attention,
                 num_batches=10, batch_size=50, num_char=50, lstm_size=128, num_layers=2,
                 learning_rate=0.001, grad_clip=5, sampling=False):

        self.vector_length = vector_length
        self.num_auxiliary = num_auxiliary
        self.num_item_attention = num_item_attention
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_char = num_char
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.sampling = sampling

        self.model = self._build_rnn(vector_length=self.vector_length,
                                     num_auxiliary=self.num_auxiliary,
                                     num_item_attention=self.num_item_attention,
                                     batch_size=self.batch_size,
                                     num_char=self.num_char,
                                     learning_rate=self.learning_rate,
                                     lstm_size=self.lstm_size,
                                     num_layers=self.num_layers,
                                     grad_clip=self.grad_clip,
                                     sampling=self.sampling)

    def _build_rnn(self, vector_length, num_auxiliary, num_item_attention, batch_size=50,
                   num_char=50, lstm_size=128, num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False):
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_char = 1, 1

        tf.reset_default_graph()

        # with tf.device('/cpu:0'):

        # Declare placeholders we'll feed into the graph
        inputs = tf.placeholder(tf.int32, [batch_size, num_char], name='inputs')
        targets = tf.placeholder(tf.int32, [batch_size, num_char], name='targets')

        auxiliary = tf.placeholder(tf.float32, [batch_size, num_char, num_auxiliary], name='auxiliary')
        item_attention = tf.placeholder(tf.int32, [batch_size, num_char], name='item_attention')

        x_one_hot = tf.one_hot(inputs, vector_length)
        y_one_hot = tf.one_hot(targets, vector_length)

        item_attention_one_hot = tf.one_hot(item_attention, num_item_attention)

        x_one_hot = tf.concat([x_one_hot, auxiliary], axis=2)
        x_one_hot = tf.concat([x_one_hot, item_attention_one_hot], axis=2)

        # Keep probability placeholder for drop out layers
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        ### Build the RNN layers
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        # prevent overfitting
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        # self.init_lstm()

        initial_state = cell.zero_state(batch_size, tf.float32)

        ### Run the data through the RNN layers
        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
        final_state = state

        # Reshape output so it's a bunch of rows, one output row for each step for each batch
        seq_output = tf.concat(outputs, axis=1)
        output = tf.reshape(seq_output, [-1, lstm_size])

        # Now connect the RNN outputs to a softmax layer
        with tf.variable_scope('softmax'):
            # softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes+num_auxiliary), stddev=0.1))
            # softmax_b = tf.Variable(tf.zeros(num_classes+num_auxiliary))
            softmax_w = tf.Variable(tf.truncated_normal((lstm_size, vector_length), stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(vector_length))

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and batch

        logits = tf.matmul(output, softmax_w) + softmax_b

        # Use softmax to get the probabilities for predicted characters
        preds = tf.nn.softmax(logits, name='predictions')

        # Reshape the targets to match the logits
        # y_reshaped = tf.reshape(y_one_hot, [-1, num_classes+num_auxiliary])
        y_reshaped = tf.reshape(y_one_hot, [-1, vector_length])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        cost = tf.reduce_mean(loss)

        # Optimizer for training, using gradient clipping to control exploding gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars, aggregation_method=0), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

        # Export the nodes
        export_nodes = ['inputs', 'targets', 'auxiliary', 'item_attention', 'initial_state',
                        'final_state', 'keep_prob', 'cost', 'preds', 'optimizer']
        Graph = namedtuple('Graph', export_nodes)
        local_dict = locals()
        graph = Graph(*[local_dict[each] for each in export_nodes])

        return graph

    def init_lstm(self):
        with tf.variable_scope("LSTM") as vs:
            # Execute the LSTM cell here in any way, for example:
            lstm_variables = [v for v in tf.all_variables()
                              if v.name.startswith(vs.name)]

            print(lstm_variables)

    def load_weights(self, saver, checkpoint_weights):
        with tf.Session() as sess:
            print('load weight checkpoint')
            saver.restore(sess, checkpoint_weights)
            lstm_variables = tf.global_variables()
            return lstm_variables


    # def train_test_fit_model(self, sess, words_auxi, x, y, test_x, test_y, auxiliary, user_attention, item_attention, keep_prob):
    #     loss, test_loss = [],[]
    #     new_state = sess.run(self.model.initial_state)
    #
    #     start = time.time()
    #     for xx, yy, a_x, user_att_x, item_att_x in words_auxi.get_auxi_attention_batch(x, y, auxiliary,
    #                                                                                    user_attention,
    #                                                                                    item_attention,
    #                                                                                    self.num_batches,
    #                                                                                    self.num_char):
    #         feed = {self.model.inputs: xx,
    #                 self.model.targets: yy,
    #                 self.model.auxiliary: a_x,
    #                 self.model.user_attention: user_att_x,
    #                 self.model.item_attention: item_att_x,
    #                 self.model.keep_prob: keep_prob,
    #                 self.model.initial_state: new_state}
    #         batch_loss, new_state, _ = sess.run([self.model.cost,
    #                                              self.model.final_state,
    #                                              self.model.optimizer],
    #                                             feed_dict=feed)
    #
    #         feed = {self.model.inputs: test_x,
    #                 self.model.targets: test_y,
    #                 self.model.auxiliary: a_x,
    #                 self.model.user_attention: user_att_x,
    #                 self.model.item_attention: item_att_x,
    #                 self.model.keep_prob: keep_prob,
    #                 self.model.initial_state: new_state}
    #         t_loss, new_state, _ = sess.run([self.model.cost,
    #                                          self.model.final_state,
    #                                          self.model.optimizer],
    #                                         feed_dict=feed)
    #
    #         loss.append(batch_loss)
    #         test_loss.append(t_loss)
    #         log.debug('loss: {}', loss)
    #         current = time.time()
    #         log.info('elapsed per batch {}, training loss = {}, test loss = {}', current - start, loss, t_loss)
    #         start = current
    #     return np.mean(loss), np.mean(test_loss)

    # def train_test_save(self, folder_to_save, config_name, trained_user_name, train_words_auxi, test_words_auxi,
    #                     keep_prob=.5, epochs_start=0, epochs_end=10,
    #                     checkpoint='', checkpoint_weights=''):
    #     start = time.time()
    #     saver = tf.train.Saver()
    #
    #     if checkpoint_weights != '':
    #         global_variables = self.load_weights(saver, checkpoint_weights)
    #     else:
    #         global_variables = []
    #
    #     # get train data
    #
    #     x, y = train_words_auxi.split_1d_date(train_words_auxi.input, self.num_batches,
    #                                           self.batch_size, self.num_char)
    #     auxiliary, _ = train_words_auxi.split_2d_date(train_words_auxi.auxiliary.transpose(),
    #                                                   self.num_batches, self.batch_size, self.num_char)
    #     item_attention_x, _ = train_words_auxi.split_1d_date(train_words_auxi.item_attention,
    #                                                          self.num_batches, self.batch_size,
    #                                                          self.num_char)
    #
    #     # # get test data
    #     #
    #     # num_test_batch = len(test_words_auxi.input) // (self.batch_size * self.num_char)
    #     #
    #     # test_x, test_y = test_words_auxi.split_1d_date(test_words_auxi.input, num_test_batch,
    #     #                                                self.batch_size, self.num_char)
    #     # test_auxiliary, _ = test_words_auxi.split_2d_date(test_words_auxi.auxiliary.transpose(),
    #     #                                                   num_test_batch, self.batch_size, self.num_char)
    #     # test_item_attention_x, _ = test_words_auxi.split_1d_date(test_words_auxi.item_attention,
    #     #                                                          num_test_batch, self.batch_size,
    #     #                                                          self.num_char)
    #
    #     log.info("{}, {}, {}, {}", x.shape, auxiliary.shape, item_attention_x.shape)
    #
    #     with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #         if checkpoint_weights != '':
    #             sess.run(tf.initialize_variables(global_variables))
    #         else:
    #             sess.run(tf.global_variables_initializer())
    #
    #         if checkpoint != '':
    #             print('load checkpoint')
    #             saver.restore(sess, checkpoint)
    #
    #         for e in range(epochs_start, epochs_end):
    #             start_ep = time.time()
    #
    #             # Train network
    #
    #             training_loss = []
    #             new_state = sess.run(self.model.initial_state)
    #
    #             i = 0
    #             # for xx, yy, a_x, item_att_x in train_words_auxi.get_auxi_batch(
    #             #         x, y, auxiliary,
    #             #         item_attention_x,
    #             #         self.num_batches,
    #             #         self.num_char):
    #             #
    #             #     start_batch = time.time()
    #             #     feed = {self.model.inputs: xx,
    #             #             self.model.targets: yy,
    #             #             self.model.auxiliary: a_x,
    #             #             self.model.item_attention: item_att_x,
    #             #             self.model.keep_prob: keep_prob,
    #             #             self.model.initial_state: new_state}
    #             #     batch_loss, new_state, _ = sess.run([self.model.cost,
    #             #                                          self.model.final_state,
    #             #                                          self.model.optimizer],
    #             #                                         feed_dict=feed)
    #             #
    #             #     training_loss.append(batch_loss)
    #             #     log.debug('loss: {}', training_loss)
    #             #     current = time.time()
    #             #     log.info('in epochs {}, batch {} / training batch {}, elapsed per batch {}, training loss = {}',
    #             #              e, i, self.num_batches, current - start_batch, np.mean(training_loss))
    #             #     i += 1
    #             #
    #             #     break
    #
    #             # i = 0
    #             # for xx, yy, a_x, item_att_x in test_words_auxi.get_auxi_batch(
    #             #         test_x, test_y, test_auxiliary,
    #             #         test_item_attention_x,
    #             #         num_test_batch,
    #             #         self.num_char):
    #             #     start_batch = time.time()
    #             #     feed = {self.model.inputs: xx,
    #             #             self.model.targets: yy,
    #             #             self.model.auxiliary: a_x,
    #             #             self.model.item_attention: item_att_x,
    #             #             self.model.keep_prob: keep_prob,
    #             #             self.model.initial_state: new_state}
    #             #     t_loss = sess.run([self.model.cost],
    #             #                       feed_dict=feed)
    #             #
    #             #     test_loss.append(t_loss)
    #             #     log.debug('loss: {}', test_loss)
    #             #     current = time.time()
    #             #     log.info('in epochs {}, batch {] / test batch {}, elapsed per batch {}, test loss = {}',
    #             #              e, i, num_test_batch, current - start_batch, np.mean(test_loss))
    #             #     i += 1
    #
    #             # end training and test
    #
    #             end_ep = time.time()
    #
    #             # if not os.path.exists(os.path.join(folder_to_save, trained_user_name)):
    #             #     os.makedirs(os.path.join(folder_to_save, trained_user_name))
    #             # checkpoint_path = save_checkpoint(os.path.join(folder_to_save, trained_user_name), sess, e)
    #             checkpoint_path = '{}.json'.format(folder_to_save + '/' + str(config_name))
    #
    #             test_loss = self.test(self.num_char, self.num_auxiliary, self.num_item_attention,self.lstm_size, checkpoint_path, test_words_auxi)
    #
    #             print(test_loss)
    #
    #             train, test = np.mean(training_loss), np.mean(test_loss)
    #
    #             log.info('Epoch {}/{}/{} '.format(e, epochs_start, epochs_end),
    #                      'Training loss: {:.4f}'.format(train), 'Test loss:{:.4f}'.format(test),
    #                      '{:.4f} sec/epoch'.format((end_ep - start_ep)),
    #                      'Save checkpoint!')
    #
    #             # save_log(folder_to_save, config_name, trained_user_name,
    #             #         self.num_auxiliary, self.vector_length, self.num_batches,
    #             #         self.num_char, self.batch_size, self.lstm_size, self.num_layers,
    #             #         self.learning_rate, keep_prob,
    #             #         e, train, test,
    #             #         end_ep - start_ep, end_ep - start)
    #
    # def test(self, vector_length, num_auxiliary, num_item_attention, lstm_size, checkpoint, test_words_auxi):
    #     test_lstm = LSTM(vector_length=vector_length,
    #                      num_auxiliary=num_auxiliary,
    #                      num_item_attention=num_item_attention,
    #                      lstm_size=lstm_size,
    #                      sampling=True)
    #     test_loss = []
    #
    #     saver = tf.train.Saver()
    #     with tf.Session() as sess_test:
    #         saver.restore(sess_test, checkpoint)
    #
    #         new_state = sess_test.run(test_lstm.model.initial_state)
    #
    #         for char, auxi, att in zip(test_words_auxi.input, test_words_auxi.auxiliary.transpose(), test_words_auxi.item_attention):
    #             char = np.array([char]).reshape(1, 1)
    #             auxi = np.array(auxi).reshape(1, 1, -1)
    #             att = np.array([att]).reshape(1, 1)
    #
    #             feed = {test_lstm.model.inputs: char,
    #                     test_lstm.model.auxiliary: auxi,
    #                     test_lstm.model.item_attention: att,
    #                     test_lstm.model.keep_prob: 1.,
    #                     test_lstm.model.initial_state: new_state}
    #
    #             t_loss = sess_test.run([self.model.cost],
    #                                   feed_dict=feed)
    #             test_loss.append(t_loss)
    #
    #     return test_loss


def train_model(vector_length, num_auxiliary, num_item_attention, num_batches, batch_size, num_char, rnn_size,
                num_layers, learning_rate, grad_clip, folder_to_save,
                x, y, auxiliary, item_attention, train_words_auxi, keep_prob=.5, epochs=0):

    rnn_model = LSTM(vector_length=vector_length,
                     num_auxiliary=num_auxiliary,
                     num_item_attention=num_item_attention,
                     num_batches=num_batches,
                     batch_size=batch_size,
                     num_char=num_char,
                     lstm_size=rnn_size,
                     num_layers=num_layers,
                     learning_rate=learning_rate,
                     grad_clip=grad_clip)

    saver = tf.train.Saver(restore_sequentially=True)
    training_loss = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if epochs != 0:
            saver.restore(sess, folder_to_save + '/' + 'e{}.ckpt'.format(epochs-1))

        new_state = sess.run(rnn_model.model.initial_state)

        i = 0
        for xx, yy, a_x, item_att_x in train_words_auxi.get_auxi_batch(
                x, y, auxiliary,
                item_attention,
                num_batches,
                num_char):

            start_batch = time.time()
            feed = {rnn_model.model.inputs: xx,
                    rnn_model.model.targets: yy,
                    rnn_model.model.auxiliary: a_x,
                    rnn_model.model.item_attention: item_att_x,
                    rnn_model.model.keep_prob: keep_prob,
                    rnn_model.model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([rnn_model.model.cost,
                                                 rnn_model.model.final_state,
                                                 rnn_model.model.optimizer],
                                                feed_dict=feed)

            training_loss.append(batch_loss)
            log.debug('loss: {}', training_loss)
            current = time.time()
            log.info('in epochs {}, batch {} / training batch {}, elapsed per batch {}, training loss = {}',
                     epochs, i, num_batches, current - start_batch, batch_loss)
            i += 1

        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        save_checkpoint(folder_to_save, sess, epochs)

    return training_loss


def test_model(vector_length, num_auxiliary, num_item_attention, num_batches, batch_size, num_char, rnn_size,
               num_layers, learning_rate, grad_clip, folder_to_load,
               x, y, auxiliary, item_attention, keep_prob=1, epochs=0):

    test_model = LSTM(vector_length=vector_length,
                      num_auxiliary=num_auxiliary,
                      num_item_attention=num_item_attention,
                      num_batches=num_batches,
                      batch_size=batch_size,
                      num_char=num_char,
                      lstm_size=rnn_size,
                      num_layers=num_layers,
                      learning_rate=learning_rate,
                      grad_clip=grad_clip,
                      sampling=True)
    saver = tf.train.Saver(restore_sequentially=True)
    test_loss = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, folder_to_load + '/' + 'e{}.ckpt'.format(epochs))

        new_state = sess.run(test_model.model.initial_state)

        for xx, yy, auxi, att in zip(x, y, auxiliary, item_attention):
            xx = np.array([xx]).reshape(1, 1)
            yy = np.array([yy]).reshape(1, 1)
            auxi = np.array(auxi).reshape(1, 1, -1)
            att = np.array([att]).reshape(1, 1)

            feed = {test_model.model.inputs: xx,
                    test_model.model.targets: yy,
                    test_model.model.auxiliary: auxi,
                    test_model.model.item_attention: att,
                    test_model.model.keep_prob: keep_prob,
                    test_model.model.initial_state: new_state}

            t_loss = sess.run([test_model.model.cost],
                              feed_dict=feed)
            test_loss.append(t_loss)

    return test_loss


















