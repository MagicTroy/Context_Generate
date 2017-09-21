import os
import json
import time
import numpy as np
import tensorflow as tf

from collections import namedtuple
from Util.log import get_logger

log = get_logger()

def save_log_checkpoint(folder_to_save, config_name, trained_user_name,
                        num_auxiliary, num_characters, num_batches, num_char,
                        batch_size, lstm_size, num_layers, learning_rate, keep_prob,
                        epochs, train_loss, ep_time, total_time,
                        sess):
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
        'epochs time': str(ep_time),
        'total_time': str(total_time)
    }
    re = []
    log.info('folder_to_save: {}', folder_to_save)
    if not os.path.exists(folder_to_save):
        re.append(results)
        os.makedirs(folder_to_save)
    else:
        with open('{}.json'.format(folder_to_save + '/' + str(config_name))) as data_input:
            re = json.load(data_input)
        re.append(results)

    with open('{}.json'.format(folder_to_save + '/' + str(config_name)), 'wt') as output:
        output.write(json.dumps(re, indent=4))

    # with tf.device('/cpu:0'):
    save_checkpoint(folder_to_save, sess, epochs)


def save_checkpoint(folder_to_save, sess, epochs):
    tf.train.Saver().save(sess, folder_to_save + '/' + 'e{}.ckpt'.format(epochs))


class LSTM:
    def __init__(self, num_characters, num_auxiliary, batch_size=50,
                 num_char=50, lstm_size=128, num_layers=2,
                 learning_rate=0.001, grad_clip=5, sampling=False):

        self.num_characters = num_characters
        self.num_auxiliary = num_auxiliary
        self.batch_size = batch_size
        self.num_char = num_char
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.sampling = sampling

        self.model = self._build_rnn(num_classes=self.num_characters,
                                     num_auxiliary=self.num_auxiliary,
                                     batch_size=self.batch_size,
                                     num_char=self.num_char,
                                     learning_rate=self.learning_rate,
                                     lstm_size=self.lstm_size,
                                     num_layers=self.num_layers,
                                     grad_clip=self.grad_clip,
                                     sampling=self.sampling)

    def _build_rnn(self, num_classes, num_auxiliary, batch_size=50, num_char=50, lstm_size=128, num_layers=2,
                   learning_rate=0.001, grad_clip=5, sampling=False):
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_char = 1, 1

        tf.reset_default_graph()

        # with tf.device('/cpu:0'):

        # Declare placeholders we'll feed into the graph
        inputs = tf.placeholder(tf.float32, [batch_size, num_char, num_classes + num_auxiliary], name='inputs')
        targets = tf.placeholder(tf.float32, [batch_size, num_char, num_classes], name='targets')

        x_one_hot = inputs
        y_one_hot = targets

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
            softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(num_classes))

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and batch

        logits = tf.matmul(output, softmax_w) + softmax_b

        # Use softmax to get the probabilities for predicted characters
        preds = tf.nn.softmax(logits, name='predictions')

        # Reshape the targets to match the logits
        # y_reshaped = tf.reshape(y_one_hot, [-1, num_classes+num_auxiliary])
        y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        cost = tf.reduce_mean(loss)

        # Optimizer for training, using gradient clipping to control exploding gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars, aggregation_method=2), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

        # Export the nodes
        export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                        'keep_prob', 'cost', 'preds', 'optimizer']
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

    def _fit_model(self, sess, chars_auxi, x, y, auxi_x, num_batches, keep_prob):
        loss = []
        new_state = sess.run(self.model.initial_state)

        start = time.time()
        for xx, yy in chars_auxi.get_concat_batch(x, y, auxi_x, num_batches, self.num_char, self.num_characters):
            feed = {self.model.inputs: xx,
                    self.model.targets: yy,
                    self.model.keep_prob: keep_prob,
                    self.model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([self.model.cost,
                                                 self.model.final_state,
                                                 self.model.optimizer],
                                                feed_dict=feed)
            loss.append(batch_loss)
            log.debug('loss: {}', loss)
            current = time.time()
            log.info('elapsed per batch {}, mean of loss = {}', current - start, np.mean(loss))
            start = current
        return np.mean(loss)

    def _training(self, sess, chars_auxi, train_x, train_y, auxi_x, num_batches, keep_prob):
        t_loss = self._fit_model(sess, chars_auxi, train_x, train_y, auxi_x, num_batches, keep_prob)

        #     print 'training'
        #     print training_loss
        #     print 'validate'
        #     print validate_loss

        return t_loss

    def load_weights(self, saver, checkpoint_weights):
        with tf.Session() as sess:
            print('load weight checkpoint')
            saver.restore(sess, checkpoint_weights)
            lstm_variables = tf.global_variables()
            return lstm_variables

    def train_save(self, folder_to_save, config_name, trained_user_name, chars_auxi,
                   train_x_one_hot, train_y_one_hot, auxi_x,
                   num_batches, keep_prob=.5, epochs_start=0, epochs_end=10, checkpoint='', checkpoint_weights=''):
        start = time.time()
        saver = tf.train.Saver()

        if checkpoint_weights != '':
            global_variables = self.load_weights(saver, checkpoint_weights)
        else:
            global_variables = []

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            if checkpoint_weights != '':
                sess.run(tf.initialize_variables(global_variables))
            else:
                sess.run(tf.global_variables_initializer())

            if checkpoint != '':
                print('load checkpoint')
                saver.restore(sess, checkpoint)

            for e in range(epochs_start, epochs_end):
                start_ep = time.time()

                # Train network
                training_loss = self._training(sess, chars_auxi,
                                               train_x_one_hot, train_y_one_hot,
                                               auxi_x, num_batches, keep_prob)

                end_ep = time.time()

                log.info('Epoch {}/{}/{} '.format(e, epochs_start, epochs_end),
                      'Training loss: {:.4f}'.format(training_loss),
                      '{:.4f} sec/epoch'.format((end_ep - start_ep)),
                      'Save checkpoint!')

                save_log_checkpoint(folder_to_save, config_name, trained_user_name,
                                    self.num_auxiliary, self.num_characters, num_batches,
                                    self.num_char, self.batch_size, self.lstm_size, self.num_layers,
                                    self.learning_rate, keep_prob,
                                    e, training_loss,
                                    end_ep - start_ep, end_ep - start,
                                    sess)















