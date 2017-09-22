import os
import sys

from string import printable
from NN.lstm_per_user import *
#from NN.gru_tf_onehot_auxiliary_attention_per_user import *
from Util.load_input import LoadCSV
from Util.chars_auxiliary_attention_per_user import CharsAuxi

import argparse
import datetime
from Util.log import get_logger

log = get_logger()

# printable = list(set(printable))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word lstm')
    parser.add_argument('-u', '--train_user_id', default='northyorksammy')
    parser.add_argument('-f', '--file_path', default=os.environ['HOME'] + '/data/')
    parser.add_argument('-o', '--folder_name', default="{}/gcn/run_{}".format(os.environ['HOME'], datetime.datetime.now().isoformat()))
    parser.add_argument('--num_item_attention', default=49005, type=int)
    parser.add_argument('--rnn_size', default=128, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.03, type=float)
    parser.add_argument('--keep_prob', default=0.3, type=float)
    parser.add_argument('--grad_clip', default=5, type=int)
    parser.add_argument('--num_batches', default=10, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs_start', default=0, type=int)
    parser.add_argument('--epochs_end', default=10, type=int)
    parser.add_argument('--rnn_tech', default='lstm')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--checkpoint_weights', default='')

    args = parser.parse_args()

    log.info('args: {}', args)

    # if True: exit(1)
    train_user_id = args.train_user_id
    file_path = args.file_path
    folder_name = args.folder_name

    load = LoadCSV(file_path, train_user_id)
    train_words_auxi = CharsAuxi(load.train_input, load.train_auxiliary, load.train_item_attention)
    test_words_auxi = CharsAuxi(load.test_input, load.test_auxiliary, load.test_item_attention)

    # set parameters
    num_auxiliary = load.train_auxiliary.shape[0]
    vector_length = len(printable)
    num_item_attention = args.num_item_attention

    # num_batches = 10
    # batch_size = 10
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_char = load.train_input.shape[0] // (num_batches * batch_size)

    log.info("number of auxiliary: {}, vector length: {}, num char: {}, num attention: {}", num_auxiliary, vector_length, num_char, num_item_attention)

    # lstm_size = 128
    # num_layers = 2
    # learning_rate = 0.03
    # keep_prob = 0.3
    # grad_clip = 5
    # epochs = 50
    rnn_size = args.rnn_size
    num_layers = args.num_layers
    learning_rate = args.learning_rate
    keep_prob = args.keep_prob
    grad_clip = args.grad_clip
    epochs_start = args.epochs_start
    epochs_end = args.epochs_end
    checkpoint = args.checkpoint
    checkpoint_weights = args.checkpoint_weights

    # generate concatenate one hot

    rnn_tech = args.rnn_tech

    folder_name = os.path.join(folder_name, train_user_id)

    x, y = train_words_auxi.split_1d_date(train_words_auxi.input, num_batches,
                                          batch_size, num_char)
    auxiliary, _ = train_words_auxi.split_2d_date(train_words_auxi.auxiliary.transpose(),
                                                  num_batches, batch_size, num_char)
    item_attention_x, _ = train_words_auxi.split_1d_date(train_words_auxi.item_attention,
                                                         num_batches, batch_size,
                                                         num_char)

    test_x, test_y = test_words_auxi.split_1d_date_test(test_words_auxi.input)
    test_auxiliary, _ = test_words_auxi.split_2d_date_test(test_words_auxi.auxiliary.transpose())
    test_item_attention_x, _ = test_words_auxi.split_1d_date_test(test_words_auxi.item_attention)

    start = time.time()

    for e in range(epochs_start, epochs_end):
        start_ep = time.time()

        training_loss = train_model(vector_length=vector_length, num_auxiliary=num_auxiliary,
                                    num_item_attention=num_item_attention, num_batches=num_batches,
                                    batch_size=batch_size, num_char=num_char, rnn_size=rnn_size,
                                    num_layers=num_layers, learning_rate=learning_rate,
                                    grad_clip=grad_clip, folder_to_save=folder_name,
                                    x=x, y=y, auxiliary=auxiliary, item_attention=item_attention_x,
                                    train_words_auxi=train_words_auxi, keep_prob=keep_prob, epochs=e)

        # test_loss = test_model(vector_length=vector_length, num_auxiliary=num_auxiliary,
        #                        num_item_attention=num_item_attention, num_batches=num_batches,
        #                        batch_size=batch_size, num_char=num_char, rnn_size=rnn_size,
        #                        num_layers=num_layers, learning_rate=learning_rate,
        #                        grad_clip=grad_clip, folder_to_load=folder_name,
        #                        x=test_x, y=test_y, auxiliary=test_auxiliary,
        #                        item_attention=test_item_attention_x,
        #                        keep_prob=keep_prob, epochs=e)
        test_loss = [0.0]

        train, test = np.mean(training_loss), np.mean(test_loss)

        end_ep = time.time()
        save_log(folder_name, train_user_id, train_user_id,
                 num_auxiliary, vector_length, num_batches,
                 num_char, batch_size, rnn_size, num_layers,
                 learning_rate, keep_prob,
                 e, train, test,
                 end_ep - start_ep, end_ep - start)

        log.info('Epoch {}/{}/{} '.format(e, epochs_start, epochs_end),
                 'Training loss: {:.4f}'.format(train), 'Test loss:{:.4f}'.format(test),
                 '{:.4f} sec/epoch'.format((end_ep - start_ep)),
                 'Save checkpoint!')









