import os
import sys
import pickle

from NN.lstm import LSTM
from Util.load_input import LoadData
from Util.chars_auxiliary_attention import CharsAuxi

import argparse
import datetime
from Util.log import get_logger

log = get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word lstm')
    parser.add_argument('-u', '--train_user_id', default='northyorksammy')
    parser.add_argument('-f', '--file_path', default=os.environ['HOME'] + '/data/')
    parser.add_argument('-o', '--folder_name', default="{}/gcn/run_{}".format(os.environ['HOME'], datetime.datetime.now().isoformat()))
    parser.add_argument('--data_size', default='', type=str)
    parser.add_argument('--lstm_size', default=128, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.03, type=float)
    parser.add_argument('--keep_prob', default=0.3, type=float)
    parser.add_argument('--grad_clip', default=5, type=int)
    parser.add_argument('--num_batches', default=10, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs_start', default=0, type=int)
    parser.add_argument('--epochs_end', default=10, type=int)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--checkpoint_weights', default='')

    args = parser.parse_args()

    log.info('args: {}', args)

    # if True: exit(1)
    train_user_id = args.train_user_id
    file_path = args.file_path
    folder_name = args.folder_name

    data_size = args.data_size

    load = LoadData(file_path, data_size)
    words_auxi = CharsAuxi(load.input, load.auxiliary, load.user_attention, load.item_attention)

    # set parameters
    num_auxiliary = load.auxiliary.shape[1]
    vector_length = len(set(load.input))
    num_user_attention = len(set(load.user_attention))
    num_item_attention = len(set(load.item_attention))

    # num_batches = 10
    # batch_size = 10
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_char = load.input.shape[0] // (num_batches * batch_size)

    log.info("number of auxiliary: {}, vector length: {}, num char: {}, num attention: {}", num_auxiliary, vector_length, num_char, num_user_attention + num_item_attention)

    # lstm_size = 128
    # num_layers = 2
    # learning_rate = 0.03
    # keep_prob = 0.3
    # grad_clip = 5
    # epochs = 50
    lstm_size = args.lstm_size
    num_layers = args.num_layers
    learning_rate = args.learning_rate
    keep_prob = args.keep_prob
    grad_clip = args.grad_clip
    epochs_start = args.epochs_start
    epochs_end = args.epochs_end
    checkpoint = args.checkpoint
    checkpoint_weights = args.checkpoint_weights

    # generate concatenate one hot
    # x, y = words_auxi.split_1d_date(words_auxi.input, num_batches, batch_size, num_char)
    # auxiliary, _ = words_auxi.split_2d_date(words_auxi.auxiliary, num_batches, batch_size, num_char)
    # user_attention_x, _ = words_auxi.split_1d_date(words_auxi.user_attention, num_batches, batch_size, num_char)
    # item_attention_x, _ = words_auxi.split_1d_date(words_auxi.item_attention, num_batches, batch_size, num_char)

    # split to train test

    frac = .9

    num_train_batch = int(num_batches * frac)
    num_test_batch = num_batches - num_train_batch

    x, y, test_x, test_y = words_auxi.split_1d_date_by_frac(words_auxi.input, num_batches, batch_size, num_char, split_frac=frac)
    auxiliary, _, test_auxiliary, _ = words_auxi.split_2d_date_by_frac(words_auxi.auxiliary, num_batches, batch_size, num_char, split_frac=frac)
    user_attention_x, _, test_user_attention_x, _ = words_auxi.split_1d_date_by_frac(words_auxi.user_attention, num_batches, batch_size, num_char, split_frac=frac)
    item_attention_x, _, test_item_attention_x, _ = words_auxi.split_1d_date_by_frac(words_auxi.item_attention, num_batches, batch_size, num_char, split_frac=frac)

    log.info("{}, {}, {}, {}", x.shape, auxiliary.shape, user_attention_x.shape, item_attention_x.shape)
    log.info("{}, {}, {}, {}", test_x.shape, test_auxiliary.shape, test_user_attention_x.shape, test_item_attention_x.shape)


    # training
    lstm_model = LSTM(vector_length=vector_length,
                      num_auxiliary=num_auxiliary,
                      num_user_attention=num_user_attention,
                      num_item_attention=num_item_attention,
                      num_batches=num_batches,
                      batch_size=batch_size,
                      num_char=num_char,
                      lstm_size=lstm_size,
                      num_layers=num_layers,
                      learning_rate=learning_rate,
                      grad_clip=grad_clip)

    log.info("{}", checkpoint)

    # sys.exit(1)

    # lstm_model.train_save(folder_to_save=folder_name,
    #                       config_name=train_user_id,
    #                       trained_user_name=train_user_id,
    #                       words_auxi=words_auxi,
    #                       x=x,
    #                       y=y,
    #                       auxiliary=auxiliary,
    #                       user_attention=user_attention_x,
    #                       item_attention=item_attention_x,
    #                       keep_prob=keep_prob,
    #                       epochs_start=epochs_start,
    #                       epochs_end=epochs_end,
    #                       checkpoint=checkpoint,
    #                       checkpoint_weights=checkpoint_weights)

    lstm_model.train_test_save(folder_to_save=folder_name,
                               config_name=train_user_id,
                               trained_user_name=train_user_id,
                               words_auxi=words_auxi,
                               x=x,
                               y=y,
                               test_x=test_x,
                               test_y=test_y,
                               auxiliary=auxiliary,
                               test_auxiliary=test_auxiliary,
                               user_attention=user_attention_x,
                               test_user_attention_x=test_user_attention_x,
                               item_attention=item_attention_x,
                               test_item_attention_x=test_item_attention_x,
                               num_train_batch=num_train_batch,
                               num_test_batch=num_test_batch,
                               keep_prob=keep_prob,
                               epochs_start=epochs_start,
                               epochs_end=epochs_end,
                               checkpoint=checkpoint,
                               checkpoint_weights=checkpoint_weights)










