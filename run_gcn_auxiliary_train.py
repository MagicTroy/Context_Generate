
import os
from NN.lstm import LSTM
from Util.load_data import LoadData
from Util.chars_auxiliary import CharsAuxi

import argparse
import datetime
from Util.log import get_logger
log = get_logger()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='char rnn')
    parser.add_argument('-u', '--train_user_id', default='northyorksammy')
    parser.add_argument('-f', '--file_path', default=os.environ['HOME'] + '/data/' + 'beeradvocate.db')
    parser.add_argument('-o', '--folder_name', default="{}/gcn/run_{}".format(os.environ['HOME'], datetime.datetime.now().isoformat()))
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

    #if True: exit(1)
    train_user_id = args.train_user_id
    file_path = args.file_path
    folder_name = args.folder_name

    #train_user_id = sys.argv[1]
    #    file_path = sys.argv[2]
    #   folder_name = sys.argv[3]

    # load data and generate chars and normalize auxiliary
    # load = LoadData(file_path)
    # training_df = load.get_df_by_userids([train_user_id])
    # chars_auxi = CharsAuxi(LoadData(file_path).get_df_by_userids([train_user_id]))

    chars_auxi = CharsAuxi(LoadData(file_path).df_total)

    # set parameters
    num_auxiliary = chars_auxi.num_auxiliary
    num_characters = len(chars_auxi.char_to_int)

    # num_batches = 10
    # batch_size = 10
    num_batches = args.num_batches
    batch_size = args.batch_size
    num_char = len(chars_auxi.chars_review) // (num_batches * batch_size)

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
    x, y, auxi_x_one_hot = chars_auxi.get_xy_auxiliary(num_batches, batch_size, num_char)

    # train_x_one_hot, train_y_one_hot = chars_auxi.get_concate_data(num_characters=num_characters,
    #                                                                num_batches=num_batches,
    #                                                                batch_size=batch_size,
    #                                                                num_char=num_char)

    log.info("{}, {}", x.shape, auxi_x_one_hot.shape)

    # training
    lstm_model = LSTM(num_characters=num_characters,
                      num_auxiliary=num_auxiliary,
                      batch_size=batch_size,
                      num_char=num_char,
                      lstm_size=lstm_size,
                      num_layers=num_layers,
                      learning_rate=learning_rate,
                      grad_clip=grad_clip)

    log.info("{}", checkpoint)

    lstm_model.train_save(folder_to_save=folder_name,
                          config_name=train_user_id,
                          trained_user_name=train_user_id,
                          chars_auxi=chars_auxi,
                          train_x_one_hot=x,
                          train_y_one_hot=y,
                          auxi_x=auxi_x_one_hot,
                          num_batches=num_batches,
                          keep_prob=keep_prob,
                          epochs_start=epochs_start,
                          epochs_end=epochs_end,
                          checkpoint=checkpoint,
                          checkpoint_weights=checkpoint_weights)










