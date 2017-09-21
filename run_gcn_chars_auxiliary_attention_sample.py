
import numpy as np
import pickle
import pandas as pd

from string import printable
from Apply.sample_auxiliary_attention import *

data_file = '/Users/troy/PhD_Research/Deep_Learning_with_Recommendation_system/GCN'
printable = list(set(printable))

def generate_partial_param(uid, iid):
    """
    :return:
    number of user attention
    number of item attention
    number of vector length of words
    prime input
    words set
    """

    user2user_id = pickle.load(open(data_file + '/words_data/user2user_id.pkl', 'rb'))
    item2item_id = pickle.load(open(data_file + '/words_data/item2item_id.pkl', 'rb'))

    # attention size
    num_user_attention = len(user2user_id)
    num_item_attention = len(item2item_id)

    # for generate text
    charac2int = pickle.load(open(data_file + '/chars_data/charac_to_int.pkl', 'rb'))
    int2charac = pickle.load(open(data_file + '/chars_data/int_to_charac.pkl', 'rb'))

    # attention input
    user_id = user2user_id[list(uid)[0]]
    item_id = item2item_id[list(iid)[0]]

    prime_input = generate_prime(charac2int, '_str_')

    return num_user_attention, num_item_attention, prime_input, user_id, item_id, charac2int, int2charac


def generate_prime(charac2int, prime):
    prime_input = []
    for p in prime:
        prime_input.append(charac2int[p])
    return prime_input


def generate_text_by_words_id(words_index, int2charac):
    text = []
    for i in words_index:
        # if int2charac[i] == '_end_' or int2charac[i] == '_str_':
        #     break
        text.append(int2charac[i])
    return ''.join(text)


def generate_sample(n_samples, checkpoint_folder, num_epochs,
                    user_id, item_id, auxiliary):
    _folder = data_file + '/' + checkpoint_folder
    _file = 'all.json'
    checkpoint = _folder + '/' + 'e' + str(num_epochs) + '.ckpt'

    num_auxiliary = len(auxiliary)
    num_user_attention, num_item_attention, prime_input, user_id, item_id, charac2int, int2charac= generate_partial_param(user_id, item_id)


    print(prime_input)

    with open(os.path.join(_folder, _file)) as _input:
        data = json.load(_input)
        print ('number of auxiliary,', num_auxiliary)

        num_characters = data[0]['number of characters']
        print ('number of classes,', num_characters)

        num_batches = data[0]['number of batchs']
        print ('number of batches,', num_batches)

        batch_size = data[0]['batch_size']
        print ('batch size,', batch_size)

        num_char = data[0]['number of characters per train']
        print ('number of characters per train,', num_char)

        lstm_size = data[0]['lstm_size']
        print ('number of lstm size,', lstm_size)
        num_layers = data[0]['num_layers']
        learning_rate = data[0]['learning rate']
        keep_prob = data[0]['keep probability']
        epochs = data[0]['epochs']

        print ('number of whole characters of this user,', num_char * num_batches)

    generate_word_index, probabiliry = get_sample(checkpoint, n_samples, lstm_size,
                                                  num_characters, num_user_attention, num_item_attention,
                                                  auxiliary, user_id, item_id, prime_input)

    return generate_text_by_words_id(generate_word_index, int2charac)





