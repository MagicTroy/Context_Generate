
import numpy as np
import pickle
import pandas as pd
import sys
import time
import paramiko
import argparse
from string import printable

# from multiprocessing import Pool
# pool = Pool(processes=2)

printable = list(set(printable))

from Apply.sample_auxiliary_attention_per_user import *
from Util.load_input import *

#data_file = '/Users/troy/PhD_Research/Deep_Learning_with_Recommendation_system/GCN'


def generate_partial_param(iid, item_id_list):
    """
    :return:
    number of user attention
    number of item attention
    number of vector length of words
    prime input
    words set
    """
    iid = item_id_list.index(iid)
    num_item_attention = len(item_id_list)

    prime_input = generate_prime(printable, '_str_')

    return iid, prime_input, num_item_attention


def generate_prime(charac2int, prime):
    prime_input = []
    for p in prime:
        prime_input.append(charac2int.index(p))
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
    _file = str(user_id) + '.json'
    checkpoint = checkpoint_folder + '/' + 'e' + str(num_epochs) + '.ckpt'

    num_auxiliary = len(auxiliary)
    num_item_attention = 49005
    prime_input = generate_prime(printable, '_str_')

    with open(os.path.join(checkpoint_folder, _file)) as _input:
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
                                                  num_characters, num_item_attention,
                                                  auxiliary, item_id, prime_input)

    return generate_text_by_words_id(generate_word_index, printable)


def download(ip, name, password, localpath, remotepath, the_file):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # print(name, password)
    ssh.connect(ip, username=name, password=password)

    # ssh.connect('137.43.130.222', username="linyi", password="1Hblfly2sky")
    sftp = ssh.open_sftp()
    localpath = os.path.join(localpath, the_file)
    remotepath = os.path.join(remotepath, the_file)
    # print(localpath, remotepath)

    if os.path.exists(localpath):
        return
    else:
        sftp.get(remotepath, localpath)
        sftp.close()
        ssh.close()
    # print(localpath)


def generate(e, localpath, n_samples, item_id, auxiliary):
    sample = generate_sample(n_samples, localpath, e, train_user_id, item_id, auxiliary)
    return sample

    # with open(os.path.join(localpath, 'sample.json'), 'w') as output:
    #     json.dump(sample_results, output, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text')
    parser.add_argument('-u', '--train_user_id', default='northyorksammy')
    parser.add_argument('--localpath', default='')
    parser.add_argument('--remotepath', default='')
    parser.add_argument('--ip', default='137.43.130.222')
    parser.add_argument('--username', default="linyi")
    parser.add_argument('--password', default="1Hblfly2sky")
    parser.add_argument('--epochs_start', default=0, type=int)
    parser.add_argument('--epochs_end', default=10, type=int)

    args = parser.parse_args()

    file_path, train_user_id = '/Users/troy/PhD_Research/Deep_Learning_with_Recommendation_system/GCN/user_data', \
                               args.train_user_id
    load = LoadCSV(file_path, train_user_id)

    # create folder
    localpath, remotepath = args.localpath, args.remotepath
    localpath = os.path.join(localpath, train_user_id)
    remotepath = os.path.join(remotepath, train_user_id)

    ip, username, password = args.ip, args.username, args.password

    if not os.path.exists(localpath):
        os.makedirs(localpath)
    download(ip, username, password, localpath, remotepath, train_user_id+'.json')
    # download(localpath, remotepath, train_user_id+'.json')

    # pool.map(generate, range(args.epochs_start, args.epochs_end)[:2])
    # pool.close()
    # pool.join()

    n_samples = 500

    start, end = args.epochs_start, args.epochs_end

    s = time.time()

    # get test items and test auxiliary
    item_attention = list(load.test_item_attention)
    auxiliary = list(load.test_auxiliary.transpose())

    item_ids, auxiliary_list = [], []
    for idx, item in enumerate(item_attention):
        if idx == len(item_attention) - 1:
            item_ids.append(item)
            auxiliary_list.append(auxiliary[idx])
            break
        if item != item_attention[idx + 1]:
            item_ids.append(item)
            auxiliary_list.append(auxiliary[idx])

    # generate
    epochs_list, item_list, sample_list = [], [], []
    for e in range(start, end):

        data_file, index_file, meta_file = 'e' + str(e) + '.ckpt.data-00000-of-00001', 'e' + str(e) + '.ckpt.index', \
                                           'e' + str(e) + '.ckpt.meta'
        download(ip, username, password, localpath, remotepath, data_file)
        download(ip, username, password, localpath, remotepath, index_file)
        download(ip, username, password, localpath, remotepath, meta_file)

        item_dict = {}
        for item, auxi in zip(item_ids, auxiliary_list):
            sample_results = generate(e, localpath, n_samples, item, auxi)
            epochs_list.append(e)
            item_list.append(item)
            sample_list.append(sample_results)
        print('finished -------------------- ', e)

        # os.remove(os.path.join(localpath, data_file))
        # os.remove(os.path.join(localpath, index_file))
        # os.remove(os.path.join(localpath, meta_file))

    sample_df = pd.DataFrame(np.array([epochs_list, item_list, sample_list]).transpose(),
                             columns=['epochs', 'item', 'sample'])
    sample_df.to_csv(os.path.join(localpath, str(start)+'-'+str(end-1)+'_sample_results.csv'), encoding='utf-8')

    print('total time ', time.time() - s)











