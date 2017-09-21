import os
import pandas as pd
import numpy as np
from .log import get_logger

log = get_logger()


class LoadData:
    def __init__(self, path, data_size=0):

        if data_size == 0:
            char_vector_file = path + "chars.npy"
            auxiliary_file = path + "auxiliary.npy"
            user_attention_file = path + "user_attention.npy"
            item_attention_file = path + "item_attention.npy"

        else:
            char_vector_file = path + "chars_" + data_size + "k.npy"
            auxiliary_file = path + "auxiliary_" + data_size + "k.npy"
            user_attention_file = path + "user_attention_" + data_size + "k.npy"
            item_attention_file = path + "item_attention_" + data_size + "k.npy"

        self.input = np.load(char_vector_file)
        self.auxiliary = np.transpose(np.load(auxiliary_file))
        self.user_attention = np.load(user_attention_file)
        self.item_attention = np.load(item_attention_file)

        # log.info('done loading data')
        return


class LoadCSV:
    def __init__(self, path, user_name):
        path = os.path.join(path, user_name)

        df = pd.DataFrame.from_csv(path=os.path.join(path, user_name) + '_train_hold_more_out.csv')
        self.train_input = df.as_matrix(columns=['chars']).flatten()
        self.train_auxiliary = df.as_matrix(columns=['appe', 'arom', 'pala', 'tast', 'rate']).transpose()
        self.train_item_attention = df.as_matrix(columns=['item']).flatten()

        df = pd.DataFrame.from_csv(path=os.path.join(path, user_name) + '_test_hold_more_out.csv')
        self.test_input = df.as_matrix(columns=['chars']).flatten()
        self.test_auxiliary = df.as_matrix(columns=['appe', 'arom', 'pala', 'tast', 'rate']).transpose()
        self.test_item_attention = df.as_matrix(columns=['item']).flatten()

        # log.info('done loading data')
        return














