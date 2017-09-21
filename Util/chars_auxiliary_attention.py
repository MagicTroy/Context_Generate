import sys

import numpy as np
from .log import get_logger

log = get_logger()


class CharsAuxi:
    def __init__(self, input, auxiliary, user_attention, item_attention):
        # shape: (number of words, words vector)
        self.input = input
        # shape: (number of words, number of auxiliary)
        self.auxiliary = auxiliary
        # shape: (number of words, number of attention)
        self.user_attention = user_attention

        self.item_attention = item_attention

    # # one hot type
    # def generate_one_hot(self, m):
    #     ee = np.eye(self.words_length)
    #     return np.array([ee[v].copy() for v in m])

    # word vector type

    def concat_data(self, data_1, data_2, axis=2):
        return np.concatenate((data_1, data_2), axis=axis)

    def split_index(self, data, num_batches, batch_size, number_words_per_batch, split_frac=.9):
        slice_size = batch_size * number_words_per_batch

        x = data[: num_batches * slice_size]
        y = data[1: num_batches * slice_size + 1]

        # split full characters by batch size                                                                        \

        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        return x, y

    def split_1d_date(self, data, num_batches, batch_size, number_words_per_batch):
        slice_size = batch_size * number_words_per_batch

        x = data[: num_batches * slice_size]
        y = data[1: num_batches * slice_size + 1]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        return x, y

    def split_2d_date(self, data, num_batches, batch_size, number_words_per_batch):
        slice_size = batch_size * number_words_per_batch

        x = data[: num_batches * slice_size, :]
        y = data[1: num_batches * slice_size + 1, :]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        return x, y

    def split_1d_date_by_frac(self, data, num_batches, batch_size, number_words_per_batch, split_frac=.9):
        slice_size = batch_size * number_words_per_batch

        x = data[: num_batches * slice_size]
        y = data[1: num_batches * slice_size + 1]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        split_idx = int(num_batches * split_frac)
        train_x, train_y = x[:, :split_idx * number_words_per_batch], y[:, :split_idx * number_words_per_batch]
        valid_x, valid_y = x[:, split_idx * number_words_per_batch:], y[:, split_idx * number_words_per_batch:]

        return train_x, train_y, valid_x, valid_y

    def split_2d_date_by_frac(self, data, num_batches, batch_size, number_words_per_batch, split_frac=.9):
        slice_size = batch_size * number_words_per_batch

        x = data[: num_batches * slice_size, :]
        y = data[1: num_batches * slice_size + 1, :]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        split_idx = int(num_batches * split_frac)
        train_x, train_y = x[:, :split_idx * number_words_per_batch], y[:, :split_idx * number_words_per_batch]
        valid_x, valid_y = x[:, split_idx * number_words_per_batch:], y[:, split_idx * number_words_per_batch:]

        return train_x, train_y, valid_x, valid_y

    def get_batch(self, x, y, attention, num_batches, number_words_per_batch):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            yy = y[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            att_x = attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]

            print(xx.shape, yy.shape)

            yield xx, yy, att_x

    def get_auxi_batch(self, x, y, auxiliary, attention, num_batches, number_words_per_batch):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            yy = y[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            a_x = auxiliary[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            att_x = attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]

            print(xx.shape, yy.shape)

            yield xx, yy, a_x, att_x

    def get_auxi_attention_batch(self, x, y, auxiliary, user_attention, item_attention, num_batches, number_words_per_batch):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            yy = y[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            a_x = auxiliary[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            user_att_x = user_attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            item_att_x = item_attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]

            print(xx.shape, yy.shape)

            yield xx, yy, a_x, user_att_x, item_att_x

    def get_auxi_attention_batch_test(self, test_x, test_y, test_auxiliary, test_user_attention, test_item_attention, number_words_per_batch):

        length = test_x.shape[1]
        if length == number_words_per_batch:
            return test_x, test_y, test_auxiliary, test_user_attention, test_item_attention
        else:
            idx = np.random.randint(low=0, high=length - number_words_per_batch)

            xx = test_x[:, idx: (idx + 1) * number_words_per_batch]
            yy = test_y[:, idx: (idx + 1) * number_words_per_batch]
            a_x = test_auxiliary[:, idx: (idx + 1) * number_words_per_batch]
            user_att_x = test_user_attention[:, idx: (idx + 1) * number_words_per_batch]
            item_att_x = test_item_attention[:, idx: (idx + 1) * number_words_per_batch]

            return xx, yy, a_x, user_att_x, item_att_x

    # def get_batch_one_hot(self, x, y, attention, num_batches, number_words_per_batch):
    #     for batch_idx in range(num_batches):
    #         xx = x[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
    #         yy = y[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
    #         att_x = attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
    #
    #         print(xx.shape, yy.shape)
    #
    #         _xx, _yy = [], []
    #         for x, y in zip(xx, yy):
    #             _xx.append(self.generate_one_hot(x))
    #             _yy.append(self.generate_one_hot(x))
    #
    #         _xx = np.array(_xx)
    #         _yy = np.array(_yy)
    #
    #         print(_xx.shape, _yy.shape)
    #
    #         yield _xx, _yy, att_x

    def get_concat_batch(self, x, y, auxiliary, attention, num_batches, number_words_per_batch):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            yy = y[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            a_x = auxiliary[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]
            att_x = attention[:, batch_idx * number_words_per_batch: (batch_idx + 1) * number_words_per_batch]


            xx = self.concat_data(xx, a_x, axis=2)

            yield xx, yy, att_x





