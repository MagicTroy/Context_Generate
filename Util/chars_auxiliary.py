
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 2017

@author: troy

MODIFIED: no split data, no validation data

"""
import sys

import numpy as np
from .load_data import LoadData
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from .log import get_logger
log = get_logger()

class CharsAuxi:

    def __init__(self, df):
        review = self._get_element(df, 'r_text')
        review = self.__insert_prime(review)

        self.char_to_int, self.int_to_char = self._get_chars_dict(review)
        self.chars_review, self.auxi_one_hot = self._get_chars_auxi_one_hot(df, review)

    def _get_num_chars_per_user(self, df):
        """
        calculating number of characters of each users
        (need about 40 minutes)
        :param df:
        :return:
        """
        user = df['user_id']

        user_char_count = {}

        def _get_chars_len(m):
            return len(''.join(m))

        for i in set(user):
            _df = df[df['user_id'].isin([i])]['r_text'].tolist()
            user_char_count[i] = _get_chars_len(_df)

        return user_char_count

    def __insert_prime(self, review, prime=u'<#str#>', end=u'<#end#>'):
        _review = []
        for i, text in enumerate(review):
            if isinstance(text, float):
                print(i, text)

            _review.append(prime + text + end)
        return _review

    def _get_element(self, df,  element):
        return df[element].tolist()

    def _get_chars_dict(self, corpus):
        """
        get characters to indices
        :param corpus:
        :return:
        """

        # join the text to a corpus
        text = ''.join(corpus)

        # generate character dictionary
        cv = CountVectorizer(analyzer='char_wb')
        cv.fit_transform([text])

        charac_to_int = cv.vocabulary_
        int_to_charac = dict(map(reversed, charac_to_int.items()))

        return charac_to_int, int_to_charac

    def _get_auxi_dict(self, auxiliary):
        """
        get auxiliary dictionary (not used)
        :param auxiliary:
        :return:
        """

        _dict = {}
        for idx, auxi in enumerate(sorted(list(set(auxiliary)))):
            _dict[auxi] = np.eye(len(set(auxiliary)))[idx]
        return _dict

    def get_char_int_auxi(self, c, auxiliary):
        """
        get indices of a char and a lis tof auxiliary value list
        :param c:
        :param appea:
        :param aroma:
        :param palat:
        :param taste:
        :param overa:
        :return:
        """
        return self.char_to_int[c], np.array(auxiliary)

    def _get_chars_auxi_one_hot(self, df, review):
        """
        get list of indices of a reviews
        get auxiliary one hot matrix
        :param df:
        :param review:
        :return:
        """

        chars = []
        auxi_one_hot_matrix = []

        appear = self._get_element(df, 'r_appearance')
        aroma = self._get_element(df, 'r_aroma')
        palate = self._get_element(df, 'r_palate')
        taste = self._get_element(df, 'r_taste')
        rating = self._get_element(df, 'r_overall')

        # user_id = self._get_element(df, 'user_id')
        # item_id = self._get_element(df, 'beer_id')
        #
        # self.num_auxiliary = len([appear, aroma, palate, taste, rating, user_id, item_id])

        # self.num_user = len(set(user_id))
        # self.num_item = len(set(item_id))
        # self.num_auxiliary = 5 + self.num_user + self.num_item

        self.num_auxiliary = len([appear, aroma, palate, taste, rating])

        for i, re in enumerate(review):
            app, aro, pal, tas, ove = appear[i], aroma[i], palate[i], taste[i], rating[i]
            # uid, iid = user_id[i], item_id[i]
            # app, aro, pal, tas, ove = 1,1,1,1,1

            # u_one_hot, i_one_hot = np.eye(len(set(user_id)))[uid], np.eye(len(set(item_id)))[iid]

            # ui_one_hot = np.append(u_one_hot, i_one_hot)

            for c in re:
                # char_int, auxi_one_hot = self.get_char_int_auxi(c, [app, aro, pal, tas, ove, uid, iid])
                char_int, auxi_one_hot = self.get_char_int_auxi(c, [app, aro, pal, tas, ove])

                # auxi_one_hot = np.append(auxi_one_hot, ui_one_hot)

                chars.append(char_int)
                auxi_one_hot_matrix.append(auxi_one_hot)

        return np.array(chars), np.array(auxi_one_hot_matrix)

    def _get_split_chars(self, chars, batch_size, num_char, split_frac=0.9):
        """
        split chars matrix
        :param chars:
        :param batch_size:
        :param num_char:
        :param split_frac:
        :return:
        """

        slice_size = batch_size * num_char
        num_batches = len(chars) // slice_size

        # get full characters and their following characters
        x = chars[: num_batches * slice_size]
        y = chars[1: num_batches * slice_size + 1]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        # split_idx = int(num_batches * split_frac)
        # train_x, train_y = x[:, :split_idx * num_char], y[:, :split_idx * num_char]
        # valid_x, valid_y = x[:, split_idx * num_char:], y[:, split_idx * num_char:]
        #
        # return train_x, train_y, valid_x, valid_y

        return x, y

    def _get_split_auxi(self, auxi, num_batches, batch_size, num_char, split_frac=1.):
        """
        split auxiliary matrix
        :param auxi:
        :param num_batches:
        :param batch_size:
        :param num_char:
        :param split_frac:
        :return:
        """
        slice_size = batch_size * num_char

        # get full characters and their following characters
        x = auxi[: num_batches * slice_size, :]
        y = auxi[1: num_batches * slice_size + 1, :]

        # split full characters by batch size
        x = np.stack(np.split(x, batch_size))
        y = np.stack(np.split(y, batch_size))

        # split_idx = int(num_batches * split_frac)
        # train_x, train_y = x[:, :split_idx * num_char, :], y[:, :split_idx * num_char, :]
        # valid_x, valid_y = x[:, split_idx * num_char:, :], y[:, split_idx * num_char:, :]
        #
        # return train_x, train_y, valid_x, valid_y

        return x, y

    def get_concate_data(self, num_characters, num_batches, batch_size, num_char):
        """
        get characters one hot
        concatenate characters ont hot with auxiliary one hot
        :param num_characters:
        :param num_batches:
        :param batch_size:
        :param num_char:
        :return:
        """

        # split reviews
        x, y = self._get_split_chars(self.chars_review, batch_size, num_char)

        # print(train_x.shape, train_y.shape)
        # print(val_x.shape, val_y.shape)
        log.info('shape: {}', x.shape, y.shape)

        # split auxiliary one hot
        auxi_x_one_hot, auxi_y_one_hot = self._get_split_auxi(self.auxi_one_hot, num_batches, batch_size, num_char)

        # generate chars one hot
        def _get_one_hot(mm):
            return np.eye(num_characters)[mm]
        # train_x_one_hot = np.array(map(_get_one_hot, train_x))
        # train_y_one_hot = np.array(map(_get_one_hot, train_y))
        # val_x_one_hot = np.array(map(_get_one_hot, val_x))
        # val_y_one_hot = np.array(map(_get_one_hot, val_x))

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromiter.html
        #train_x_one_hot = np.array([_get_one_hot(v) for v in x])
        #train_y_one_hot = np.array([_get_one_hot(v) for v in y])

        ee = np.eye(num_characters)
        train_x_one_hot = np.array([ee[v].copy() for v in x])
        print (sys.getsizeof(train_x_one_hot))
        train_y_one_hot = np.array([ee[v].copy() for v in y])
        print (sys.getsizeof(train_y_one_hot))

        # concate
        # train_x_one_hot = np.concatenate((train_x_one_hot, auxi_train_x_one_hot), axis=2)
        # val_x_one_hot = np.concatenate((val_x_one_hot, auxi_val_x_one_hot), axis=2)

        train_x_one_hot = np.concatenate((train_x_one_hot, auxi_x_one_hot), axis=2)
        # train_y_one_hot = np.concatenate((train_y_one_hot, auxi_y_one_hot), axis=2)

        return train_x_one_hot, train_y_one_hot

    def get_batch(self, train_xy, num_char):
        """
        get batch of inout data
        :param train_xy:
        :param num_char:
        :return:
        """
        batch_size, slice_size, _ = train_xy[0].shape

        num_batches = slice_size // num_char
        for batch_idx in range(num_batches):
            yield [xy[:, batch_idx * num_char: (batch_idx + 1) * num_char] for xy in train_xy]


    # version 0.2: first get batched data, then concatenated each batch before trianing
    
    def get_xy_auxiliary(self, num_batches, batch_size, num_char):
        # split reviews
        x, y = self._get_split_chars(self.chars_review, batch_size, num_char)

        log.info('shape: {}', x.shape, y.shape)

        # split auxiliary one hot
        auxi_x_one_hot, _ = self._get_split_auxi(self.auxi_one_hot, num_batches, batch_size, num_char)
        
        return x, y, auxi_x_one_hot


    # use this generator to get each batch of dataset
    def get_concat_batch(self, x, y, auxi_x, num_batches, num_char, num_characters):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * num_char: (batch_idx + 1) * num_char]
            yy = y[:, batch_idx * num_char: (batch_idx + 1) * num_char]
            a_x = auxi_x[:, batch_idx * num_char: (batch_idx + 1) * num_char]
            
            ee = np.eye(num_characters)
            train_x_one_hot = np.array([ee[v].copy() for v in xx])
            train_y_one_hot = np.array([ee[v].copy() for v in yy])
            
            train_x_one_hot = np.concatenate((train_x_one_hot, a_x), axis=2)
            
            yield train_x_one_hot, train_y_one_hot

    # version 0.3: compute chars and auxiliary one hot in each batch
    def get_concat_batch_one_hot(self, x, y, auxi_x, num_batches, num_char, num_characters):
        for batch_idx in range(num_batches):
            xx = x[:, batch_idx * num_char: (batch_idx + 1) * num_char]
            yy = y[:, batch_idx * num_char: (batch_idx + 1) * num_char]
            a_x = auxi_x[:, batch_idx * num_char: (batch_idx + 1) * num_char]

            ee = np.eye(num_characters)
            train_x_one_hot = np.array([ee[v].copy() for v in xx])
            train_y_one_hot = np.array([ee[v].copy() for v in yy])

            a_x_one_hot = []
            for chars_auxi in a_x:

                _chars_auxi = []

                for auxi in chars_auxi:

                    uid, iid = auxi[-2], auxi[-1]
                    u_one_hot, i_one_hot = np.eye(self.num_user)[int(uid)], np.eye(self.num_item)[int(iid)]

                    ui_one_hot = np.append(u_one_hot, i_one_hot)

                    auxi_one_hot = np.append([a_x[:-2]], ui_one_hot)

                    _chars_auxi.append(auxi_one_hot)

                a_x_one_hot.append(np.array(_chars_auxi))

            a_x_one_hot = np.array(a_x_one_hot)

            print(a_x_one_hot.shape)

            train_x_one_hot = np.concatenate((train_x_one_hot, a_x_one_hot), axis=2)

            yield train_x_one_hot, train_y_one_hot

            
"""
example

example user:
[u'kegatron', u'ppoitras', u'jwc215', u'beerchitect', u'phyl21ca']

"""
# if __name__ == "__main__":
#     load = LoadData()
#     training_df = load.get_df_by_userids(u'kegatron')
#     chars_auxi = CharsAuxi(training_df)
#     train_x_one_hot, train_y_one_hot, val_x_one_hot, val_y_one_hot = chars_auxi.get_concate_data(58, 10, 10, 26303)
#     print train_x_one_hot.shape, train_y_one_hot.shape
#     print val_x_one_hot.shape, val_y_one_hot.shape











