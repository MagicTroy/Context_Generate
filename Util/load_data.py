

from sqlalchemy import create_engine
import pandas as pd
import pickle as pkl
import os
from .log import get_logger
log = get_logger()

class LoadData:

    def __init__(self, path):
        pkl_file = path + ".pkl"
        if os.path.exists(pkl_file):
            log.info('loading from pickle {}', pkl_file)
            self.df_total = pkl.load(open(pkl_file, 'rb'))
        else:
            _file = 'sqlite:///' + path
            log.info('reading from: {}', _file)
            engine = create_engine(_file)

            log.info("table name: {}", engine.table_names())

            review = engine.execute('SELECT * FROM REVIEWS limit 1')
            log.info("keys: {}", type(review.keys()))

            df_review = pd.read_sql_table("reviews", engine)
            df_beer = pd.read_sql_table('beers', engine)
            self.df_total = pd.merge(df_review, df_beer, on='beer_id', how='outer')
            self.df_total = self._norm_auxi_value(self.df_total)
            log.info('pickle dumping to : {}', pkl_file)
            pkl.dump(self.df_total, open(pkl_file, 'wb'))
        log.info('done loading data')
        return

        # _file = 'sqlite:///' + path
        # log.info('reading from: {}', _file)
        # engine = create_engine(_file)
        #
        # log.info("table name: {}", engine.table_names())
        #
        # review = engine.execute('SELECT * FROM REVIEWS limit 1')
        # log.info("keys: {}", type(review.keys()))
        #
        # df_review = pd.read_sql_table("reviews", engine)
        # df_beer = pd.read_sql_table('beers', engine)
        # self.df_total = pd.merge(df_review, df_beer, on='beer_id', how='outer')
        # self.df_total = self._norm_auxi_value(self.df_total)
        # log.info('done loading data')


    def _norm_auxi_value(self, df):
        for col in ['r_appearance', 'r_aroma', 'r_overall', 'r_palate', 'r_taste']:
            df[col] = df[col] / 5.0

        return df

    def get_df_by_userids(self, userids):
        return self.df_total[self.df_total['user_id'].isin(userids)]












