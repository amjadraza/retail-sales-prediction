"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class FeaturePreparation():

    def __init__(self, df_train, df_test, df_items, df_stores):

        self.items = df_items
        self.stores = df_stores
        self.train = df_train
        self.test = df_test
        self._labeltovalues()
        logger.info('Pre Processing the data for training and promotions')


        # (self.df_2017,
        #  self.promo_2017,
        #  self.df_2017_item,
        #  self.promo_2017_item,
        #  self.df_2017_store_class_index,
        #  self.df_2017_store_class,
        #  self.df_2017_promo_store_class_index,
        #  self.df_2017_promo_store_class) = self._pre_process_data()

    def _labeltovalues(self):

        # Assigning the values to the labels
        logger.info('Performing the label to values conversion')
        le = LabelEncoder()
        self.items['family'] = le.fit_transform(self.items['family'].values)

        self.stores['city'] = le.fit_transform(self.stores['city'].values)
        self.stores['state'] = le.fit_transform(self.stores['state'].values)
        self.stores['type'] = le.fit_transform(self.stores['type'].values)
        return self

    def pre_process_data(self):

        # print('Converting Label to values')
        # self._labeltovalues()
        # Extracting the training data for 2017 until 2017-08-14
        df_2017 = self.train.loc[self.train.date >= pd.datetime(2017, 1, 1)]
        # Delete the big training data
        # del df_train
        # On each store what items are on promotion on each day
        promo_2017_train = df_2017.set_index(
            ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
        logger.info('promo_2017_train')
        # Training data fro 2017 consists of 23 billions records ans it
        # covers only 8.5 months. This problem becomes complex and
        # computational expensive if we have to use more data and more derived features.

        # promotion columns from training data
        promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

        # promotion data from test data
        promo_2017_test = self.test[["onpromotion"]].unstack(level=-1).fillna(False)

        # Test data Starts from 2017-08-16 to 2017-08-31

        promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)

        promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

        # Combined the promotion data from training and test data
        logger.info('Merging the promotion data for whole 2017 year ')
        promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
        # Working on Target Variable:
        # In this problem, our target variable is unit_sales.

        df_2017_tmp = df_2017.set_index(
            ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1)

        # ---------------------------------------------------------------------------------------------------------
        # There are some missing values for unit sales as can be seen from above plot.
        # In the solution, they filled with 0, however I am not fully convinced.
        # There could be other method to handle these missing values.
        # One of the argument is that missing values are due to issues with data collection.

        # This issue can be further studied.

        # One potential candidate to be used for forward filling can be average over
        # last 20 days or so.

        logger.info('Working on Unit_Sales data')
        df_2017 = df_2017.set_index(
            ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)

        df_2017.columns = df_2017.columns.get_level_values(1)

        self.items = self.items.reindex(df_2017.index.get_level_values(1))
        self.stores = self.stores.reindex(df_2017.index.get_level_values(0))

        df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()
        promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()

        # df_2017_item.sum(axis=0).plot()
        # promo_2017_item.sum(axis=0).plot()

        df_2017_store_class = df_2017.reset_index()
        df_2017_store_class['class'] = self.items['class'].values
        df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
        df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()

        # df_2017_store_class.sum(axis=1).plot()

        df_2017_promo_store_class = promo_2017.reset_index()
        df_2017_promo_store_class['class'] = self.items['class'].values
        df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]
        df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()
        logger.info('Data Preparation step Completed')
        return (df_2017, promo_2017,
                df_2017_item, promo_2017_item,
                df_2017_store_class, df_2017_store_class_index,
                df_2017_promo_store_class, df_2017_promo_store_class_index)

    def get_timespan(self, df, dt, minus, periods, freq='D'):
        return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

    def prepare_dataset(self, df, promo_df, t2017, is_train=True, name_prefix=None):
        X = {
            "promo_14_2017": self.get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,
            "promo_60_2017": self.get_timespan(df=promo_df, dt=t2017, minus=60, periods=60).sum(axis=1).values,
            "promo_140_2017": self.get_timespan(df=promo_df, dt=t2017, minus=140, periods=140).sum(axis=1).values,
            "promo_3_2017_aft": self.get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,
            "promo_7_2017_aft": self.get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
            "promo_14_2017_aft": self.get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
        }

        for i in [3, 7, 14, 30, 60, 140]:
            tmp1 = self.get_timespan(df, t2017, i, i)
            tmp2 = (self.get_timespan(promo_df, t2017, i, i) > 0) * 1

            X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
            X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values

            X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
            X['no_promo_mean_%s_decay' % i] = (
                    tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        for i in [3, 7, 14, 30, 60, 140]:
            tmp = self.get_timespan(df, t2017, i, i)
            X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s' % i] = tmp.mean(axis=1).values
            X['median_%s' % i] = tmp.median(axis=1).values
            X['min_%s' % i] = tmp.min(axis=1).values
            X['max_%s' % i] = tmp.max(axis=1).values
            X['std_%s' % i] = tmp.std(axis=1).values

        for i in [3, 7, 14, 30, 60, 140]:
            tmp = self.get_timespan(df, t2017 + timedelta(days=-7), i, i)
            X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s_2' % i] = tmp.mean(axis=1).values
            X['median_%s_2' % i] = tmp.median(axis=1).values
            X['min_%s_2' % i] = tmp.min(axis=1).values
            X['max_%s_2' % i] = tmp.max(axis=1).values
            X['std_%s_2' % i] = tmp.std(axis=1).values

        for i in [7, 14, 30, 60, 140]:
            tmp = self.get_timespan(df, t2017, i, i)
            X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
            X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
            X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

            tmp = self.get_timespan(promo_df, t2017, i, i)
            X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
            X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
            X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        tmp = self.get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)
        X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values
        X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values
        X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

        for i in range(1, 16):
            X['day_%s_2017' % i] = self.get_timespan(df, t2017, i, 1).values.ravel()

        for i in range(7):
            X['mean_4_dow{}_2017'.format(i)] = self.get_timespan(df, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
            X['mean_20_dow{}_2017'.format(i)] = self.get_timespan(df, t2017, 140 - i, 20, freq='7D').mean(axis=1).values

        for i in range(-16, 16):
            X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

        X = pd.DataFrame(X)

        if is_train:
            y = df[
                pd.date_range(t2017, periods=16)
            ].values
            return X, y
        if name_prefix is not None:
            X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
        return X

    def get_training_data(self, df_2017, promo_2017,
                         df_2017_item, promo_2017_item,
                         df_2017_store_class, df_2017_store_class_index,
                         df_2017_promo_store_class, df_2017_promo_store_class_index,
                          anchor_date=date(2017, 6, 14), num_days =6):
        print("Preparing dataset...")
        t2017 = anchor_date
        # num_days = 6
        X_l, y_l = [], []
        for i in range(num_days):
            delta = timedelta(days=7 * i)
            X_tmp, y_tmp = self.prepare_dataset(df_2017, promo_2017, t2017 + delta)

            X_tmp2 = self.prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta,
                                          is_train=False, name_prefix='item')
            X_tmp2.index = df_2017_item.index
            X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

            X_tmp3 = self.prepare_dataset(df_2017_store_class,
                                          df_2017_promo_store_class,
                                          t2017 + delta,
                                          is_train=False,
                                          name_prefix='store_class')
            X_tmp3.index = df_2017_store_class.index
            X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

            X_tmp = pd.concat([X_tmp,
                               X_tmp2,
                               X_tmp3,
                               self.items.reset_index(),
                               self.stores.reset_index()],
                              axis=1)
            X_l.append(X_tmp)
            y_l.append(y_tmp)

            del X_tmp2
            gc.collect()

        X_train = pd.concat(X_l, axis=0)
        y_train = np.concatenate(y_l, axis=0)

        return X_train, y_train

    def get_validation_data(self,df_2017, promo_2017,
                           df_2017_item, promo_2017_item,
                           df_2017_store_class, df_2017_store_class_index,
                           df_2017_promo_store_class, df_2017_promo_store_class_index,
                           val_start_date=date(2017, 7, 26)):

        # del X_l, y_l

        # Prepare Validation data set
        # val_start_date = date(2017, 7, 26)
        X_val, y_val = self.prepare_dataset(df_2017, promo_2017, val_start_date)

        X_val2 = self.prepare_dataset(df_2017_item, promo_2017_item,
                                      val_start_date, is_train=False, name_prefix='item')
        X_val2.index = df_2017_item.index
        X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

        X_val3 = self.prepare_dataset(df_2017_store_class, df_2017_promo_store_class,
                                      val_start_date, is_train=False,
                                      name_prefix='store_class')
        X_val3.index = df_2017_store_class.index
        X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

        X_val = pd.concat([X_val, X_val2, X_val3, self.items.reset_index(), self.stores.reset_index()], axis=1)

        return X_val, y_val

    def get_test_data(self, df_2017, promo_2017,
                      df_2017_item, promo_2017_item,
                      df_2017_store_class, df_2017_store_class_index,
                      df_2017_promo_store_class, df_2017_promo_store_class_index,
                      test_start_date=date(2017, 7, 26)):
        # Prepare Test data set
        # test_start_date = date(2017, 8, 16)
        X_test = self.prepare_dataset(df_2017, promo_2017, test_start_date, is_train=False)

        X_test2 = self.prepare_dataset(df_2017_item,
                                       promo_2017_item,
                                       test_start_date, is_train=False, name_prefix='item')
        X_test2.index = df_2017_item.index
        X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

        X_test3 = self.prepare_dataset(df_2017_store_class,
                                       df_2017_promo_store_class,
                                       test_start_date, is_train=False,
                                        name_prefix='store_class')
        X_test3.index = df_2017_store_class.index
        X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)

        X_test = pd.concat([X_test, X_test2, X_test3, self.items.reset_index(), self.stores.reset_index()], axis=1)

        return X_test
# del X_test2, X_val2, df_2017_item, promo_2017_item, df_2017_store_class, df_2017_promo_store_class, df_2017_store_class_index
# gc.collect()
