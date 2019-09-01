import pandas as pd
import numpy as np


class readDataStore():
    """
    This module is to read the data from data store.

    In our case, our date store is local and is in .csv format
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_stores(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_items(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_transactions(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_oil(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_holidays(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_train(self, file_name):
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_test(self, file_name):
        df = pd.read_csv(self.data_dir + file_name, usecols=[0, 1, 2, 3, 4],
                         dtype={'onpromotion': bool},
                         parse_dates=["date"]  # , date_parser=parser
                         ).set_index(
            ['store_nbr', 'item_nbr', 'date']
        )

        return df

    def read_train(self, file_name):
        df = pd.read_csv(self.data_dir + file_name,
                         usecols=[1, 2, 3, 4, 5],
                         dtype={'onpromotion': bool},
                         converters={'unit_sales': lambda u: np.log1p(
                             float(u)) if float(u) > 0 else 0},
                         parse_dates=["date"],
                         skiprows=range(1, 66458909)  # 2016-01-01
                         )

        return df
