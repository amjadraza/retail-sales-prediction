"""
.. module:: data_loader
   :synopsis: Data Reader
.. moduleauthor:: MA Raza

This module is to read the data

Todo:
    * Add more readers
    * Add features to read from Kaggle using API
"""

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
        """
        reads the data table with stores information
        Args:
            file_name: Name of the file for stores information

        Returns: dataframe

        """

        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_items(self, file_name):
        """
        reads the data table with items information
        Args:
            file_name: name of the items data table

        Returns: dataframe

        """

        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_transactions(self, file_name):
        """
        reade the transactions data table
        Args:
            file_name: name of transaction data table file

        Returns: dataframe

        """
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_oil(self, file_name):
        """
        reading the oil prices time series
        Args:
            file_name: oil prices file name

        Returns: dataframe

        """
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_holidays(self, file_name):
        """
        Holidays data table reader
        Args:
            file_name:  holidays data table name

        Returns: dataframe

        """
        df = pd.read_csv(self.data_dir + file_name)

        return df

    def read_test(self, file_name):
        """
        Test data reader
        Args:
            file_name: name of the test data table

        Returns: dataframe with index on ['store_nbr', 'item_nbr', 'date']

        """
        df = pd.read_csv(self.data_dir + file_name, usecols=[0, 1, 2, 3, 4],
                         dtype={'onpromotion': bool},
                         parse_dates=["date"]  # , date_parser=parser
                         ).set_index(
            ['store_nbr', 'item_nbr', 'date']
        )

        return df

    def read_train(self, file_name):
        """
        Training data reader. Original training data starts from 2013.
        Args:
            file_name: Name of training data file name

        Returns: dataframe, unit_sales are converted into log

        """
        df = pd.read_csv(self.data_dir + file_name,
                         usecols=[1, 2, 3, 4, 5],
                         dtype={'onpromotion': bool},
                         converters={'unit_sales': lambda u: np.log1p(
                             float(u)) if float(u) > 0 else 0},
                         parse_dates=["date"],
                         skiprows=range(1, 66458909)  # 2016-01-01
                         )

        return df
