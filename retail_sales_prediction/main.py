"""Main module."""

"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from os.path import basename
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retail_sales_prediction.utils.data_loader import readDataStore
from retail_sales_prediction.utils.data_preparation import FeaturePreparation
from retail_sales_prediction.logging_configurator import LoggingConfigurator
from retail_sales_prediction.utils.run_model import run_model_lgbm

from retail_sales_prediction import logger

if __name__ == '__main__':
    data_dir = '/media/farmshare2/Research/raza/p_data/'

    LoggingConfigurator.init(data_dir + 'logs.log')
    logger.info('Logger started')

    logger.info('Loading the Read Data Module')

    rd = readDataStore(data_dir)
    # Reading the test data
    df_test = rd.read_test('test.csv')
    logger.info('Test Data Loaded')

    # Read the items data frame
    df_items = rd.read_items('items.csv').set_index("item_nbr")
    logger.info('Items Data Loaded')

    # Read the stores data frame
    df_stores = rd.read_items('stores.csv').set_index("store_nbr")
    logger.info('Stores Data Loaded')

    # Reading the training data
    df_train = rd.read_train('train.csv')
    logger.info('Training Data Loaded')

    feature_prep = FeaturePreparation(df_train, df_test, df_items, df_stores)

    (df_2017, promo_2017,
     df_2017_item, promo_2017_item,
     df_2017_store_class, df_2017_store_class_index,
     df_2017_promo_store_class, df_2017_promo_store_class_index) = feature_prep.pre_process_data()

    X_train, y_train = feature_prep. \
        get_training_data(df_2017, promo_2017,
                          df_2017_item, promo_2017_item,
                          df_2017_store_class, df_2017_store_class_index,
                          df_2017_promo_store_class, df_2017_promo_store_class_index,
                          anchor_date=date(2017, 6, 14), num_days=6)

    X_val, y_val = feature_prep. \
        get_validation_data(df_2017, promo_2017,
                            df_2017_item, promo_2017_item,
                            df_2017_store_class, df_2017_store_class_index,
                            df_2017_promo_store_class, df_2017_promo_store_class_index,
                            val_start_date=date(2017, 7, 26))

    X_test = feature_prep. \
        get_test_data(df_2017, promo_2017,
                      df_2017_item, promo_2017_item,
                      df_2017_store_class, df_2017_store_class_index,
                      df_2017_promo_store_class, df_2017_promo_store_class_index,
                      test_start_date=date(2017, 8, 16))

    run_model_lgbm(feature_prep, X_train, y_train,
                   X_val, y_val, X_test, num_days=6)
