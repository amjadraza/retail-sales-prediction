"""
.. module:: main
   :synopsis: training of the model
.. moduleauthor:: MA Raza

This is the main file to run Forecasting pipe line. This pipline is based on upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.

Example

    >>> python retail_sales_prediction/main.py

Todo:
    * Add configuration file to control the parameters --Done
    * Covert the pipeline into spark version
"""

from datetime import date, timedelta
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retail_sales_prediction.utils.data_loader import readDataStore
from retail_sales_prediction.utils.data_preparation import FeaturePreparation
from retail_sales_prediction.utils.run_model import run_model_lgbm
from loguru import logger

import argparse

parser = argparse.ArgumentParser(description='Retail_sales_prediction')
parser.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
args = parser.parse_args()

if args.config_file:
        config = yaml.load(args.config_file)
        delattr(args, 'config_file')
logger.info(config)

if __name__ == '__main__':
    # data_dir = "Location of Data"

    data_dir = config['data_dir']

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
    logger.info('Start Data Preparation')
    (df_2017, promo_2017,
     df_2017_item, promo_2017_item,
     df_2017_store_class, df_2017_store_class_index,
     df_2017_promo_store_class, df_2017_promo_store_class_index) = feature_prep.pre_process_data()

    logger.info('Started Preparing Training Data')
    X_train, y_train = feature_prep. \
        get_training_data(df_2017, promo_2017,
                          df_2017_item, promo_2017_item,
                          df_2017_store_class, df_2017_store_class_index,
                          df_2017_promo_store_class, df_2017_promo_store_class_index,
                          anchor_date=date(config['anchor_date']), num_days=6)
    logger.info('Started Preparing Validation Data')
    X_val, y_val = feature_prep. \
        get_validation_data(df_2017, promo_2017,
                            df_2017_item, promo_2017_item,
                            df_2017_store_class, df_2017_store_class_index,
                            df_2017_promo_store_class, df_2017_promo_store_class_index,
                            val_start_date=config['val_start_date'])

    logger.info('Started Preparing Test Data')
    X_test = feature_prep. \
        get_test_data(df_2017, promo_2017,
                      df_2017_item, promo_2017_item,
                      df_2017_store_class, df_2017_store_class_index,
                      df_2017_promo_store_class, df_2017_promo_store_class_index,
                      test_start_date=config['test_start_date'])

    logger.info('Started Training the Light GBM Model')
    run_model_lgbm(feature_prep, X_train, y_train,
                   X_val, y_val, X_test,
                   config, num_days=6)
