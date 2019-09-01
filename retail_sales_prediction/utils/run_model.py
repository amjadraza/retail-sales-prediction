"""
.. module:: run_model
   :synopsis: Collection of Models
.. moduleauthor:: MA Raza

This modules consists of collection of various machine learning models. We start with Light GBM.

Depending on the time, we can add more

Todo:
    * Add more machine learning models, such as GBM, RF and XGBoost
    * Spark Compatible GBM and Light GBM Models
    * Add Model Diagnostic plots using SHAP Library
    * Feature Reduction
    * Config file
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from retail_sales_prediction import logger


def run_model_lgbm(feature_prep, X_train, y_train, X_val,y_val, X_test, num_days=6):
    """
    Training the Light GBM Model.
    Args:
        feature_prep:
        X_train:
        y_train:
        X_val:
        y_val:
        X_test:
        num_days:

    Returns:

    """

    logger("Training and predicting models...")
    params = {
        'num_leaves': 3,
        'objective': 'regression',
        'min_data_in_leaf': 200,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'metric': 'l2',
        'num_threads': 20
    }

    MAX_ROUNDS = 200
    output_dir = '/media/farmshare2/Research/raza/p_data/'
    logger.info('output directory : {}'.fromat(output_dir))
    val_pred = []
    test_pred = []
    cate_vars = []
    for i in range(16):
        logger.info("=" * 50)
        logger.info("Step %d" % (i+1))
        logger.info("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            categorical_feature=cate_vars,
            weight=pd.concat([feature_prep.items["perishable"]] * num_days) * 0.25 + 1
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,
            weight=feature_prep.items["perishable"] * 0.25 + 1,
            categorical_feature=cate_vars)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50
        )
        logger.info("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

    logger.info('**** Finished Training *****')

    logger.info("Validation mse:", mean_squared_error(
        y_val, np.array(val_pred).transpose()))

    weight = feature_prep.items["perishable"] * 0.25 + 1
    err = (y_val - np.array(val_pred).transpose())**2
    err = err.sum(axis=1) * weight
    err = np.sqrt(err.sum() / weight.sum() / 16)
    logger.info('nwrmsle = {}'.format(err))

    y_val = np.array(val_pred).transpose()
    df_preds = pd.DataFrame(
        y_val, index=feature_prep.df_2017.index,
        columns=pd.date_range("2017-07-26", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
    df_preds["unit_sales"] = np.clip(np.expm1(df_preds["unit_sales"]), 0, 1000)
    df_preds.reset_index().to_csv(output_dir + 'lgb_cv.csv', index=False)

    logger.info("Making submission...")
    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(
        y_test, index=feature_prep.df_2017.index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

    submission = feature_prep.test[["id"]].join(df_preds, how="left").fillna(0)
    submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
    submission.to_csv(output_dir + 'lgb_sub.csv', float_format='%.4f', index=None)

