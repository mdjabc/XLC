### Advanced XGBoost usage:
### https://www.kaggle.com/ashhafez/xgb-learning-rate-eta-decay
### https://github.com/dmlc/xgboost/issues/892

### May borrow some ideas from LightGBM:
### https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py

import io
import os
import gc
import sys
import time
import logging
import argparse
import datetime
import sklearn
# import numba
import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import *
from bayes_opt import *
from utils import *

class MyXGB:
    '''
    Some parameters in the Tree Booster session need to be reviewed.
    '''

    def __init__(self,
                 # Customized parameters:
                 train_data=None, booster_params=None,
                 # General parameters:
                 booster='gbtree', # Or 'gblinear', 'dart'
                 silent=0, # Or 1
                 nthread=-1,
                 random_seed=2018,
                 # Task parameters:
                 objective=None, # Ex: 'reg:linear', 'reg:logistic', 'binary:logistic', 'binary:logitraw', see: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
                 metric=None, # Ex: 'rmse', 'mae', 'logloss', 'error', 'auc', 'mlogloss'.
                 # Parameters for Tree Booster:
                 learning_rate=0.1, # alias: eta
                 gamma=0, # alias: min_split_loss
                 max_depth=6,
                 min_child_weight=1,
                 max_delta_step=0, # Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_lambda=1,
                 reg_alpha=0,
                 tree_method='auto',
                 # sketch_eps=0.03, # Usually no need to tune this.
                 scale_pos_weight=1 # Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases)
                 ):

        if booster_params != None:
            self.__booster_params = booster_params
        else:
            self.__booster_params = dict(
                booster=booster,
                silent=silent,
                nthread=nthread,
                objective=objective,
                eval_metric=metric,
                learning_rate=learning_rate,
                gamma=gamma,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step, # No tunable except for extreme cases in logistic regression.
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                tree_method=tree_method,
                # sketch_eps=sketch_eps,
                scale_pos_weight=scale_pos_weight,
                seed=random_seed
            )

        self.__precheck_booster_params()

        self.__train_data = train_data

    def __precheck_booster_params(self, booster_params=None):
        # placeholder for now.
        if booster_params == None:
            __booster_params = self.__booster_params
        else:
            __booster_params = booster_params

        try:
            _xgb = xgb.Booster(params=__booster_params)
            del _xgb
            gc.collect()
        except Exception as e:
            print('Input booster parameters are illegal. Please double check. Here is the ref:\n'
                  'https://github.com/dmlc/xgboost/blob/master/doc/parameter.md\n'
                  'Error: {}'.format(e))
            raise

    def get_booster_params(self):
        return self.__booster_params

    def train(self,
              train_data,
              params=None,
              num_boost_round=10,
              eta=0.1, # Alias: learning rate. Attn: Diff from learning_rates. See official doc for more details.
              evals=(),
              obj=None, # Customized objective function.
              feval=None, # Customized evaluation function.
              maximize=False, # Whether to maximize feval.
              early_stopping_rounds=100,
              evals_result=None,
              verbose_eval=100,
              xgb_model=None,
              callbacks=None,
              learning_rates=None,
              random_seed=2018,
              inplace_class_model=True
              ):

        try:
            if params == None:
                __params = self.__booster_params
            else:
                __params = params

            __params['eta'] = eta
            __params['seed'] = random_seed

            xgb_model = xgb.train(
                params=__params,
                dtrain=train_data,
                num_boost_round=num_boost_round,
                evals=evals,
                obj=obj,
                feval=feval,
                maximize=maximize,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=verbose_eval,
                xgb_model=xgb_model,
                callbacks=callbacks,
                learning_rates=learning_rates
            )

            if inplace_class_model:
                self.trained_xgb_model = xgb_model

            return xgb_model
        except Exception as e:
            print('Failed in training a XGBoost model. Error: {}'.format(e))
            raise

    def cross_validation(self,
                         params=None,
                         train_data=None,
                         eta=0.1,
                         num_boost_round=10,
                         nfold=5,
                         stratified=False,
                         folds=None,
                         metrics=(),
                         obj=None,
                         feval=None,
                         maximize=False,
                         early_stopping_rounds=100,
                         fpreproc=None,
                         as_pandas=True,
                         verbose_eval=100,
                         show_stdv=True,
                         random_seed=2018,
                         callbacks=None,
                         shuffle=True
                         ):
        try:
            if train_data == None:
                __train_data = self.__train_data
            else:
                __train_data = train_data

            if params == None:
                __params = self.__booster_params
            else:
                __params = params

            __params['eta'] = eta

            __cv_results = xgb.cv(
                params=__params,
                dtrain=__train_data,
                num_boost_round=num_boost_round,
                nfold=nfold,
                stratified=stratified,
                folds=folds,
                metrics=metrics,
                obj=obj,
                feval=feval,
                maximize=maximize,
                early_stopping_rounds=early_stopping_rounds,
                fpreproc=fpreproc,
                as_pandas=as_pandas,
                verbose_eval=verbose_eval,
                show_stdv=show_stdv,
                seed=random_seed,
                callbacks=callbacks,
                shuffle=shuffle
            )

            return __cv_results
        except Exception as e:
            print('Failed in running XGBoost cross-validation. Error: {}'.format(e))
            raise

    def __set_booster_params(self):

        __booster_params = dict(
            max_depth=(5, 8),
            min_child_weight=(0.5, 1.5),
            reg_alpha=(0, 1),
            reg_lambda=(0.5, 1.5),
            subsample=(0.04, 1),
            colsample_bytree=(0.04, 1),
            colsample_bylevel=(0.04, 1),
            gamma=(0, 0.5),
            scale_pos_weight=(0.5, 2.5)
        )

        return __booster_params

    def __eval_params_using_cv_pos(
            self,
            max_depth,
            min_child_weight,
            reg_alpha,
            reg_lambda,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            gamma,
            max_delta_step
            ):

        __params = dict(
            objective=self.__booster_params['objective'],
            eval_metric=self.__booster_params['eval_metric'],
            max_depth=int(np.round(max_depth)),
            min_child_weight=np.round(min_child_weight, 3),
            reg_alpha=np.round(reg_alpha, 3),
            reg_lambda=np.round(reg_lambda, 3),
            subsample=np.round(subsample, 3),
            colsample_bylevel=np.round(colsample_bylevel, 3),
            colsample_bytree=np.round(colsample_bytree, 3),
            gamma=np.round(gamma, 3),
            max_delta_step=int(np.round(max_delta_step))
        )

        __cv_results = self.cross_validation(
            params=__params,
            train_data=self.__train_data,
            eta=0.01,
            num_boost_round=10000,
            nfold=5,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        return np.max(__cv_results.iloc[:, 0])

    def __eval_params_using_cv_neg(
            self,
            max_depth,
            min_child_weight,
            reg_alpha,
            reg_lambda,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            gamma,
            max_delta_step
            ):

        __params = dict(
            objective=self.__booster_params['objective'],
            eval_metric=self.__booster_params['eval_metric'],
            max_depth=int(np.round(max_depth)),
            min_child_weight=np.round(min_child_weight, 3),
            reg_alpha=np.round(reg_alpha, 3),
            reg_lambda=np.round(reg_lambda, 3),
            subsample=np.round(subsample, 3),
            colsample_bylevel=np.round(colsample_bylevel, 3),
            colsample_bytree=np.round(colsample_bytree, 3),
            gamma=np.round(gamma, 3),
            max_delta_step=int(np.round(max_delta_step))
        )

        __cv_results = self.cross_validation(
            paras=__params,
            train_data=self.__train_data,
            eta=0.01,
            num_boost_round=10000,
            nfold=5,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        return -np.max(__cv_results.iloc[:, 0])

    def bayes_tuning(
            self,
            init_points=5,
            n_iter=25,
            acq='ei',
            xi=0.0,
            eval_func=None,
            pos_eval=True
        ):

        __params = self.__set_booster_params()

        try:
            if eval_func == None:
                if pos_eval:
                    bo = BayesianOptimization(self.__eval_params_using_cv_pos, __params)
                else:
                    bo = BayesianOptimization(self.__eval_params_using_cv_neg, __params)
            else:
                bo = BayesianOptimization(eval_func, __params)

            bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)

            opt_res = pd.DataFrame(bo.res['all']['params'])
            opt_res['values'] = bo.res['all']['values']

            return opt_res
        except Exception as e:
            print('Failed in Bayesian optimization. Error: {}'.format(e))
            raise














