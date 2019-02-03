# Usage Example:
# https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/#custom-loss-function-eval-metric

# Data visualization:
# https://tech.yandex.com/catboost/doc/dg/installation/python-installation-additional-data-visualization-packages-docpage/#python-installation-additional-data-visualization-packages

# Future improvements:
# Pairwise training?
# Feature data?
# Test set cat feats in pool and cat feats in fit method.

import io
import os
import sys
import json
import time
import logging
import argparse
import datetime
from datetime import datetime as dt
import sklearn
import numba
import numpy as np
import pandas as pd
import catboost as cb
from catboost import *
from bayes_opt import *

from sklearn.metrics import *
from sklearn.model_selection import *

# from utils import *

class MyCat(object):

    def __init__(self,
                 # Class parameters:
                 train_data=None,
                 log_dir=None,

                 # Booster parameters:
                 catboost_reg_params='./catboost_reg_params.json',
                 catboost_clf_params='./catboost_clf_params.json',
                 booster_params=None,
                 model_artifact=None,
                 objective=None, # alias: loss_function.
                 metric=None,
                 **kwargs
                 ):

        # Record system time to keep log names and other solutions consistent:
        __timestamp_format = "%Y-%m-%d-%H-%M-%S"
        self.start_time = dt.now().strftime(__timestamp_format)

        # Initialize logging.
        self.__init_logging(log_dir=log_dir)

        # Create a tuple to store default CGBM objectives and precheck user-defined objective.
        self.__set_catboost_default_objectives()
        if isinstance(objective, str):
            assert objective in self.__catboost_default_clf_objectives or objective in self.__catboost_default_reg_objectives, \
                'Illegal input objective, it can only be within:\n{}'.format(
                self.__catboost_default_clf_objectives + self.__catboost_default_reg_objectives
            )

        # Load booster core parameters:
        with open(catboost_clf_params, 'r') as clf, open(catboost_reg_params, 'r') as reg:
            self.__catboost_default_clf_params, self.__catboost_default_reg_params = json.load(clf), json.load(reg)

        # if thread_count <=4 and thread_count > 0:
        #     print('Warning: Parameter of thread_count is low: {}'.format(thread_count))

        # self.__objective = objective
        # self.__loss_function = loss_function

        self.__booster_params = {
            'objective' : objective,
            'loss_function':objective,
            # 'metric' : metric
        }

        # if booster_params != None:
        #     self.__booster_params = booster_params
        # else:
        #     self.__booster_params = {
        #         'objective':objective,
        #         'custom_metric':metric,
        #         'iterations':iterations,
        #         'random_seed':random_seed,
        #         'l2_leaf_reg':l2_leaf_reg,
        #         'bootstrap_type':bootstrap_type,
        #         'bagging_temperature':bagging_temperature,
        #         # 'subsample':subsample,
        #         'sampling_frequency':sampling_frequency,
        #         'random_strength':random_strength,
        #         # 'use_best_model':use_best_model,
        #         'depth':depth,
        #         # 'ignored_features':ignored_features,
        #         'one_hot_max_size':one_hot_max_size,
        #         'has_time':has_time,
        #         'rsm':rsm,
        #         'nan_mode':nan_mode,
        #         # 'fold_permutation_block_size': fold_permutation_block_size,
        #         # 'calc_feature_importance':calc_feature_importance,
        #         # 'leaf_estimation_iterations':leaf_estimation_iterations,
        #         'leaf_estimation_method':leaf_estimation_method,
        #         'fold_len_multiplier':fold_len_multiplier,
        #         'approx_on_full_history':approx_on_full_history,
        #         # 'class_weights':class_weights,
        #         'thread_count':thread_count,
        #         'used_ram_limit':used_ram_limit,
        #         'gpu_ram_part':gpu_ram_part
        #     }

        # self.__precheck_input_params()

    def __init_logging(self, log_dir=None):
        try:
            if log_dir is None:
                self.__log_stream = io.StringIO()
                logging.basicConfig(
                    stream=self.__log_stream,
                    level=logging.INFO,
                    format='[%(asctime)s (Local Time)] %(levelname)s : %(message)s', # Local time may vary for cloud services.
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                )
            else:
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)

                log_file = 'CatBoost_ToolBox_log_{}.txt'.format(self.start_time)
                self.log_file_dir = log_dir + log_file
                logging.basicConfig(
                    filename=self.log_file_dir,
                    level=logging.INFO,
                    format='[%(asctime)s (Local Time)] %(levelname)s : %(message)s', # Local time may vary for cloud services.
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                )
        except Exception as e:
            print('Failed in logging. Error: {}'.format(e))
            raise

    def __precheck_input_params(self, booster_params=None):
        '''
        Precheck the if the CatBoost core can recognize all the input parameters.
        :param booster_params: Parameter of CatBoost core. Type: dict.
        :return: NULL.
        '''
        try:
            if booster_params == None:
                _small_cat = CatBoost(params=self.__booster_params)
            else:
                _small_cat = CatBoost(params=booster_params)

            del _small_cat
        except Exception as e:
            logging.error('Illegal input parameters.')
            raise ValueError('Please double check your input hyper-parameters.')

    def get_log(self):
        """
        Retrive running log.
        :return:
        """
        try:
            __log = self.__log_stream.getvalue()
            return __log
        except Exception as e:
            print("Failed in getting log stream. Error: {}".format(e))

    def __set_catboost_default_objectives(self):
        self.__catboost_default_unclassified_objectives = (
            'Quantile', 'LogLinQuantile',  'Poisson', 'PairLogit', 'PairLogitPairwise', 'QueryRMSE',
            'QuerySoftMax', 'YetiRank', 'YetiRankPairwise'
        )

        self.__catboost_default_clf_objectives = (
            'Logloss', 'CrossEntropy', 'MultiClass', 'MultiClassOneVsAll',
        )

        self.__catboost_default_reg_objectives = (
            'RMSE', 'MAE', 'SMAPE', 'MAPE'
        )

    @property
    def objective(self):
        return self.__booster_params['objective']

    @objective.setter
    def objective(self, objective):
        if isinstance(objective, str):
            assert objective in self.__catboost_default_objective, "Illegal input objective."
            "CatBoost default objectives: {}".format(
                self.__catboost_default_objective
            )
            self.__booster_params["objective"] = objective
            logging.info("Updated objective to {}".format(objective))

    @property
    def trained_model(self):
        return self.__trained_model

    @trained_model.setter
    def trained_model(self, *args, **kwargs):
        logging.warning('Attempt to change trained CatBoost model outside the class.')
        raise ValueError('Cannot change trained CatBoost model outside the class.')

    @property
    def is_trained(self):
        return True if self.__trained_model is not None else False

    @classmethod
    def make_pool(cls,
                  data,
                  label=None,
                  cat_features=None,
                  feature_names=None,
                  weight=None,
                  pairs=None,
                  thread_count=-1,
                  **kwargs
                  ):
        try:
            _pooled_data = Pool(data=data, label=label, cat_features=cat_features, feature_names=feature_names, pairs=pairs,
                                weight=weight, thread_count=thread_count, **kwargs)
            return _pooled_data
        except Exception as e:
            logging.error('Failed in creating data pool. Error: {}'.format(e))
            raise

    # Wrapper of catboost train methods.
    def train(self,
              train_X,
              train_y,
              valid_X=None,
              valid_y=None,
              booster_params=None,
              eval_set=None,
              num_iterations=500,
              learning_rate=0.1,
              random_seed=8,
              early_stopping_rounds=100,
              plot=False,
              verbose=None,
              inplace_class_model=True,
              **kwargs):

        assert isinstance(num_iterations, int) and isinstance(learning_rate, float) and isinstance(random_seed, int),  \
            'Please double check your input parameters.'
        _cat_config = self.__booster_params if booster_params is None else booster_params
        _cat_config['iterations'] = num_iterations
        _cat_config['random_seed'] = random_seed
        _cat_config['learning_rate'] = learning_rate

        if isinstance(early_stopping_rounds, int):
            _cat_config['od_type'] = 'Iter'
            _cat_config['od_wait'] = early_stopping_rounds
        elif early_stopping_rounds is None:
            print('Warning: No early stopping was set up. CatBoost will train {} iterations.'.format(num_iterations))
        else:
            logging.error('Provided an illegal early stop value.')
            raise TypeError('Provided an illegal early stop value.')

        # Make validation data pool(s).
        if valid_X is None and valid_y is None:
            logging.info('Randomly split training data into 70% to 30% using class random seed.')
            train_X_, valid_X_, train_y_, valid_y_ = train_test_split(train_X, train_y, test_size=0.3,
                                                                      random_state=self.__booster_params['random_seed'])
            train_pool = MyCat.make_pool(data=train_X_, label=train_y_)
            valid_pool = MyCat.make_pool(data=valid_X_, label=valid_y_)
            eval_set = [valid_pool] # Loss of training data will be displayed by default.
        else:
            train_pool = MyCat.make_pool(train_X, train_y)
            eval_set = []
            valid_X_ = [valid_X] if not isinstance(valid_X, list) else valid_X
            valid_y_ = [valid_y] if not isinstance(valid_y, list) else valid_y

            assert len(valid_X_) == len(valid_y_), 'Input data size mis-matched.'

            for idx in range(len(valid_X_)):
                tmp_valid_pool = MyCat.make_pool(valid_X_[idx], valid_y_[idx])
                eval_set.append(tmp_valid_pool)

        # Model training.
        trained_cb_model = cb.train(
            params = _cat_config,
            pool = train_pool,
            iterations = num_iterations,
            eval_set = eval_set,
            verbose = verbose,
            plot = plot
        )

        if inplace_class_model:
            self.__trained_model = trained_cb_model

        return trained_cb_model

    # Use provided parameters to train model.
    def fit(self,
            #data_pool,
            train_X,
            train_y,
            valid_X=None,
            valid_y=None,
            booster_params=None,
            eval_metric=None,
            cat_features=None,
            iterations=100,
            learning_rate=0.1,
            random_seed=8,
            early_stopping_rounds=100,
            sample_weight=None,
            plot=False,
            verbose=None,
            inplace_class_model=True,
            save_snapshot=False,
            snapshot_file=None,
            snapshot_interval=60,
            **kwargs
            ):

        # _cat_config = self.__booster_params if booster_params is None else booster_params
        # _cat_config['iterations'] = int(iterations)
        # _cat_config['random_seed'] = int(random_seed)
        # _cat_config['learning_rate'] = learning_rate
        #
        # if isinstance(early_stop, int):
        #     _cat_config['od_type'] = 'Iter'
        #     _cat_config['od_wait'] = int(early_stop)
        # else:
        #     logging.error('Provided an illegal early stop value.')
        #     raise TypeError('Early stop value can only be integers.')

        # Make validation data pool(s).
        if valid_X is None and valid_y is None:
            logging.info('Randomly split training data into 70% to 30% using class random seed.')
            train_X_, valid_X_, train_y_, valid_y_ = train_test_split(train_X, train_y, \
                                                                      test_size=0.3,random_state=random_seed)
            train_pool = MyCat.make_pool(data=train_X_, label=train_y_, cat_features=cat_features)
            valid_pool = MyCat.make_pool(data=valid_X_, label=valid_y_, cat_features=cat_features)
            eval_set = [valid_pool] # Loss of training data will be displayed by default.
        else:
            train_pool = MyCat.make_pool(data=train_X, label=train_y, cat_features=cat_features)
            eval_set = []
            valid_X_ = [valid_X] if not isinstance(valid_X, list) else valid_X
            valid_y_ = [valid_y] if not isinstance(valid_y, list) else valid_y

            assert len(valid_X_) == len(valid_y_), 'Input data size mis-matched.'

            for idx in range(len(valid_X_)):
                tmp_valid_pool = MyCat.make_pool(data=valid_X_[idx], label=valid_y_[idx], cat_features=cat_features)
                eval_set.append(tmp_valid_pool)

        # Build model.
        # __booster_params = self.__catboost_default_clf_params \
        #                    if self.__booster_params['objective'] in \
        #                    self.__catboost_default_clf_objectives else \
        #                    self.__catboost_default_reg_params
        # __booster_params['loss_function'] = self.__booster_params['objective']

        if booster_params is not None:
            assert isinstance(booster_params, dict), 'Booster params can only be dict.'
            __booster_params = booster_params
            # for key in booster_params:
            #     assert key in __booster_params.keys(), 'Illegal input parameter: {}'.format(key)
            #     __booster_params[key] = booster_params[key]
        else:
            if self.__booster_params['objective'] in self.__catboost_default_clf_objectives:
                print('Classification')
                __booster_params = self.__catboost_default_clf_params
                __booster_params['loss_function'] = self.__booster_params['objective']
                __booster_params['eval_metric'] = eval_metric
                __booster_params['iterations'] = iterations
                __booster_params['learning_rate'] = learning_rate
                __booster_params['random_seed'] = random_seed
                for key in kwargs.keys():
                    assert key in __booster_params.keys(), 'Unexpected input parameter: {}'.format(key)
                    __booster_params[key] = kwargs[key]
                # this_cat = CatBoostClassifier(**__booster_params)
            elif self.__booster_params['objective'] in self.__catboost_default_reg_objectives:
                print('Regression')
                __booster_params = self.__catboost_default_reg_params
                __booster_params['loss_function'] = self.__booster_params['objective']
                __booster_params['eval_metric'] = eval_metric
                __booster_params['iterations'] = iterations
                __booster_params['learning_rate'] = learning_rate
                __booster_params['random_seed'] = random_seed
                for key in kwargs.keys():
                    assert key in __booster_params.keys(), 'Unexpected input parameter: {}'.format(key)
                    __booster_params[key] = kwargs[key]
                # this_cat = CatBoostRegressor(**__booster_params)
            else:
                raise ValueError('User defined objective is illegal.')

        try:
            this_cat = None
            # self.__precheck_input_params(booster_params=_cat_config)
            if __booster_params['loss_function'] in self.__catboost_default_clf_objectives:
                this_cat = CatBoostClassifier(**__booster_params)
            elif __booster_params['loss_function'] in self.__catboost_default_reg_objectives:
                this_cat = CatBoostRegressor(**__booster_params)

            if save_snapshot:
                assert snapshot_file is not None, 'Snapshot file cannot be None.'

            _this_trained_cat = this_cat.fit(
                X=train_pool,
                eval_set=eval_set,
                verbose=verbose,
                sample_weight=sample_weight,
                plot=plot,
                early_stopping_rounds=early_stopping_rounds,
                save_snapshot=save_snapshot,
                snapshot_file=snapshot_file,
                snapshot_interval=snapshot_interval,
                **kwargs
            )

            # Store feature names for analyzing feature importance.
            self.__feature_names = train_pool.get_feature_names()

            if inplace_class_model:
                self.__trained_cat = _this_trained_cat

            return _this_trained_cat
        except Exception as e:
            logging.error('Failed in training a CatBoost model. Error: {}.'.format(e))
            raise

    def kfolds_cv_ensemble(self,
                           train_X,
                           train_y,
                           booster_params=None,
                           folds=3,
                           eval_metric=None,
                           cat_features=None,
                           iterations=100,
                           learning_rate=0.1,
                           random_seed=8,
                           kfold_random_status=7,
                           shuffle=True,
                           early_stopping_rounds=100,
                           sample_weight=None,
                           predict_proba=False,
                           plot=False,
                           verbose=None,
                           save_models=True,
                           **kwargs
                           ):

        # Make kfold splits.
        kfold = KFold(n_splits=folds, random_state=kfold_random_status, shuffle=shuffle)

        if booster_params is not None:
            __booster_params = booster_params
        else:
            __booster_params = self.__booster_params
            for key in kwargs.keys():
                assert key in __booster_params.keys(), 'Illegal parameter: {}'.format(key)
                __booster_params[key] = booster_params[key]

        if isinstance(train_X, pd.DataFrame):
            train_X_ = train_X.values
        elif isinstance(train_X, np.ndarray):
            train_X_ = train_X
        else:
            raise TypeError('For this method, only pd.DataFrame and np.ndarray are supported.')

        if isinstance(train_y, pd.DataFrame):
            train_y_ = train_y.values.squeeze()
        elif isinstance(train_y, np.ndarray):
            train_y_ = train_y.squeeze()
        else:
            raise TypeError('For this method, only pd.DataFrame and np.ndarray are supported.')

        print('Start K-Fold cross validation (ensemble)...')
        preds = []
        trained_models = []
        for i, (train_idx, valid_idx) in enumerate(kfold.split(train_X_)):
            print('\nFor fold {}...\n'.format(i+1))
            fold_train_X = train_X_[train_idx]
            fold_valid_X = train_X_[valid_idx]
            fold_train_y = train_y_[train_idx]
            fold_valid_y = train_y_[valid_idx]

            fold_booster = self.fit(
                train_X=fold_train_X,
                train_y=fold_train_y,
                valid_X=fold_valid_X,
                valid_y=fold_valid_y,
                booster_params=__booster_params,
                eval_metric=eval_metric,
                cat_features=cat_features,
                iterations=iterations,
                learning_rate=learning_rate,
                random_seed=random_seed,
                early_stopping_rounds=early_stopping_rounds,
                sample_weight=sample_weight,
                plot=plot,
                verbose=verbose,
                inplace_class_model=False,
                **kwargs
            )

            pred_pool = MyCat.make_pool(data=train_X_, cat_features=cat_features)
            if predict_proba:
                preds.append(fold_booster.predict_proba(pred_pool))
            else:
                preds.append(fold_booster.predict(pred_pool))

            if save_models:
                trained_models.append(fold_booster)

        # Average predictions.
        # preds /= float(folds)

        if save_models:
            return preds, trained_models
        else:
            return preds

    def analyze_feature_importance(self, top_features=50, plot=True):

        if self.__trained_cat == None:
            raise('No trained CatBoost model is loaded for analysis.')

        try:
            self.__feature_importance = pd.DataFrame()
            self.__feature_importance['Features'] = self.__feature_names
            self.__feature_importance['Importances'] = self.__trained_cat.feature_importances_

            if self.__feature_importance.shape[0] < int(top_features):
                print('Selected top feature numbers are larger than existing features.')
                _top_N = self.__feature_importance.shape[0]
            else:
                _top_N = top_features

            if plot:
                self.__feature_importance.set_index('Features').sort_values('Importances', ascending=True).tail(_top_N).plot(figsize=(5,10), fontsize=12, kind='barh')
        except Exception as e:
            raise('Error: {}'.format(e))

    def cross_validation(self, data_pool, params, nfold=5, train_valid_inverted=False, random_seed=2018,
                         iterations=50000,shuffle=True, logging_level=None, stratified=False,
                         save_as_pandas=True, plot=False, verbose_eval=100, early_stopping_rounds=100):

        try:
            print('Running {} folds cross-validation...'.format(nfold))
            _output = cb.cv(pool=data_pool, params=params, iterations=iterations, fold_count=nfold,
                            inverted=train_valid_inverted, partition_random_seed=random_seed, shuffle=shuffle,
                            stratified=stratified, as_pandas=save_as_pandas, plot=plot, verbose_eval=verbose_eval,
                            early_stopping_rounds=early_stopping_rounds)

            return _output
        except Exception as e:
            raise('Failed in cross-validation by CatBoost. Error: {}'.format(e))

    def __set_opt_booster_params(self):

        __opt_booster_params = {
                                "reg_lambda": [0.1, 10],
                                "colsample_bylevel": [0.05, 1],
                                "bagging_temperature": [0, 1.5],
                                # "one_hot_max_size": [2, 50],
                                "subsample": [0.05, 1], # Used only for bootstrap types is either 'Poisson' or 'Bernoulli'.
                                "max_depth": [2, 10], # Alias: depth
                                "bootstrap_type":[0, 1], # Special. Label-encoded bootstrap methods.
                                "nan_mode": [0, 1], # Special. Label-encoded nan modes.
                                "scale_pos_weight":[0.8, 1.5] # Special. Label-encoded.
                            }

        return __opt_booster_params

    # Needs to be updated according to the updated booster params.
    def __eval_params_using_cv(self, rsm,
                               # bagging_temperature,
                               l2_leaf_reg,
                               # subsample,
                               one_hot_max_size,
                               depth, bootstrap_type,
                               nan_mode, leaf_estimation_method):

        bootstrap_types = ['Bayesian', 'Bernoulli']
        nan_modes = ['Min', 'Max']
        leaf_estimation_methods = ['Newton', 'Gradient']

        _bootstrap_type = bootstrap_types[int(np.round(bootstrap_type))]

        __params = dict(
            loss_function=self.__booster_params['loss_function'],
            rsm=rsm,
            # bagging_temperature=bagging_temperature, # Cannot use with Bayesian.
            l2_leaf_reg=l2_leaf_reg,
            # subsample=subsample,
            one_hot_max_size=int(np.round(one_hot_max_size)),
            depth=int(np.round(depth)),
            bootstrap_type=_bootstrap_type,
            nan_mode=nan_modes[int(np.round(nan_mode))],
            leaf_estimation_method=leaf_estimation_methods[int(np.round(leaf_estimation_method))]
        )

        __cv_results = self.cross_validation(data_pool=self.__train_data, params=__params, iterations=50000, nfold=10,
                                             shuffle=True, verbose_eval=200, early_stopping_rounds=100)

        return -np.max(__cv_results.iloc[:, 0])

    def bayes_tuning(self, init_points=5, n_iter=25, acq='ucb', kappa=1, eval_params=None, eval_func=None,
                     learning_rate=0.03, metric=None, **kwargs):

        try:
            # if mode == 'prefer_exploitation':
            #     bo = BayesianOptimization(__eval_func, __params)
            #     bo.maximize(init_point=init_point, n_iter=n_iter, acq='ucb', kappa=1)
            #     return bo
            # if mode == 'prefer_exploration':
            #     bo = BayesianOptimization(__eval_func, __params)
            #     bo.maximize(init_point=init_point, n_iter=n_iter, acq='ucb', kappa=10)

            if eval_params == None:
                __params = self.__set_booster_params()
            else:
                __params = eval_params

            # self.__precheck_input_params(booster_params=__params)

            if eval_func == None:
                bo = BayesianOptimization(self.__eval_params_using_cv, __params)
            else:
                bo = BayesianOptimization(eval_func, __params)

            bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa)

            return bo
        except Exception as e:
            print('Failed in Bayesian optimization. Error: {}'.format(e))
            raise

    def get_params(self):
        '''
        Return parameters used in CatBoost model training.
        :return:
        '''
        try:
            return self.__trained_cat.get_params()
        except Exception as e:
            print('Failed in returning training hyper-parameters. Error: {}'.format(e))
            raise

    def load_model(self, directory, format='catboost', inplace_class_model=False):
        try:
            _trained_cat = CatBoost.load_model(fname=directory, format=format)
            print('Successfully loaded the trained CatBoost model.')
            if inplace_class_model:
                self.__model_artifact = _trained_cat
            return _trained_cat
        except Exception as e:
            print('Failed in loading pre-trained model artifact. Error: {}'.format(e))
            raise

    def save_model(self, directory, trained_model=None, format='cbm', export_parameters=None):
        try:
            if format not in ('cbm', 'coreml', 'cpp'):
                raise ValueError('Format of model artifact can only be one of these: cbm, coreml or cpp.')

            if trained_model == None:
                _model = self.__model_artifact
            else:
                _model = trained_model

            _model.save_model(fname=directory, format=format, export_parameters=export_parameters)
            del _model
            print('Trained CatBoost model has been saved successfully.')
        except Exception as e:
            print('Failed in saving trained model artifact. Error: {}'.format(e))
            raise

    def predict(self, pred_pool, model=None, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, thread_count=-1,
                verbose=False):

        try:
            if model == None:
                _model = self.__model_artifact
            else:
                _model = model

            _output_res = _model.predict(pred_pool, prediction_type=prediction_type, ntree_start = ntree_start,
                                         ntree_end=ntree_end, thread_count=thread_count, verbose=verbose)

            return _output_res
        except Exception as e:
            print('Cannot make prediction. Error: {}'.format(e))
















