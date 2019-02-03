# Usage Example:
# https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/#custom-loss-function-eval-metric

import io
import os
import sys
import time
import logging
import argparse
import datetime
import sklearn
import numba
import numpy as np
import pandas as pd
import catboost as cb
from catboost import *
from bayes_opt import *

from utils import *

class MyCat(utils):

    def __init__(self, train_data=None, booster_params=None, model_artifact=None, loss_function=None,
                 iterations=100, random_seed=7, l2_leaf_reg=3,
                 bootstrap_type='Bayesian', bagging_temperature=1, subsample=0.66, sampling_frequency='PerTreeLevel',
                 random_strength=1, depth=6, ignored_features=None, one_hot_max_size=5,
                 has_time=False, rsm=1, nan_mode='Min', calc_feature_importance=True,
                 fold_permutation_block_size=1,
                 # leaf_estimation_iterations=None,
                 leaf_estimation_method='Gradient',
                 fold_len_multiplier=2, approx_on_full_history=False, class_weights=None, thread_count=-1,
                 used_ram_limit=None, gpu_ram_part=0.95,
                 **kwargs):
        '''
        Wrapper of Yandex CatBoost. This project is still in DEV phase.

        For future improvements:
        (1) booster parameters: custom_metric and eval_metric.
        (2) Set learning rate here?
        (3) Parameter of 'use_best_model'.
        (4) Parameter of 'has_time'.
            Ref: https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list
        (5) Parameter of 'leaf_estimation_iterations'.
        (6) Parameter of 'class_weights'.
        (7) Performance settings.
        (8) Booster method: eval_metrics.
        (9) Booster method: staged_predict. (May work for reducing overfitting.)
        (10) Parameters for Bayesian Optimization.

        For usage examples:
        https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/#custom-loss-function-eval-metric

        For hyper-parameters tuning:
        https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/

        Core parameters:

        :param booster_params: User-defined booster core parameters. This will override the default parameters.
        :param model_artifact: The path where model artifact was stored.
        :param loss_function: Alias objective. Type: string (Default values set by CatBoost) or object (Customized python
                object, see the usage examples for more details). Supported metrics: RMSE, Logloss (default), MAE, C
                rossEntropy, Quantile, LogLinQuantile, SMAPE, MultiClass — Not supported on GPU,
                MultiClassOneVsAll — Not supported on GPU, MAPE, Poisson, PairLogit, QueryRMSE, QuerySoftMax.
                See this link:
                https://tech.yandex.com/catboost/doc/dg/concepts/loss-functions-docpage/#loss-functions
        :param iterations: Alias: num_boost_round, n_estimators, num_trees.
                The maximum number of trees that can be built when solving machine learning problems.
                When using other parameters that limit the number of iterations, the final number of trees may be less
                than the number specified in this parameter. Type: int.
        :param random_seed: Alias: random_state. The random seed used for training. CatBoost default value: None. A new
                random value is selected on each run. Type: int.
        :param l2_leaf_reg: Alias: reg_lambda.
                L2 regularization coefficient. Used for leaf value calculation. Any positive values are allowed.
                Default value: 3. Type: int.
        :param bootstrap_type: Bootstrap type. Defines the method for sampling the weights of objects.
                Supported methods: 'Poisson' (support GPU only), 'Bayesian', 'Bernoulli', 'No'
                Default value: 'Bayesian'. Type: string.
        :param bagging_temperature: Controls the intensity of Bayesian bagging. The higher the temperature, the more
               aggressive bagging will be. Typical values are in range of [0, 1]. 0 is no bagging. Possible values are
               in range of [0, inf).
               Default value: 1. Type: float.
        :param subsample: Sample rate for bagging. This parameter can be used if one of the following bootstrap types
               is defined: 'Poisson', 'Bernoulli'. Type: float.
        :param subsample_frequency: Frequency to sample weights and objects when building trees. Supported values:
               'PerTree' and 'PerTreeLevel'. Type: string.
        :param random_strength: Score standard deviation multiplier. Type: float.

        ###
        :param use_best_model: If this parameter is set, the number of trees that are saved in the resulting model is
               defined as follows: 1. Build the number of trees defined by the training parameters. 2. Use the test data
               set to identify the iteration with the optimal value of the metric specified in 'eval_metric'. This
               option requires a test dataset. Type: bool.
        ###

        :param depth: Depth of the tree. The value can be any integer up to 16. It is recommended to use values in the
               range of 1 to 10. Type: int.
        :param ignored_features: Indices of features to exclude from training. The non-negative indices that do not
               match any features are successfully ignored. The identifier corresponds to the features index. Feature
               indices used in train and feature importance are numbered from 0 to feature count-1. Type: list.
        :param one_hot_max_size: Use one-hot-encoding for all features whose distinct values <= given hyper-parameter
               value. Type: int.
        :param has_time: Use the order of subjects in the input data (do not perform random permutation).The timestamp
               column type is used to determine the order of objects if specified. Type: bool.
        :param rsm: Alias: colsample_bylevel.
                Random subspace method. The percentage of features to use at each split selection, when features are
                selected over again at random. Type: float.
        :param The method to process NaN within the input dataset. Support values: ('Forbidden': raise an exception,
               'Min', 'Max') Type: string.
        :param calc_feature_importance: This parameter turn on/off feature importance calculation. Type: bool.
        :param fold_permutation_block_size: Objects in the dataset are grouped in blocks before the random permutation.
               This parameter defines the size of the blocks. The smaller the value is, the slower the training will be.
               Too larger value may result in performance degradation. Type: int.
        :param leaf_estimation_iterations: The number of gradient steps when calculating the values in leaves. Type: int.
        :param leaf_estimation_method: The method used to calculate the values in leaves. Supported values: 'Newton',
               'Gradient'. Type: string.
        :param fold_len_multiplier: Coefficient for changing the length of folds. The value must be greater than 1.
               The best validation result is achieved with minimum values. With values close to 1, each iteration takes
               a quadratic amount of memory and time for the number of objects in the iteration.
               Thus, low values are possible only when there is a small number of objects.
        :param approx_on_full_history: The principles for calculating the approximated values. If set to True,
               will use all the preceding rows in the fold for calculating the approximated values.
               This mode is slower and in rare cases slightly more accurate. Type: bool.
        :param class_weights: Classes weights. The values are used as multipliers for the object weights.
               This parameter can be used for solving classification and multi-classification problems.
               For imbalanced datasets with binary classification the weight multiplier can be set to 1 for class 0 and
               (sum_negative/sum_positive) for class 1. For example,  class_weights=[0.1, 4] multiplies the weights of
               objects from class 0 by 0.1 and the weights of object from class 1 by 4.
        :param thread_count: The number of threads to use during training. The purpose depends on the selected processing unit:
               CPU: For CPU, Optimizes training time. This parameter doesn't affect results. For GPU, The given value is
               used for reading the data from the hard drive and does not affect the training. During the training one
               main thread and one thread for each GPU are used. GPU: The given value is used for reading the data from
               the hard drive and does not affect the training. During the training one main thread and one thread for
               each GPU are used. Type: int.
        :param used_ram_limit: The maximum amount of memory available to the CTR calculation process.
               Format: <size><measure of information>
               Supported measures of information (non case-sensitive): MB, KB, GB. Type: string.
        :param gpu_ram_part: How much of the GPU RAM to use for training. Type: float.

        :param objective:

        :param kwargs:
        '''

        self.__train_data = train_data

        self.__trained_cat = None

        self.__log_stream = io.StringIO()
        logging.basicConfig(stream=self.__log_stream, level=logging.INFO)

        # self.__objective = objective
        self.__loss_function = loss_function

        self.__model_artifact = model_artifact

        if booster_params != None:
            self.__booster_params = booster_params
        else:
            self.__booster_params = {
                'loss_function':self.__loss_function,
                'iterations':iterations,
                'random_seed':random_seed,
                'l2_leaf_reg':l2_leaf_reg,
                'bootstrap_type':bootstrap_type,
                'bagging_temperature':bagging_temperature,
                # 'subsample':subsample,
                'sampling_frequency':sampling_frequency,
                'random_strength':random_strength,
                # 'use_best_model':use_best_model,
                'depth':depth,
                # 'ignored_features':ignored_features,
                'one_hot_max_size':one_hot_max_size,
                'has_time':has_time,
                'rsm':rsm,
                'nan_mode':nan_mode,
                # 'fold_permutation_block_size': fold_permutation_block_size,
                # 'calc_feature_importance':calc_feature_importance,
                # 'leaf_estimation_iterations':leaf_estimation_iterations,
                'leaf_estimation_method':leaf_estimation_method,
                'fold_len_multiplier':fold_len_multiplier,
                'approx_on_full_history':approx_on_full_history,
                # 'class_weights':class_weights,
                # 'thread_count':thread_count,
                'used_ram_limit':used_ram_limit,
                'gpu_ram_part':gpu_ram_part
            }

        self.__precheck_input_params()

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

    # Log functions are not enabled for now.
    def get_log(self):
        """
        Retrive running log.
        :return:
        """
        __log = self.__log_stream.getvalue()
        return __log

    @classmethod
    def make_pool(cls, data, label=None, cat_vars=None, feature_names=None, weight=None, thread_count=-1, **kwargs):
        try:
            _pooled_data = Pool(data=data, label=label, cat_features=cat_vars, feature_names=feature_names,
                                weight=weight, thread_count=thread_count, **kwargs)
            return _pooled_data
        except Exception as e:
            logging.error('Failed in creating data pool. Error: {}'.format(e))
            raise

    # def train(self, X, y=None, eval_set=None, iterations=500, learning_rate=0.1, random_seed=8, early_stop=100, plot=False,
              # verbose=None, **kwargs):

        # if isinstance(early_stop, int):
            # self.__booster_params['od_type'] = 'Iter'
            # self.__booster_params['od_wait'] = early_stop
        # else:
            # logging.error('Provided an illegal early stop value.')
            # raise TypeError('Early stop value can only be integers.')

    # Use provided parameters to train model.
    def train(self, data_pool, eval_set=None, iterations=500, learning_rate=0.1, random_seed=8, early_stop=100, plot=False,
              verbose=None, inplace_class_model=True, **kwargs):

        try:
            _cat_config = self.__booster_params
            _cat_config['iterations'] = int(iterations)
            _cat_config['random_seed'] = int(random_seed)
            _cat_config['learning_rate'] = learning_rate

            if isinstance(early_stop, int):
                _cat_config['od_type'] = 'Iter'
                _cat_config['od_wait'] = int(early_stop)
            else:
                logging.error('Provided an illegal early stop value.')
                raise TypeError('Early stop value can only be integers.')

            self.__precheck_input_params(booster_params=_cat_config)

            _this_cat = CatBoost(params=_cat_config)
            _this_trained_cat = _this_cat.fit(X=data_pool, eval_set=eval_set, verbose=verbose, plot=plot, **kwargs)

            # Store feature names for analyzing feature importance.
            self.__feature_names = data_pool.get_feature_names()

            if inplace_class_model:
                self.__trained_cat = _this_trained_cat

            return _this_trained_cat
        except Exception as e:
            logging.error('Failed in training a CatBoost model. Error: {}.'.format(e))
            raise

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

    def __set_booster_params(self):

        __booster_params = {
                                # "fold_len_multiplier": [1, 5],
                                "reg_lambda": [1, 5],
                                "colsample_bylevel": [0.1, 1],
                                "bagging_temperature": [0, 1.5],
                                # "one_hot_max_size": [2, 50],
                                "subsample": [0.05, 1], # Used only for bootstrap types is either 'Poisson' or 'Bernoulli'.
                                "max_depth": [5, 9], # Alias: max_depth
                                "bootstrap_type":[0, 2], # Special. Label-encoded bootstrap methods.
                                "nan_mode": [0, 1], # Special. Label-encoded nan modes.
                                "scale_pos_weight":[0.8, 1.5] # Special. Label-encoded.
                            }

        return __booster_params

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
















