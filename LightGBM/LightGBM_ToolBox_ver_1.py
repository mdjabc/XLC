# Official Documents of LightGBM:
# https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#config

# Interactive hyperparameters doc:
# https://sites.google.com/view/lauraepp/parameters

# Quick start:
# https://lightgbm.readthedocs.io/en/latest/Quick-Start.html#examples

# Advanced usage:
# https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
# Cat vars:
# https://www.kaggle.com/c/home-credit-default-risk/discussion/58950
# https://github.com/Microsoft/LightGBM/pull/762
# https://github.com/Microsoft/LightGBM/issues/699

# Further improvements:
# Data Visualization
# https://github.com/slundberg/shap
# While making lgb.Dataset, set free_raw to False to reuse the data.

# Support multiple metrics.

import io
import os
import sys
import time
import logging
import argparse
import datetime
import sklearn
# import numba
import numpy as np
import pandas as pd

from bayes_opt import *
import lightgbm as lgb
from lightgbm import *

from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error

from utils import *

class MyLight(utils):
    '''
    This is a high level wrapper of LightGBM, still in DEV phase.

    For advanced usage:
    https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84

    For all hyper-parameters:
    https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#config

    For hyper-parameters tuning:
    http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

    Future improvement:
    (1) boosting type.
    (2) Objective parameters.
	(3) Callback function in training API.
	(4) Need to add visualization module of training processes.
	(5) Dict of hyper-parameters.

    :param objective: default = regression, type = enum, options: regression, regression_l1, huber, fair, poisson,
            quantile, mape, gammma, tweedie, binary, multiclass, multiclassova, xentropy, xentlambda, lambdarank,
            aliases: objective_type, app, application
            More details: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#objective
    :param boosting: More details: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#boosting.
    :param num_iterations 🔗︎, default = 100, type = int, aliases: num_iteration, num_tree, num_trees, num_round,
            num_rounds, num_boost_round, n_estimators, constraints: num_iterations >= 0
    '''

    def __init__(self,
                # Tool box parameters setup.
                booster_params=None,
                train_data=None,
                valid_data=None,
                log_dir=None,

                # Core booster parameters:
                objective=None,
                boosting='gbdt',
                num_iterations=100,
                learning_rate=0.1,
                num_leaves=31,
                tree_learner='serial',
                num_thread=0,
                # device_type='cpu',
                seed=2018,

                # Learning control parameters:
                max_depth=-1,
                min_data_in_leaf=20,
                min_sum_hessian_in_leaf=1e-3,
                bagging_fraction=1.0,
                bagging_freq=0,
                bagging_seed=3,
                feature_fraction=1.0,
                feature_fraction_seed=2,
                early_stopping_round=0,
                # max_delta_step=0.0,
                lambda_l1=0.0,
                lambda_l2=0.0,
                min_gain_to_split=0.0,
                drop_rate=0.1,
                max_drop=50,
                skip_drop=0.5,
                xgboost_dart_mode=False,
                uniform_drop=False,
                drop_seed=4,
                top_rate=0.2,
                other_rate=0.1,
                # min_data_per_group=100,
                # max_cat_threshold=32,
                # cat_l2=10.0,
                # cat_smooth=10.0,
                # max_cat_to_onehot=4,
                top_k=20,
                # monotone_constraints=None,
                # feature_contri=None, # Got an error saying unexpected parameter?
                forcedsplits_filename='',
                # IO parameters:
                verbose=1,
                max_bin=255,
                min_data_in_bin=3,
                subsample_for_bin=200000,
                histogram_pool_size=-1.0,
                data_random_seed=1,
                is_sparse=True,
                sparse_threshold=0.8,
                use_missing=True,
                two_round=False,
                save_binary=False,
                enable_load_from_binary_file=True,
                categorical_feature='',
                # Objective parameters:
                num_class=1,
                is_unbalanced=False,
                # Metric parameters:
                metric='',
                metric_freq=1,
                # Network parameters:
                num_machines=1,

                # GPU parameters:
                gpu_platform_id=-1,
                gpu_device_id=-1,
                gpu_use_dp=False
        ):

        # self.__create_log()
        if log_dir != None:
            self.__init_logging(log_dir=log_dir)
        else:
            self.__init_logging()

        logging.info('External logging initialized.')

        # Create a tuple to store default LGBM objectives and precheck user-defined objective.
        self.__set_lgb_default_objectives()
        assert objective in self.__lgb_default_objectives, 'Illegal input objective, it can only be within:\n{}'.format(
            self.__lgb_default_objectives
        )

        # Create a tuple to store default LGBM metrics and precheck user-defined metric.
        self.__set_lgb_default_metrics()
        if not isinstance(metric, list):
            metric = [metric]
        for _metric in metric:
            assert _metric in self.__lgb_default_metrics, 'Illegal input metric(s), can only be within:\n{}'.format(
                self.__lgb_default_metrics
            )

        self.trained_model = None

        self.__train_data = train_data
        self.__valid_data = valid_data

        if booster_params != None:
            self.__booster_params = booster_params
        else:
            self.__booster_params = {}
            self.__booster_params['objective'] = objective
            self.__booster_params['metric'] = metric
            self.__booster_params['boosting'] = boosting
            self.__booster_params['num_iterations'] = num_iterations
            self.__booster_params['learning_rate'] = learning_rate
            self.__booster_params['num_leaves'] = num_leaves
            self.__booster_params['tree_learner'] = tree_learner
            self.__booster_params['num_thread'] = num_thread
            # self.__booster_params['device_type'] = device_type
            self.__booster_params['seed'] = seed
            self.__booster_params['max_depth'] = max_depth
            self.__booster_params['min_data_in_leaf'] = min_data_in_leaf
            self.__booster_params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
            self.__booster_params['bagging_fraction'] = bagging_fraction
            self.__booster_params['bagging_freq'] = bagging_freq
            self.__booster_params['bagging_seed'] = bagging_seed
            self.__booster_params['feature_fraction'] = feature_fraction
            self.__booster_params['feature_fraction_seed'] = feature_fraction_seed
            self.__booster_params['early_stopping_round'] = early_stopping_round
            # self.__booster_params['max_delta_step'] = max_delta_step
            self.__booster_params['lambda_l1'] = lambda_l1
            self.__booster_params['lambda_l2'] = lambda_l2
            self.__booster_params['min_gain_to_split'] = min_gain_to_split
            self.__booster_params['drop_rate'] = drop_rate
            self.__booster_params['max_drop'] = max_drop
            self.__booster_params['skip_drop'] = skip_drop
            self.__booster_params['xgboost_dart_mode'] = xgboost_dart_mode
            self.__booster_params['uniform_drop'] = uniform_drop
            self.__booster_params['drop_seed'] = drop_seed
            self.__booster_params['top_rate'] = top_rate
            self.__booster_params['other_rate'] = other_rate
            # self.__booster_params['min_data_per_group'] = min_data_per_group
            # self.__booster_params['max_cat_threshold'] = max_cat_threshold
            # self.__booster_params['cat_l2'] = cat_l2
            # self.__booster_params['cat_smooth'] = cat_smooth
            # self.__booster_params['max_cat_to_onehot'] = max_cat_to_onehot
            self.__booster_params['top_k'] = top_k
            # self.__booster_params['monotone_constraints'] = monotone_constraints # Got an error saying unexpected param?
            # self.__booster_params['feature_contri'] = feature_contri # Got an error saying unexpected param?
            self.__booster_params['forcedsplits_filename'] = forcedsplits_filename
            self.__booster_params['verbose'] = verbose
            self.__booster_params['max_bin'] = max_bin
            self.__booster_params['min_data_in_bin'] = min_data_in_bin
            self.__booster_params['subsample_for_bin'] = subsample_for_bin
            self.__booster_params['histogram_pool_size'] = histogram_pool_size
            self.__booster_params['data_random_seed'] = data_random_seed
            self.__booster_params['is_sparse'] = is_sparse
            self.__booster_params['sparse_threshold'] = sparse_threshold
            self.__booster_params['use_missing'] = use_missing
            self.__booster_params['two_round'] = two_round
            self.__booster_params['save_binary'] = save_binary
            self.__booster_params['enable_load_from_binary_file'] = enable_load_from_binary_file
            self.__booster_params['categorical_feature'] = categorical_feature
            self.__booster_params['num_class'] = num_class
            self.__booster_params['is_unbalance'] = is_unbalanced
            self.__booster_params['metric'] = metric
            self.__booster_params['metric_freq'] = metric_freq
            self.__booster_params['num_machines'] = num_machines
            self.__booster_params['gpu_platform_id'] = gpu_platform_id
            self.__booster_params['gpu_device_id'] = gpu_device_id
            self.__booster_params['gpu_use_dp'] = gpu_use_dp

        # For Bayesian optimization.
        self.__bayes_opt_X = None
        self.__bayes_opt_y = None

    # truncated.
    def __create_log(self):

        self.__log_stream = io.StringIO()
        logging.basicConfig(stream=self.__log_stream, level=logging.INFO)

    def __init_logging(self, log_dir='./LightGBM_ToolBox_log/', timestamp_format='%Y-%m-%d-%H-%M-%S'):
        try:
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

            _now = dt.now().strftime(timestamp_format)
            log_file = 'LightGBM_ToolBox_log_{}.txt'.format(_now)
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

    # truncated.
    def get_log(self):
        return self.__log_stream.getvalue()

    def __set_lgb_default_objectives(self):
        self.__lgb_default_objectives = (
            'regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie', 'binary',
            'multiclass', 'multiclassova', 'xentropy', 'xentlambda', 'lambdarank', 'regression_l2', 'mean_squared_error',
            'mse', 'l2_root', 'root_mean_squared_error', 'rmse', 'mean_absolute_error', 'mae',
            'mean_absolute_percentage_error', 'softmax', 'multiclass_ova', 'ova', 'ovr', 'cross_entropy',
            'cross_entropy_lambda'
        )

    def __set_lgb_default_metrics(self):
        self.__lgb_default_metrics = (
            'l1', 'mean_absolute_error', 'mae', 'regression_l1', 'l2', 'mean_squared_error', 'mse', 'regression_l2',
            'regression', 'l2_root', 'root_mean_squared_error', 'rmse', 'quantile', 'mape',
            'mean_absolute_percentage_error', 'huber', 'fair', 'poisson', 'gamma', 'gamma_deviance',
            'tweedie', 'ndcg', 'map', 'auc', 'binary_logloss', 'binary_error', 'multi_logloss', 'multiclass',
            'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr', 'multi_error', 'xentropy', 'cross_entropy',
            'xentlambda', 'cross_entropy_lambda', 'kldiv', 'kullback_leibler'
        )

    @property
    def lgb_default_objectives(self):
        return self.__lgb_default_objectives

    @lgb_default_objectives.setter
    def lgb_default_objectives(self, *args, **kwargs):
        raise ValueError('Default lgb objectives cannot be changed.')

    @property
    def objective(self):
        return self.__booster_params['objective']

    @objective.setter
    def objective(self, objective):
        if isinstance(objective, str):
            assert objective in self.__lgb_default_objectives, 'Illegal input objective, it can only be within:\n{}'.format(
                self.__lgb_default_objectives
            )
            self.__booster_params['objective'] = objective
            logging.info('Updated class objective to {}.'.format(objective))
        else:
            raise ValueError('Objective can only be in {}'.format(self.__lgb_default_objectives))

    @property
    def metrics(self):
        return self.__booster_params['metric']

    @metrics.setter
    def metrics(self, metrics):
        for _metric in metrics:
            assert _metric in self.__lgb_default_metrics, 'Illegal input metric, it can only be within:\n{}'.format(
                self.__lgb_default_metrics
            )
            self.__booster_params['metric'] = metrics
            logging.info('Updated class metrics to {}'.format(metrics))

    @property
    def bayes_opt_X(self):
        return self.__bayes_opt_X

    @bayes_opt_X.setter
    def bayes_opt_X(self, data):
        if isinstance(data, pd.DataFrame):
            self.__bayes_opt_X = data.values
        elif isinstance(data, np.ndarray):
            self.__bayes_opt_X = data
        else:
            raise TypeError('Data used for Bayesian optimization can only be pd.DataFrame or np.ndarray.')

    @property
    def bayes_opt_y(self):
        return self.__bayes_opt_y

    @bayes_opt_y.setter
    def bayes_opt_y(self, data):
        if isinstance(data, pd.DataFrame):
            self.__bayes_opt_y = data.values.squeeze()
        elif isinstance(data, np.ndarray):
            self.__bayes_opt_y = data.squeeze()
        else:
            raise TypeError('Data used for Bayesian optimization can only be pd.DataFrame or np.ndarray.')

    def get_params(self):
        return self.__booster_params

    def __precheck_input_params(self, booster_params):
        try:
            _tmp_booster = Booster(params=booster_params)
            del _tmp_booster
        except Exception as e:
            print('Input hyper-parameters are illegal. Error: {}'.format(e))
            raise

    @classmethod
    def make_dataset(cls,
                     data,
                     label=None,
                     reference=None,
                     weight=None,
                     group=None,
                     # init_score=None, # Got an error?
                     silent=False,
                     feature_name='auto',
                     categorical_feature='auto',
                     params=None,
                     free_raw_data=True,
                     **kwargs
                     ):

        try:
            _output_dataset = Dataset(
                                    data=data,
                                    label=label,
                                    reference=reference,
                                    weight=weight,
                                    group=group,
                                    # init_score=init_score, # Got an error?
                                    silent=silent,
                                    feature_name=feature_name,
                                    categorical_feature=categorical_feature,
                                    params=params,
                                    free_raw_data=free_raw_data
                                    )
            return _output_dataset
        except Exception as e:
            print('Failed in making LightGBM Dataset obj. Error: {}'.format(e))
            raise

    def fit(self,
            train_X,
            train_y,
            valid_X=None,
            valid_y=None,
            params=None,
            num_iterations=100,
            learning_rate=0.1,
            learning_rates=None,
            data_names=None,
            fobj=None,
            feval=None,
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            early_stopping_rounds=100,
            eval_metric=None,
            evals_result=None,
            verbose_eval=100,
            # keep_training_booster=False,
            callbacks=None,
            inplace_class_model=True
            ):
        assert not isinstance(train_X, lgb.Dataset) and not isinstance(valid_X, lgb.Dataset), \
            'Training data cannot be lgb.Dataset.'

        if params != None:
            __booster_params = params
        else:
            # Make a copy of class booster params, then overwrite a few parameters from training API.
            __booster_params = self.__booster_params

        __booster_params['num_iterations'] = num_iterations
        __booster_params['learning_rate'] = learning_rate # Attn: learning_rates should be callable and not a float.

        if eval_metric is not None:
            __booster_params['metrics'] = eval_metric

        if valid_X is None and valid_y is None:
            logging.info('Randomly split training data into 70% to 30% using class random seed.')
            _train_X, _valid_X, _train_y, _valid_y = train_test_split(train_X,
                                                                      train_y,
                                                                      random_state=self.__booster_params['seed'],
                                                                      test_size=0.3)
            logging.info('Training data size: {}, validation data size: {}'.format(_train_X.shape, _valid_X.shape))

            train_dataset = MyLight.make_dataset(data=_train_X, label=_train_y)
            valid_dataset = MyLight.make_dataset(data=_valid_X, label=_valid_y)
            datasets_for_eval = [train_dataset, valid_dataset]
            if data_names is not None:
                assert len(data_names)==2, 'For this case, data names can only have two elements.'
                datasets_names = data_names
            else:
                datasets_names = ['Training', 'Validation']
        else:
            logging.info('Training data size: {}'.format(train_X.shape))
            train_dataset = MyLight.make_dataset(data=train_X, label=train_y)
            datasets_for_eval = [train_dataset]
            if data_names is not None:
                datasets_names = [data_names[0]]
            else:
                datasets_names = ['Training']
            logging.info('Will use validation data sets provided by user.')

            try:
                if isinstance(valid_X, list) and isinstance(valid_y, list):
                    for i in range(len(valid_X)):
                        logging.info('Validation data set {} size: {}'.format(i, valid_X[i].shape))
                        _valid_dataset = MyLight.make_dataset(valid_X[i], valid_y[i])
                        datasets_for_eval.append(_valid_dataset)
                        if data_names is not None:
                            datasets_names.append(data_names[i+1])
                        else:
                            datasets_names.append('Validation_{}'.format(i))
                else:
                    logging.info('Validation data set size: {}'.format(valid_X.shape))
                    _valid_dataset = MyLight.make_dataset(valid_X, valid_y)
                    datasets_for_eval.append(_valid_dataset)
                    if data_names is not None:
                        datasets_names.append(data_names[-1])
                    else:
                        datasets_names.append('Validation')
            except Exception as e:
                print('Failed in making training/ validation data sets. Error: {}'.format(e))
                logging.error('Failed in making training/ validation data sets. Error: {}'.format(e))
                raise

        try:
            logging.info('Booster parameters: {}'.format(__booster_params))
            __trained_model = lgb.train(params=__booster_params,
                                        train_set=train_dataset,
                                        num_boost_round=num_iterations,
                                        valid_sets=datasets_for_eval,
                                        valid_names=datasets_names,
                                        fobj=fobj,
                                        feval=feval,
                                        init_model=init_model,
                                        feature_name=feature_name,
                                        categorical_feature=categorical_feature,
                                        early_stopping_rounds=early_stopping_rounds,
                                        evals_result=evals_result,
                                        verbose_eval=verbose_eval, learning_rates=learning_rates,
                                        #keep_training_booster=keep_training_booster, # Got an error of unexpected param?
                                        callbacks=callbacks
										)
            if inplace_class_model:
                self.trained_model = __trained_model

            return __trained_model
        except Exception as e:
            print('Failed in training a LightGBM model. Error: {}'.format(e))
            logging.error('Failed in training a LightGBM model. Error: {}'.format(e))
            raise

    def analyze_feature_importance(self, trained_model=None, top_features=20, plot=False, **kwargs):

        if trained_model == None:
            _trained_model = self.trained_model
        else:
            _trained_model = trained_model

        try:
            feature_importance = pd.DataFrame()
            feature_importance['Features'] = _trained_model.feature_name()
            feature_importance['Importances'] = _trained_model.feature_importance().squeeze()

            if plot:
                plot_importance(_trained_model, max_num_features=top_features, figsize=(5,10))

            return feature_importance
        except Exception as e:
            print('Failed in calculating feature importance. Error: {}'.format(e))
            raise

    def lgb_cv(self,
               train_X,
               train_y,
               params=None,
               num_iterations=100,
               learning_rate=0.1,
               folds=None,
               nfold=5,
               stratified=False,
               shuffle=True,
               metrics=None,
               fobj=None,
               feval=None,
               init_model=None,
               feature_name='auto',
               categorical_feature='auto',
               early_stopping_rounds=100,
               fpreproc=None,
               verbose_eval=100,
               show_stdv=True,
               booster_random_status=7,
               cv_random_status=7,
               callbacks=None,
               **kwargs
               ):

        # Precheck the input metrics.
        if not isinstance(metrics, list):
            metrics = [metrics]
            for _metric in metrics:
                assert _metric in self.__lgb_default_metrics, 'Illegal input metric(s), can only be within:\n{}'.format(
                    self.__lgb_default_metrics
                )

        try:
            cv_dataset = MyLight.make_dataset(train_X, train_y)

            if params != None:
                __params = params
            else:
                __params = self.__booster_params

            __params['learning_rate'] = learning_rate
            __params['seed'] = booster_random_status

            __cv_results = lgb.cv(params=__params, train_set=cv_dataset, num_boost_round=num_iterations, folds=folds,
                             nfold=nfold, stratified=stratified, shuffle=shuffle, metrics=metrics, fobj=fobj,
                             feval=feval, init_model=init_model, feature_name=feature_name,
                             categorical_feature=categorical_feature, early_stopping_rounds=early_stopping_rounds,
                             fpreproc=fpreproc, verbose_eval=verbose_eval, show_stdv=show_stdv, seed=cv_random_status,
                             callbacks=callbacks
                             )

            logging.info('Executed LGBM CV. Params: {}, Results: {}'.format(__params, __cv_results))
            __cv_results = pd.DataFrame(__cv_results)
            return __cv_results
        except Exception as e:
            print('Failed in cross validation by using user-provided data set. Error: {}'.format(e))
            raise

    def kfolds_cv_ensemble(self,
                        train_X,
                        train_y,
                        booster_params=None,
                        folds=3,
                        shuffle=True,
                        save_models=True,
                        loss_func='rmse',
                        num_iterations=100,
                        learning_rate=0.1,
                        learning_rates=None,
                        data_names=None,
                        fobj=None,
                        feval=None,
                        init_model=None,
                        feature_name='auto',
                        categorical_feature='auto',
                        early_stopping_rounds=100,
                        eval_metric=None,
                        evals_result=None,
                        verbose_eval=100,
                        # keep_training_booster=False,
                        callbacks=None,
                        kfold_random_status=7,
                        booster_random_status=7
                        ):

        kfold = KFold(n_splits=folds, random_state=kfold_random_status, shuffle=shuffle)

        if isinstance(train_X, pd.DataFrame):
            train_X_ = train_X.values
        elif isinstance(train_X, np.ndarray):
            train_X_ = train_X
        else:
            raise TypeError('For this method, only pd.DataFrame and np.ndarray are supported.')

        print('Start K-Fold cross validation (ensemble)...')
        preds = np.zeros(len(train_X_), )
        trained_models = []
        for i, (train_idx, valid_idx) in enumerate(kfold.split(train_X_)):
            print('\nFor fold {}...\n'.format(i+1))
            fold_train_X = train_X_[train_idx]
            fold_valid_X = train_X_[valid_idx]
            fold_train_y = train_y[train_idx]
            fold_valid_y = train_y[valid_idx]

            fold_booster = self.fit(
                train_X=fold_train_X,
                train_y=fold_train_y,
                valid_X=fold_valid_X,
                valid_y=fold_valid_y,
                params=booster_params if booster_params is not None else self.__booster_params,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                learning_rates=learning_rates,
                data_names=data_names if data_names is not None else ['Training', 'Validation'],
                fobj=fobj,
                feval=feval,
                init_model=init_model,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                evals_result=evals_result,
                verbose_eval=verbose_eval,
                callbacks=callbacks
            )

            preds += fold_booster.predict(train_X_)
            # preds[valid_idx] = fold_booster.predict(fold_valid_X)

            if save_models:
                trained_models.append(fold_booster)

        # Average predictions.
        preds /= float(folds)

        if loss_func is 'rmse':
            loss = np.sqrt(mean_squared_error(train_y, preds))

        logging.info('For {}-folds cross validation, the {} is {:.5f}'.format(folds, loss_func, loss))
        print('For {}-folds cross validation, the {} is {:.5f}'.format(folds, loss_func, loss))

        if save_models:
            return trained_models

    def __set_booster_params(self):
        __booster_params = {
                             "boosting": [0, 1],
                             "min_split_gain": [0, 1],
                             "colsample_bytree": [0.7, 1], # alias: feature_fraction.
                             "lambda_l1": [0, 1.5],
                             "num_leaves_shrinkage": [0.55, 0.75],
                             "bagging_freq": [0, 10],
                             "lambda_l2": [0, 1.5],
                             "subsample": [0.5, 1],
                             "max_depth": [5, 9],
                             "min_data_in_leaf": [20, 80],
                             "min_child_weight": [0, 70], # alias: min_sum_hessian_in_leaf.
                             # "learning_rate": [0.001, 0.01]

                             # Use for boosting type of 'dart'.
                             "drop_rate" : [0.0, 0.25],
                             "max_drop" : [20, 50],
                             "skip_drop" : [0.0, 0.95],
                             "xgboost_dart_mode" : [0, 1], # Boolean.
                             "uniform_drop" : [0, 1], # Boolean.
                        }

        return __booster_params

    def __eval_params_using_cv(self,
                               boosting,
                               min_split_gain,
                               colsample_bytree,
                               lambda_l1,
                               num_leaves_shrinkage,
                               bagging_freq,
                               lambda_l2,
                               subsample,
                               max_depth,
                               min_data_in_leaf,
                               min_child_weight,
                               drop_rate,
                               max_drop,
                               skip_drop,
                               xgboost_dart_mode,
                               uniform_drop
                               ):

        assert (self.__bayes_opt_X is not None) and (self.__bayes_opt_y is not None), \
            'For using Bayesian Optimization, please define data for tuning.'

        if boosting <= 0.5:
            __booster_params = dict(
                objective=self.__booster_params['objective'], # Define these two parameters in bayes_tuning.
                metric=self.__booster_params['metric'],
                boosting='gbdt',
                min_split_gain=min_split_gain,
                colsample_bytree=colsample_bytree,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                num_leaves=int(np.round(num_leaves_shrinkage * 2 ** max_depth)),
                bagging_freq=int(np.round(bagging_freq)),
                subsample=subsample,
                max_depth=int(np.round(max_depth)),
                min_data_in_leaf=int(np.round(min_data_in_leaf)),
                min_child_weight=min_child_weight
            )
        else:
            __booster_params = dict(
                objective=self.__booster_params['objective'],
                metric=self.__booster_params['metric'],
                boosting='dart',
                min_split_gain=min_split_gain,
                colsample_bytree=colsample_bytree,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                num_leaves=int(np.round(num_leaves_shrinkage * 2 ** max_depth)),
                bagging_freq=int(np.round(bagging_freq)),
                subsample=subsample,
                max_depth=int(np.round(max_depth)),
                min_data_in_leaf=int(np.round(min_data_in_leaf)),
                min_child_weight=min_child_weight,
                drop_rate=drop_rate,
                max_drop=int(np.round(max_drop)),
                skip_drop=skip_drop,
                xgboost_dart_mode=False if xgboost_dart_mode<0.5 else True,
                uniform_drop=False if uniform_drop<0.5 else True
            )


        __cv_results = self.lgb_cv(train_X=self.__bayes_opt_X, train_y=self.__bayes_opt_y, params=__booster_params,
                                   num_iterations=50000, metrics=self.__booster_params['metric'],
                                   early_stopping_rounds=100, learning_rate=0.03
                                   )

        __cv_results = pd.DataFrame(__cv_results)

        return np.max(__cv_results.iloc[:,0])

    def bayes_tuning(self,
                     params=None,
                     eval_func=None,
                     init_points=5,
                     n_iter=25,
                     acq='ei',
                     xi=0.0,
                     learning_rate=0.03,
                     metric=None,
                     **kwargs):
        '''
        Attn For eval_func, the hyper-parameters have to be the same as stored dict. And adjust the postive/ negative
        return values accordingly by metrics.
        :param learning_rate: float or list.
        # ref: https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
        '''

        if params == None:
            __params = self.__set_booster_params()

        # __params['objective'] = self.__booster_params['objective']
        #
        # if metric == None:
        #     __params['metric'] = self.__booster_params['metric']
        # else:
        #     __params['metric'] = metric

        try:
            ### Test bayes_opt modes:
            # if mode == 'prefer_exploitation':
            #     bo = BayesianOptimization(__eval_func, __params)
            #     bo.maximize(init_point=init_point, n_iter=n_iter, acq='ucb', kappa=1)
            #     return bo
            # if mode == 'prefer_exploration':
            #     bo = BayesianOptimization(__eval_func, __params)
            #     bo.maximize(init_point=init_point, n_iter=n_iter, acq='ucb', kappa=10)

            if eval_func == None:
                bo = BayesianOptimization(self.__eval_params_using_cv, __params)
            else:
                bo = BayesianOptimization(eval_func, __params)

            bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)

            opt_res = pd.DataFrame(bo.res['all']['params'])
            opt_res['values'] = bo.res['all']['values']

            return opt_res
        except Exception as e:
            print('Failed in Bayesian optimization. Error: {}'.format(e))
            raise












































