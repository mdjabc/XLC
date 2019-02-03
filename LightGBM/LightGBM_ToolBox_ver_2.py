# Official Documents of LightGBM:
# https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#config

# Difference between cross-entropy and binary classification:
# https://github.com/Microsoft/LightGBM/issues/1348

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

# is_unbalanced for binary clf.

# Support multiple metrics.

# Tweak hyperparameter tweaking part: gbdt, rf, goss, dart
# Autosearch? Model load back and decreased learning rate?

import io
import os
import sys
import json
import time
import logging
import argparse
import sklearn
# import numba
import numpy as np
import pandas as pd

from bayes_opt import *
import lightgbm as lgb
from datetime import datetime
from contextlib import contextmanager
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from .config.lgb_booster_core_params import *
from .config.lgb_default_settings import *
from .utils.lgb_customized_functions import *
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

class MyLight(object):
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
                booster_params=None,
                objective=None,
                log_dir=None,
                checkpoint_dir=None
        ):

        # Record system time to keep log names and other solutions consistent:
        self.start_time = MyLight.tic()

        # Truncated:
        # # Config files dir:
        # self.__lgb_booster_core_params_file = self.__translate_file_dir('./LightGBM/config/lgb_booster_core_params.json')
        # self.__lgb_default_settings_file = self.__translate_file_dir('./LightGBM/config/lightgbm_default_settings.json')

        # Initialize logging.
        self.__init_logging(log_dir=log_dir)
        logging.info('Start logging...')

        # Create a tuple to store default LGBM objectives and precheck user-defined objective.
        self.__retrieve_lgb_default_setting()
        assert objective in self.__lgb_default_objectives, 'Illegal input objective, it can only be within:\n{}'.format(
            self.__lgb_default_objectives
        )

        self.__booster_params = booster_params if booster_params is not None else self.__lgb_default_booster_params
        self.__objective = objective
        self.__booster_params['objective'] = self.__objective
        logging.info('Update instance booster params to: {}'.format(self.__booster_params))

        # Check booster parameters:
        # pass

        # Create dir for autosave model checkpoints:
        self.__create_checkpoints_dir(checkpoint_dir)

        # For Bayesian optimization.
        self.__bayes_opt_X = None
        self.__bayes_opt_y = None
        self.__bayes_opt_metrics = None

    def __create_checkpoints_dir(self, checkpoint_dir):
        if checkpoint_dir is None:
            checkpoint_dir = self.__translate_file_dir('./LightGBM/LightGBM_model_checkpoints/')

        try:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.__checkpoint_dir = checkpoint_dir
        except Exception as e:
            logging.error('Failed in creating model checkpoint dir. Error: {}'.format(e))
            raise

    # May need to truncate this for future:
    def __translate_file_dir(self, file_dir):
        _abs_dir = os.getcwd()
        return os.path.realpath(os.path.join(_abs_dir, file_dir))

    def __init_logging(self, log_dir=None):
        try:
            if log_dir is None:
                log_dir = self.__translate_file_dir('./LightGBM/LightGBM_logs/')

            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

            log_file = 'LightGBM_ToolBox_log_{}.txt'.format(self.start_time)
            self.log_file_dir = log_dir + log_file
            logging.basicConfig(
                filename=self.log_file_dir,
                level=logging.INFO,
                format='[%(asctime)s (Local Time)] %(levelname)s : %(message)s', # Local time may vary for cloud services.
                datefmt='%m/%d/%Y %I:%M:%S %p'
            )
        except Exception as e:
            print('Failed in creating log. Error: {}'.format(e))
            raise

    def get_log(self, output_log=False):
        """
        Retrive running log.
        :return:
        """
        try:
            with open(self.log_file_dir, 'r') as log_file:
                _log = log_file.read()
                print(_log)

            if output_log:
                return _log
        except Exception as e:
            print("Failed in getting content of log file {}. Error: {}".format(self.log_file_dir, e))

    def __merge_two_dict(self, dict1, dict2):
        output_dict = dict1.copy()
        output_dict.update(dict2)
        return output_dict

    # Move to a json file.
    def __retrieve_lgb_default_setting(self):

        try:
            # Truncated:
            # # Read default booster params:
            # with open(self.__lgb_booster_core_params_file, 'r') as booster_params_file:
            #     self.__lgb_default_booster_params = json.load(booster_params_file)
            # Read default objectives and metrics:
            self.__lgb_default_booster_params = lgb_default_core_params
            
            # Truncated:
            # with open(self.__lgb_default_settings_file, 'r') as file:
                # self.__lgb_default_settings = json.load(file)
            
            self.__lgb_default_settings = lgb_default_settings
            self.__lgb_reg_objectives = self.__lgb_default_settings['regression-objectives']
            self.__lgb_binary_objectives = self.__lgb_default_settings['binary-objectives']
            self.__lgb_multiclass_objectives = self.__lgb_default_settings['multi-class-objectives']
            self.__lgb_cross_entropy_objectives = self.__lgb_default_settings['cross-entropy-objectives']
            self.__lgb_lambdarank_objectives = self.__lgb_default_settings['lambdarank-objectives']
            self.__lgb_default_metrics = self.__lgb_default_settings['default-metrics']
            # Make a full list of default objectives:
            self.__lgb_default_objectives = self.__lgb_reg_objectives + self.__lgb_binary_objectives + \
                                            self.__lgb_multiclass_objectives + self.__lgb_cross_entropy_objectives + \
                                            self.__lgb_lambdarank_objectives
            self.__lgb_clf_objectives = self.__lgb_binary_objectives + self.__lgb_multiclass_objectives + \
                                        self.__lgb_cross_entropy_objectives
        except Exception as e:
            error_msg = 'Failed in retrieving LightGBM default parameters, objectives and metrics. ' + \
                        'The file should be stored in a child folder ./config/ \n' + \
                        'Error: {}'.format(e)
            print(error_msg)
            logging.error(error_msg)


    # Move to a json file.
    # def __set_lgb_default_metrics(self):
    #     self.__lgb_default_metrics = (
    #         'l1', 'mean_absolute_error', 'mae', 'regression_l1', 'l2', 'mean_squared_error', 'mse', 'regression_l2',
    #         'regression', 'l2_root', 'root_mean_squared_error', 'rmse', 'quantile', 'mape',
    #         'mean_absolute_percentage_error', 'huber', 'fair', 'poisson', 'gamma', 'gamma_deviance',
    #         'tweedie', 'ndcg', 'map', 'auc', 'binary_logloss', 'binary_error', 'multi_logloss', 'multiclass',
    #         'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr', 'multi_error', 'xentropy', 'cross_entropy',
    #         'xentlambda', 'cross_entropy_lambda', 'kldiv', 'kullback_leibler'
    #     )

    @property
    def lgb_default_objectives(self):
        return self.__lgb_default_objectives

    @lgb_default_objectives.setter
    def lgb_default_objectives(self, *args, **kwargs):
        print('Warning: Default objectives cannot be changed.')

    @property
    def lgb_default_metrics(self):
        return self.__lgb_default_metrics

    @lgb_default_metrics.setter
    def lgb_default_metrics(self, *args, **kwargs):
        print('Warning: Default metrics cannot be changed.')

    @property
    def objective(self):
        return self.__objective

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

    @classmethod
    def print_warning(cls, warning):
        assert isinstance(warning, str), 'Input warning must be strings.'
        print('*'*len(warning))
        print(warning)
        print('*'*len(warning))

    @property
    def application(self):
        try:
            return self.__application
        except:
            MyLight.print_warning('Application is unknown.')

    @application.setter
    def application(self, *args, **kwargs):
        MyLight.print_warning('Application cannot be changed.')

    @property
    def booster_params(self):
        return self.__booster_params

    @booster_params.setter
    def booster_params(self, **kwargs):
        for _key in kwargs:
            assert _key in self.__booster_params, '{} is not valid in booster params.'.format(_key)

        for _key in kwargs:
            logging.info('Updated booster params {} from {} to {}'.format(_key, self.__booster_params[_key], kwargs[_key]))
            self.__booster_params[_key] = kwargs[_key]

    # Truncated. Metric will only be needed during training:
    # @property
    # def metrics(self):
    #     return self.__booster_params['metric']
    #
    # @metrics.setter
    # def metrics(self, metrics):
    #     for _metric in metrics:
    #         assert _metric in self.__lgb_default_metrics, 'Illegal input metric, it can only be within:\n{}'.format(
    #             self.__lgb_default_metrics
    #         )
    #         self.__booster_params['metric'] = metrics
    #         logging.info('Updated class metrics to {}'.format(metrics))

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
    
    @property
    def bayes_opt_metrics(self):
        return self.__bayes_opt_metrics

    @bayes_opt_metrics.setter
    def bayes_opt_metrics(self, metrics):
        self.__bayes_opt_metrics = metrics

    def get_booster_params(self):
        return self.__booster_params

    def __precheck_input_params(self, booster_params):
        try:
            _tmp_booster = lgb.Booster(params=booster_params, train_set=MyLight.make_dataset([1.0], [1.0]))
            del _tmp_booster
        except Exception as e:
            print('Input hyper-parameters are illegal. Error: {}'.format(e))
            raise

    @classmethod
    def tic(cls, timestamp_format=None):
        if timestamp_format is None:
            timestamp_format = '%Y-%m-%d-%H-%M-%S'

        try:
            return datetime.now().strftime(timestamp_format)
        except:
            raise ValueError('Timestamp format is wrong.')

    @classmethod
    @contextmanager
    def timer(cls, title=None):
        assert isinstance(title, str), 'Title can only be strings.'
        print(title)

        _tic = time.time()
        yield
        _lag = time.time() - _tic

        if _lag <= 60:
            _unit = 'second(s)'
        elif (_lag > 60) & (_lag <= 3600):
            _unit = 'minute(s)'
            _lag /= 60
        else:
            _unit = 'hour(s)'
            _lag /= 3600

        MyLight.print_warning('For {}, Time cost: {:.2f} {}'.format(title.lower(), _lag, _unit))

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
            _output_dataset = lgb.Dataset(
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
            _error_msg = 'Failed in making LightGBM Dataset obj. Error: {}'.format(e)
            print(_error_msg)
            logging.error(_error_msg)
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
            learning_rate_decay_rate=None,
            data_names=None,
            fobj=None,
            feval=None,
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            early_stopping_rounds=100,
            random_seed=7,
            n_jobs=-1,
            eval_metric=None,
            evals_result=None,
            verbose_eval=100,
            # keep_training_booster=False,
            callbacks=None,
            inplace_class_model=True,
            autosave_ckpt=True,
            analyze_feature_importance=False
            ):
        assert not isinstance(train_X, lgb.Dataset) and not isinstance(valid_X, lgb.Dataset), \
            'Training data cannot be lgb.Dataset.'

        if params != None:
            __booster_params = params
        else:
            # Make a copy of class booster params, then overwrite a few parameters from training API.
            __booster_params = self.__booster_params

        # __booster_params['objective'] = self.objective
        __booster_params['num_iterations'] = num_iterations
        __booster_params['learning_rate'] = learning_rate # Attn: learning_rates should be callable and not a float.
        __booster_params['seed'] = random_seed
        __booster_params['n_jobs'] = n_jobs

        # Define adjustable learning rate:
        if learning_rates is not None:
            learning_rates = learning_rates
        else:
            if learning_rate_decay_rate is not None:
                learning_rates = lambda iter: learning_rate * (learning_rate_decay_rate ** iter)
        _msg = 'Initial learning rate: {learning_rate}, adjustable learning rates: {learning_rates}'.format(
            learning_rate=learning_rate,
            learning_rates=learning_rates
        )
        logging.info(_msg)

        # After updating booster params, check if it's valid:
        # assert self.__precheck_input_params(__booster_params), \
        #     logging.error('Incorrect booster params: {}'.format(__booster_params))

        if eval_metric is not None:
            __booster_params['metrics'] = eval_metric

        if valid_X is None and valid_y is None:
            logging.info('Randomly split training data into 70% to 30%.')
            _train_X, _valid_X, _train_y, _valid_y = train_test_split(train_X,
                                                                      train_y,
                                                                      random_state=random_seed,
                                                                      test_size=0.3)
            logging.info('Training data size: {}, validation data size: {}'.format(_train_X.shape, _valid_X.shape))

            train_dataset = MyLight.make_dataset(data=_train_X, label=_train_y, categorical_feature=categorical_feature)
            valid_dataset = MyLight.make_dataset(data=_valid_X, label=_valid_y, categorical_feature=categorical_feature)
            datasets_for_eval = [train_dataset, valid_dataset]
            # Override user provided data set names:
            datasets_names = ['Training', 'Validation']
        else:
            logging.info('Training data size: {}'.format(train_X.shape))
            train_dataset = MyLight.make_dataset(data=train_X, label=train_y, categorical_feature=categorical_feature)
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
                        _valid_dataset = MyLight.make_dataset(valid_X[i], valid_y[i], \
                                                              categorical_feature=categorical_feature)
                        datasets_for_eval.append(_valid_dataset)
                        if data_names is not None:
                            datasets_names.append(data_names[i+1])
                        else:
                            datasets_names.append('Validation_{}'.format(i))
                else:
                    logging.info('Validation data set size: {}'.format(valid_X.shape))
                    _valid_dataset = MyLight.make_dataset(valid_X, valid_y, categorical_feature=categorical_feature)
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
            _training_execution_time = MyLight.tic()
            logging.info('Run training at: {}'.format(_training_execution_time))
            with MyLight.timer('model training'):
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
                                            verbose_eval=verbose_eval,
                                            learning_rates=learning_rates,
                                            #keep_training_booster=keep_training_booster, # Got an error of unexpected param?
                                            callbacks=callbacks
                                            )
                # If inplace class model:
                if inplace_class_model:
                    self.trained_model = __trained_model

                # If auto-save:
                if autosave_ckpt:
                    _ckpt_file = 'LightGBM_model_ckpt_{}'.format(_training_execution_time)
                    _ckpt_file = os.path.join(self.__checkpoint_dir, _ckpt_file)
                    __trained_model.save_model(_ckpt_file)
                    logging.info('Save model artifact {}'.format(_ckpt_file))

                if analyze_feature_importance:
                    feature_importance = self.analyze_feature_importance(__trained_model)
                    return __trained_model, feature_importance
                else:
                    return __trained_model
        except Exception as e:
            print('Failed in training a LightGBM model. Error: {}'.format(e))
            logging.error('Failed in training a LightGBM model. Error: {}'.format(e))
            raise

    def analyze_feature_importance(self, trained_model=None, top_features=20, **kwargs):

        if trained_model == None:
            _trained_model = self.trained_model
        else:
            _trained_model = trained_model

        try:
            feature_importance = pd.DataFrame()
            feature_importance['Features'] = _trained_model.feature_name()
            feature_importance['Importances'] = _trained_model.feature_importance().squeeze()

            # Truncated:
            # if plot:
            #     plot_importance(_trained_model, max_num_features=top_features, figsize=(5,10))

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
            _feature_name = train_X.columns.tolist()
            print("Retrieved feature names from cross-validation data set.")
        except Exception as e:
            logging.info("Cannot get feature names from training data. Reason: {}".format(e))

        try:
            cv_dataset = MyLight.make_dataset(train_X, train_y, \
                                              categorical_feature=categorical_feature, feature_name=_feature_name)

            if params != None:
                __params = params
            else:
                __params = self.__lgb_default_booster_params

            __params['num_iterations'] = num_iterations
            __params['learning_rate'] = learning_rate
            __params['seed'] = booster_random_status

            # Start cross-validation and measure time cost:
            with MyLight.timer('LGB default cross-validation tool'):
                __cv_results = lgb.cv(params=__params, train_set=cv_dataset, num_boost_round=num_iterations, folds=folds,
                                 nfold=nfold, stratified=stratified, shuffle=shuffle, metrics=metrics, fobj=fobj,
                                 feval=feval, init_model=init_model, feature_name=feature_name,
                                 categorical_feature=categorical_feature, early_stopping_rounds=early_stopping_rounds,
                                 fpreproc=fpreproc, verbose_eval=verbose_eval, show_stdv=show_stdv, seed=cv_random_status,
                                 callbacks=callbacks)

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

        _msg = 'Start K-Fold cross validation (ensemble)...'
        print(_msg)
        logging.info(_msg)

        preds = np.zeros(len(train_X_), )
        trained_models = []
        for i, (train_idx, valid_idx) in enumerate(kfold.split(train_X_)):
            _msg = '\nFor fold {}...\n'.format(i+1)
            print(_msg)
            logging.info(_msg)

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
                data_names=['Training', 'Validation'],
                fobj=fobj,
                feval=feval,
                init_model=init_model,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                evals_result=evals_result,
                verbose_eval=verbose_eval,
                callbacks=callbacks,
                autosave_ckpt=True
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
            return trained_models, preds
        else:
            return preds

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
                metric=self.__bayes_opt_metrics,
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
                metric=self.__bayes_opt_metrics,
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
                                   num_iterations=50000, metrics=self.__bayes_opt_metrics,
                                   early_stopping_rounds=100, learning_rate=0.03
                                   )

        __cv_results = pd.DataFrame(__cv_results)

        return np.max(__cv_results.iloc[:,0])

    def bayes_tuning(self,
                     params=None,
                     eval_func=None,
                     init_points=5,
                     n_iter=25,
                     mode='ei',
                     # acq='ei',
                     # xi=0.0,
                     learning_rate=0.03,
                     metric=None,
                     random_state=7,
                     **kwargs):
        '''
        Attn For eval_func, the hyper-parameters have to be the same as stored dict. And adjust the postive/ negative
        return values accordingly by metrics.
        :param learning_rate: float or list.
        ref: https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
        kwargs for Bayesian optimizer:
        :param acq = 'ei'
        :param xi=0.0
        :param kappa
        '''

        if params == None:
            __params = self.__set_booster_params()

        bayesian_optimizer_params = kwargs if kwargs is not None else {}

        if mode == 'exploration':
            bayesian_optimizer_params['acq'] = 'ucb'
            bayesian_optimizer_params['kappa'] = 10
        elif mode == 'ei' : # Expected Improvement
            bayesian_optimizer_params['acq'] = 'ei'
            bayesian_optimizer_params['xi'] = 1e-4

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
                bo = BayesianOptimization(self.__eval_params_using_cv, __params, random_state=random_state)
            else:
                bo = BayesianOptimization(eval_func, __params, random_state=random_state)

            logger = JSONLogger(path=self.__translate_file_dir('BayesianOptimization_log.json'))
            bo.subscribe(Events.OPTMIZATION_STEP, logger)

            bo.maximize(init_points=init_points, n_iter=n_iter, **kwargs)

            opt_res = pd.DataFrame(bo.res['all']['params'])
            opt_res['values'] = bo.res['all']['values']

            return opt_res
        except Exception as e:
            print('Failed in Bayesian optimization. Error: {}'.format(e))
            raise












































