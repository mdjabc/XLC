# Plan to add:
# Bunch of objective and metrics functions.
# Automatic hyper-parameters tuning.

import io
import os
import sys
import json
import time
import logging
import argparse
import sklearn
import numpy as np
import pandas as pd

from bayes_opt import *
import catboost as cb
from datetime import datetime
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, KFold

class MyCat(object):

    def __init__(self,
                 booster_params=None,
                 objective=None,
                 log_dir=None,
                 checkpoint_dir=None):

        # Record system time to keep log names and other solutions consistent:
        self.start_time = MyCat.tic()

        # Config files dir:
        self.__cb_default_booster_params_file = self.__translate_file_dir('./CatBoost/config/catboost_default_booster_params.json')
        self.__cb_default_loss_func_file = self.__translate_file_dir('./CatBoost/config/catboost_loss_functions.json')

        # Initialize logging.
        self.__init_logging(log_dir=log_dir)
        logging.info('Start logging...')

        # Retrieve default CatBoost loss functions:
        self.__retrieve_cb_default_setting()
        for application in ['classification', 'regression', 'others']:
            # Automatically decide application type by objective / loss function:
            if objective in self.__cb_default_loss_func[application]:
                self.__objective = objective
                self.__loss_function = self.__objective
                self.__application = application
                logging.info('Will train {} models. Loss function: {}'.format(application, self.__objective))
                break
            else:
                self.__application = None
        # Check if objective is incorrect:
        assert self.__application is not None, \
            ValueError('Illegal input objective. CatBoost default objective/ loss functions are: {}'.format(self.__cb_default_loss_func))

        # Define instance booster parameters:
        if booster_params is not None:
            self.__booster_params = booster_params
        else:
            self.__booster_params = self.__cb_default_booster_params[self.__application]

        # Update objective / loss function to booster params:
        self.__booster_params['loss_function'] = self.__loss_function
        logging.info('Update instance booster params to: {}'.format(self.__booster_params))

        # Create dir for autosave model checkpoints:
        self.__create_checkpoints_dir(checkpoint_dir)

        # Placeholders for Bayesian Optimization:
        self.bayes_opt_X = None
        self.bayes_opt_y = None
        self.bayes_opt_metrics = None

        # For automatically save model artifact, if it's not claimed, will use default format. And JSON is more readable:
        self.__model_artifact_format = 'json'

    def __create_checkpoints_dir(self, checkpoint_dir):
        if checkpoint_dir is None:
            checkpoint_dir = self.__translate_file_dir('./CatBoost/CatBoost_model_checkpoints/')

        try:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.__checkpoint_dir = checkpoint_dir
        except Exception as e:
            logging.error('Failed in creating model checkpoint dir. Error: {}'.format(e))
            raise

    @property
    def objective(self):
        return self.__objective

    @objective.setter
    def objective(self, *args, **kwargs):
        MyCat.print_warning('Objective can only be claimed during creating MyCat instances.')

    @property
    def application(self):
        return self.__application

    @application.setter
    def application(self, *args, **kwargs):
        MyCat.print_warning('Application cannot be changed here.')

    @property
    def loss_function(self):
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, *args, **kwargs):
        MyCat.print_warning('Loss function can only be claimed during creating MyCat instances.')

    @classmethod
    def print_warning(cls, warning):
        assert isinstance(warning, str), 'Input warning must be strings.'
        print('*'*len(warning))
        print(warning)
        print('*'*len(warning))

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

        MyCat.print_warning('For {}, Time cost: {:.2f} {}'.format(title.lower(), _lag, _unit))

    @classmethod
    def tic(cls, timestamp_format=None):
        if timestamp_format is None:
            timestamp_format = '%Y-%m-%d-%H-%M-%S'

        try:
            return datetime.now().strftime(timestamp_format)
        except:
            raise ValueError('Timestamp format is wrong.')

    def __merge_two_dict(self, dict1, dict2):
        _output_dict = dict1.copy()
        _output_dict.update(dict2)
        return _output_dict

    # Could change this into a class method though...
    def __translate_file_dir(self, file_dir):
        _abs_dir = os.getcwd()
        return os.path.realpath(os.path.join(_abs_dir, file_dir))

    def __init_logging(self, log_dir=None):
        try:
            if log_dir is None:
                log_dir = self.__translate_file_dir('./CatBoost_logs/')

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

    def __retrieve_cb_default_setting(self):
        try:
            with open(self.__cb_default_booster_params_file, 'r') as booster_params_file:
                self.__cb_default_booster_params = json.load(booster_params_file)
            with open(self.__cb_default_loss_func_file, 'r') as loss_func_file:
                self.__cb_default_loss_func = json.load(loss_func_file)
        except Exception as e:
            error_msg = 'Failed in retrieving CatBoost default parameters and loss functions.' + \
                        'The file should be stored in a child folder ./config/ \n' + \
                        'Error: {}'.format(e)
            print(error_msg)
            logging.error(error_msg)

    @property
    def catboost_default_booster_params(self):
        return self.__cb_default_booster_params

    @catboost_default_booster_params.setter
    def catboost_default_booster_params(self, *args, **kwargs):
        MyCat.print_warning('Warning: Default parameters cannot be changed.')

    @property
    def catboost_default_loss_functions(self):
        return self.__cb_default_loss_func

    @catboost_default_loss_functions.setter
    def catboost_default_loss_function(self, *args, **kwargs):
        MyCat.print_warning('Warning: CatBoost default loss functions cannot be changed.')

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

    def __update_booster_params(self, old_params, new_params):
        updated_params = old_params.copy()
        for _key in new_params.keys():
            updated_params[_key] = new_params[_key]

        return updated_params

    # Too much complexity of the original version...
    @classmethod
    def make_pool(cls,
                  data,
                  label=None,
                  cat_features=None,
                  column_description=None,
                  pairs=None,
                  delimiter='\t', # Truncated this parameter to avoid complexity.
                  has_header=False,
                  weight=None,
                  group_id=None,
                  group_weight=None,
                  subgroup_id=None,
                  pairs_weight=None,
                  baseline=None,
                  feature_names=None,
                  thread_count=-1):

        try:
            pooled_data = cb.Pool(data=data,
                               label=label,
                               cat_features=cat_features,
                               column_description=column_description,
                               pairs=pairs,
                               delimiter=delimiter,
                               has_header=has_header,
                               weight=weight,
                               group_id=group_id,
                               group_weight=group_weight,
                               subgroup_id=subgroup_id,
                               pairs_weight=pairs_weight,
                               baseline=baseline,
                               feature_names=feature_names,
                               thread_count=thread_count)
            return pooled_data
        except Exception as e:
            _error_msg = 'Failed in creating data pool. Error: {}'.format(e)
            print(_error_msg)
            logging.error(_error_msg)
            raise

    def fit(self,
            train_X,
            train_y,
            valid_X=None,
            valid_y=None,
            auto_split_train_data=True,
            booster_params=None,
            num_boost_round=10,
            learning_rate=0.01,
            nthread=-1,
            eval_metric=None,
            # learning_rates=None, # callback...
            cat_features=None,
            pairs=None,
            pretrained_model=None,
            sample_weight=None,
            group_id=None,
            group_weight=None,
            subgroup_id=None,
            pairs_weight=None,
            baseline=None,
            use_best_model=None,
            verbose=None,
            # verbose_eval=None, # Alias as verbose...
            logging_level=None,
            plot=False,
            column_description=None,
            metric_period=None,
            silent=None,
            random_seed=0,
            early_stopping_rounds=100,
            # save_snapshot=None,
            # snapshot_file=None,
            # snapshot_interval=None,
            inplace_class_model=True,
            autosave_ckpt=True,
            ):

        if booster_params is not None:
            __booster_params = booster_params
        else:
            __booster_params = self.__booster_params

        # Update some training parameters:
        __booster_params['iterations'] = num_boost_round
        __booster_params['eta'] = learning_rate
        # __booster_params['eta'] = learning_rate
        __booster_params['random_seed'] = random_seed # Alias: random_state
        __booster_params['thread_count'] = nthread
        __booster_params['eval_metric'] = eval_metric
        __booster_params['logging_level'] = logging_level

        # Split training and validating data to prevent overfitting:
        if valid_X is None and valid_y is None:
            if auto_split_train_data:
                logging.info('Randomly split training data into 70% and 30%.')
                _train_X, _valid_X, _train_y, _valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=random_seed)
                logging.info('Training data size: {}, validation data size: {}'.format(_train_X.shape, _valid_X.shape))

                # Make CatBoost Pool:
                train_pool = MyCat.make_pool(data=_train_X, label=_train_y, cat_features=cat_features, pairs=pairs)
                valid_pool = MyCat.make_pool(data=_valid_X, label=_valid_y, cat_features=cat_features, pairs=pairs)
                data_for_eval = [valid_pool] # [train_pool, valid_pool]
            else:
                train_pool = MyCat.make_pool(data=train_X, label=train_y, cat_features=cat_features, pairs=pairs)
                data_for_eval = None # [train_pool]
        elif valid_X is not None and valid_y is not None:
            if not isinstance(valid_X):
                valid_X = [valid_X]
            if not isinstance(valid_y):
                valid_y = [valid_y]

            assert len(valid_X) == len(valid_y), 'Input valid_X and valid_y should have same length.'

            logging.info('Training data size: {}'.format(train_X.shape[0]))
            train_pool = MyCat.make_pool(data=train_X, label=train_y, cat_features=cat_features, pairs=pairs)
            data_for_eval = []

            for i in range(len(valid_X)):
                _valid_pool = MyCat.make_pool(data=valid_X[i], label=valid_y[i], cat_features=cat_features, pairs=pairs)
                data_for_eval.append(_valid_pool)
                logging.info('Validation data {} size: {}'.format(i, valid_X[i].shape))

        # Start training procedure...
        try:
            _training_execution_time = MyCat.tic()
            if pretrained_model is not None:
                print('Use pretrained model...')
                _this_cat = pretrained_model
                _this_cat.set_params(eta=learning_rate)
            elif self.__application == 'regression':
                _this_cat = cb.CatBoostRegressor(**__booster_params)
            elif self.__application == 'classification':
                _this_cat = cb.CatBoostClassifier(**__booster_params)
            else:
                raise ValueError('Unknown application type. Should be either classification or regression.')

            with MyCat.timer('Model training'):
                _this_cat.fit(
                    X=train_pool,
                    eval_set=data_for_eval,
                    verbose=verbose,
                    plot=plot,
                    early_stopping_rounds=early_stopping_rounds,
                    silent=silent
                )

                if autosave_ckpt:
                    _ckpt_file = 'CatBoost_model_ckpt_{}'.format(_training_execution_time)
                    _ckpt_file = os.path.join(self.__checkpoint_dir, _ckpt_file)
                    _this_cat.save_model(_ckpt_file, self.__model_artifact_format)
                    logging.info('Saved model artifact to {}'.format(_ckpt_file))

                return _this_cat
        except Exception as e:
            _error_msg = 'Failed in training CatBoost model. Error: {}'.format(e)
            print(_error_msg)
            logging.error(_error_msg)
            raise

    def auto_fit(self,
            train_X,
            train_y,
            valid_X=None,
            valid_y=None,
            auto_split_train_data=True,
            booster_params=None,
            num_boost_round=10,
            init_learning_rate=0.01,
            learning_rate_tolerance=0.001,
            higher_better=False,
            nthread=-1,
            eval_metric=None,
            # learning_rates=None, # callback...
            cat_features=None,
            pairs=None,
            sample_weight=None,
            group_id=None,
            group_weight=None,
            subgroup_id=None,
            pairs_weight=None,
            baseline=None,
            use_best_model=None,
            verbose=None,
            # verbose_eval=None, # Alias as verbose...
            logging_level=None,
            plot=False,
            column_description=None,
            metric_period=None,
            silent=None,
            random_seed=0,
            early_stopping_rounds=100,
            # save_snapshot=None,
            # snapshot_file=None,
            # snapshot_interval=None,
            inplace_class_model=True,
            autosave_ckpt=True,
            ):

        best_score = -np.inf if higher_better else np.inf
        best_model = None
        pretrained_model = None
        adjustable_learning_rate = init_learning_rate
        _training_execution_time = MyCat.tic()
        logging.info('Start CatBoost autofit at: {}'.format(_training_execution_time))
        while adjustable_learning_rate >= learning_rate_tolerance:
            _to_update_best_model = False
            _trained_cat = self.fit(
                train_X=train_X,
                train_y=train_y,
                valid_X=valid_X,
                valid_y=valid_y,
                auto_split_train_data=auto_split_train_data,
                booster_params=booster_params,
                num_boost_round=num_boost_round,
                learning_rate=adjustable_learning_rate,
                nthread=nthread,
                eval_metric=eval_metric,
                # learning_rates=None, # callback...
                cat_features=cat_features,
                pairs=pairs,
                pretrained_model=pretrained_model, # After first round, this parameter will be update...
                sample_weight=sample_weight,
                group_id=group_id,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                pairs_weight=pairs_weight,
                baseline=baseline,
                use_best_model=use_best_model,
                verbose=verbose,
                # verbose_eval=None, # Alias as verbose...
                logging_level=logging_level,
                plot=plot,
                column_description=column_description,
                metric_period=metric_period,
                silent=silent,
                random_seed=random_seed,
                early_stopping_rounds=early_stopping_rounds,
                # save_snapshot=None,
                # snapshot_file=None,
                # snapshot_interval=None,
                inplace_class_model=inplace_class_model,
                autosave_ckpt=autosave_ckpt,
            )

            _model_best_score = pd.DataFrame(_trained_cat.best_score_).iloc[0, 1:].mean()
            if higher_better:
                if _model_best_score > best_score:
                    best_score = _model_best_score
                    best_model = _trained_cat
                    _to_update_best_model = True
            else:
                if _model_best_score < best_score:
                    best_score = _model_best_score
                    best_model = _trained_cat
                    _to_update_best_model = True

            # Feed back to fit method:
            pretrained_model = best_model

            # Save model to checkpoint:
            if _to_update_best_model:
                print('For learning rate of {}, best score is {}'.format(adjustable_learning_rate, best_score))
                _ckpt_file = 'CatBoost_AutoSearch_model_ckpt_{}'.format(_training_execution_time)
                _ckpt_file = os.path.join(self.__checkpoint_dir, _ckpt_file)
                # xgb_model = _ckpt_file
                _trained_cat.save_model(_ckpt_file, self.__model_artifact_format)
                logging.info('Save model artifact {}. Best score: {}. Learning rate: {}.'.format(_ckpt_file, \
                                                                                                 best_score, \
                                                                                                 adjustable_learning_rate))

            # Update learning rate:
            adjustable_learning_rate *= 0.9

        return best_model






