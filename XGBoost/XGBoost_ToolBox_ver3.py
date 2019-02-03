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
import xgboost as xgb
from customized_obj_metric import *
from datetime import datetime
from contextlib import contextmanager

from sklearn.model_selection import train_test_split, KFold

class MyXGB(object):

    def __init__(self,
                booster_params=None,
                objective=None,
                log_dir=None,
                checkpoint_dir=None):

        # Record system time to keep log names and other solutions consistent:
        self.start_time = MyXGB.tic()

        # Config files dir:
        self.__xgb_gbtree_core_params_file = self.__translate_file_dir('./XGBoost/config/xgb_default_gbtree_booster_params.json')
        self.__xgb_dart_core_params_file = self.__translate_file_dir('./XGBoost/config/xgb_default_dart_booster_params.json')
        self.__xgb_default_objectives_file = self.__translate_file_dir('./XGBoost/config/xgb_default_objectives.json')
        self.__xgb_default_metrics_file = self.__translate_file_dir('./XGBoost/config/xgb_default_metrics.json')

        # Initialize logging.
        self.__init_logging(log_dir=log_dir)
        logging.info('Start logging...')

        # Create a tuple to store default XGB objectives and precheck user-defined objective.
        self.__retrieve_xgb_default_setting()
        assert objective in self.__xgb_default_objectives, 'Illegal input objective, it can only be within:\n{}'.format(
            self.__xgb_default_objectives
        )

        self.__booster_params = booster_params if booster_params is not None else self.__xgb_gbtree_params
        self.__objective = objective
        self.__booster_params['objective'] = self.__objective
        logging.info('Update instance booster params to: {}'.format(self.__booster_params))

        # Check booster parameters:
        # pass

        # Create dir for autosave model checkpoints:
        self.__create_checkpoints_dir(checkpoint_dir)

        # For Bayesian optimization.
        self.bayes_opt_X = None
        self.bayes_opt_y = None
        self.bayes_opt_metrics = None
        self.bayes_is_loss = None

    def __create_checkpoints_dir(self, checkpoint_dir):
        if checkpoint_dir is None:
            checkpoint_dir = self.__translate_file_dir('./XGBoost/XGBoost_model_checkpoints/')

        try:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.__checkpoint_dir = checkpoint_dir
        except Exception as e:
            logging.error('Failed in creating model checkpoint dir. Error: {}'.format(e))
            raise

    def __translate_file_dir(self, file_dir):
        _abs_dir = os.getcwd()
        return os.path.realpath(os.path.join(_abs_dir, file_dir))

    def __init_logging(self, log_dir=None):
        try:
            if log_dir is None:
                log_dir = self.__translate_file_dir('./XGBoost_logs/')

            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

            log_file = 'XGBoost_ToolBox_log_{}.txt'.format(self.start_time)
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

    @classmethod
    def tic(cls, timestamp_format=None):
        if timestamp_format is None:
            timestamp_format = '%Y-%m-%d-%H-%M-%S'

        try:
            return datetime.now().strftime(timestamp_format)
        except:
            raise ValueError('Timestamp format is wrong.')

    def __retrieve_xgb_default_setting(self):
        try:
            with open(self.__xgb_gbtree_core_params_file, 'r') as gbtree_params_file:
                self.__xgb_gbtree_params = json.load(gbtree_params_file)
            with open(self.__xgb_dart_core_params_file, 'r') as dart_params_file:
                self.__xgb_dart_params = json.load(dart_params_file)
            with open(self.__xgb_default_objectives_file, 'r') as objectives_file:
                self.__xgb_default_objectives = json.load(objectives_file)
            with open(self.__xgb_default_metrics_file, 'r') as metrics_file:
                self.__xgb_default_metrics = json.load(metrics_file)
        except Exception as e:
            error_msg = 'Failed in retrieving XGBoost default parameters, objectives and metrics. ' + \
                        'The file should be stored in a child folder ./config/ \n' + \
                        'Error: {}'.format(e)
            print(error_msg)
            logging.error(error_msg)

    @property
    def xgb_default_objectives(self):
        return self.__xgb_default_objectives

    @xgb_default_objectives.setter
    def xgb_default_objectives(self, *args, **kwargs):
        print('Warning: Default objectives cannot be changed.')

    @property
    def xgb_default_metrics(self):
        return self.__xgb_default_metrics

    @xgb_default_metrics.setter
    def xgb_default_metrics(self, *args, **kwargs):
        print('Warning: Default metrics cannot be changed.')

    @property
    def objective(self):
        return self.__objective

    @objective.setter
    def objective(self, objective):
        assert objective in self.__xgb_default_objectives, \
            ValueError('Illegal XGBoost objective. It can only be from: {}'.format(self.__xgb_default_objectives))
        self.__objective = objective
        self.__booster_params['objective'] = objective
        logging.info('Updated instance objective to {}'.format(objective))

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

        MyXGB.print_warning('For {}, Time cost: {:.2f} {}'.format(title.lower(), _lag, _unit))

    @classmethod
    def make_dmatrix(cls,
                     data,
                     label=None,
                     missing=None,
                     weight=None,
                     silent=False,
                     feature_names=None,
                     feature_types=None,
                     nthread=-1
                     ):
        '''
        data (string/numpy array/scipy.sparse/pd.DataFrame/DataTable) – Data source of DMatrix. When data is string type, it represents the path libsvm format txt file, or binary file that xgboost can read from.
        label (list or numpy 1-D array, optional) – Label of the training data.
        missing (float, optional) – Value in the data which needs to be present as a missing value. If None, defaults to np.nan.
        weight (list or numpy 1-D array , optional) – Weight for each instance.
        silent (boolean, optional) – Whether print messages during construction
        feature_names (list, optional) – Set names for features.
        feature_types (list, optional) – Set types for features.
        nthread (integer, optional) – Number of threads to use for loading data from numpy array. If -1, uses maximum threads available on the system.
        '''
        try:
            _output_dmatrix = xgb.DMatrix(
                                    data=data,
                                    label=label,
                                    missing=missing,
                                    weight=weight,
                                    silent=silent,
                                    feature_names=feature_names,
                                    feature_types=feature_types,
                                    nthread=nthread)
            return _output_dmatrix
        except Exception as e:
            _error_msg = 'Failed in making XGBoost DMatrix obj. Error: {}'.format(e)
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
            obj=None,
            eval_metric=None,
            feval=None,
            learning_rate=0.1,
            maximize=False,
            early_stopping_rounds=None,
            evals_result=None,
            xgb_model=None,
            callbacks=None,
            nthread=-1,
            verbose_eval=100,
            inplace_class_model=True,
            autosave_ckpt=True,
            random_seed=7,
            silent=True
            ):
        '''
        For learning_rates:
            list l: eta = l[boosting_round]
            function f: eta = f(boosting_round, num_boost_round)
        '''

        if booster_params is not None:
            __booster_params = booster_params
        else:
            __booster_params = self.__booster_params

        # Update booster params before training:
        __booster_params['learning_rate'] = learning_rate
        __booster_params['eval_metric'] = eval_metric
        __booster_params['nthread'] = nthread
        __booster_params['silent'] = silent

        if valid_X is None and valid_y is None:
            if auto_split_train_data:
                logging.info('Randomly split training data into 70% and 30%.')
                _train_X, _valid_X, _train_y, _valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=random_seed)
                logging.info('Training data size: {}, validation data size: {}'.format(_train_X.shape, _valid_X.shape))

                train_dmatrix = MyXGB.make_dmatrix(data=_train_X, label=_train_y)
                valid_dmatrix = MyXGB.make_dmatrix(data=_valid_X, label=_valid_y)
                data_for_eval = [[train_dmatrix, 'Train'], [valid_dmatrix, 'Valid']]
            else:
                train_dmatrix = MyXGB.make_dmatrix(data=train_X, label=train_y)
                data_for_eval = [[train_dmatrix, 'Train']]
        elif valid_X is not None and valid_y is not None:
            if not isinstance(valid_X):
                valid_X = [valid_X]
            if not isinstance(valid_y):
                valid_y = [valid_y]

            assert len(valid_X) == len(valid_y), 'Input valid_X and valid_y should have same length.'

            logging.info('Training data size: {}'.format(train_X.shape[0]))
            train_dmatrix = MyXGB.make_dmatrix(data=train_X, label=train_y)
            data_for_eval = [[train_dmatrix, 'Train']]

            for i in range(len(valid_X)):
                _valid_dmatrix = MyXGB.make_dmatrix(data=valid_X[i], label=valid_y[i])
                data_for_eval.append([_valid_dmatrix, 'Valid_{}'.format(i+1)])
                logging.info('Validation data {} size: {}'.format(i, valid_X[i].shape))

        # Start training procedure...
        try:
            logging.info('Booster parameters: {}'.format(__booster_params))
            _training_execution_time = MyXGB.tic()
            logging.info('Run training at: {}'.format(_training_execution_time))

            with MyXGB.timer('model training'):
                _trained_model = xgb.train(params=__booster_params,
                                           dtrain=train_dmatrix,
                                           num_boost_round=num_boost_round,
                                           evals=data_for_eval,
                                           obj=obj,
                                           feval=feval,
                                           maximize=maximize,
                                           early_stopping_rounds=early_stopping_rounds,
                                           evals_result=evals_result,
                                           verbose_eval=verbose_eval,
                                           xgb_model=xgb_model,
                                           callbacks=callbacks
                                           )

                if inplace_class_model:
                    self.trained_model = _trained_model

                # If auto-save:
                if autosave_ckpt:
                    _ckpt_file = 'XGBoost_model_ckpt_{}'.format(_training_execution_time)
                    _ckpt_file = os.path.join(self.__checkpoint_dir, _ckpt_file)
                    _trained_model.save_model(_ckpt_file)
                    logging.info('Save model artifact {}'.format(_ckpt_file))

                return _trained_model
        except Exception as e:
            _error_msg = 'Failed in training a XGB model. Error: {}'.format(e)
            logging.error(_error_msg)
            print(_error_msg)
            raise

    def auto_fit(self,
                 train_X,
                 train_y,
                 valid_X=None,
                 valid_y=None,
                 auto_split_train_data=True,
                 booster_params=None,
                 num_boost_round=10,
                 obj=None,
                 eval_metric=None,
                 feval=None,
                 learning_rate=0.1,
                 learning_rate_tolerance=0.00001,
                 higher_better=False,
                 maximize=False,
                 early_stopping_rounds=None,
                 evals_result=None,
                 xgb_model=None,
                 callbacks=None,
                 nthread=-1,
                 verbose_eval=100,
                 autosave_ckpt=True,
                 random_seed=7,
                 silent=True
                 ):

        best_score = -np.inf if higher_better else np.inf
        best_model = None
        adjustable_learning_rate = learning_rate
        _training_execution_time = MyXGB.tic()
        logging.info('Start XGBoost auto fitting at: {}'.format(_training_execution_time))
        while adjustable_learning_rate >= learning_rate_tolerance:
            _to_update_best_model = False
            _trained_model = self.fit(
                train_X=train_X,
                train_y=train_y,
                valid_X=valid_X,
                valid_y=valid_y,
                auto_split_train_data=auto_split_train_data,
                booster_params=booster_params,
                num_boost_round=num_boost_round,
                obj=obj,
                eval_metric=eval_metric,
                feval=feval,
                learning_rate=adjustable_learning_rate,
                maximize=maximize,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                xgb_model=best_model,
                callbacks=callbacks,
                nthread=nthread,
                verbose_eval=verbose_eval,
                autosave_ckpt=False,
                random_seed=random_seed,
                silent=silent
            )

            if higher_better:
                if _trained_model.best_score > best_score:
                    best_score = _trained_model.best_score
                    best_model = _trained_model
                    _to_update_best_model = True
            else:
                if _trained_model.best_score < best_score:
                    best_score = _trained_model.best_score
                    best_model = _trained_model
                    _to_update_best_model = True

            # Save model to checkpoint:
            if _to_update_best_model:
                print('For learning rate of {}, best score is {}'.format(adjustable_learning_rate, best_score))
                _ckpt_file = 'XGBoost_AutoSearch_model_ckpt_{}'.format(_training_execution_time)
                _ckpt_file = os.path.join(self.__checkpoint_dir, _ckpt_file)
                xgb_model = _ckpt_file
                _trained_model.save_model(_ckpt_file)
                logging.info('Save model artifact {}. Best score: {}'.format(_ckpt_file, best_score))

            # Update learning rate:
            adjustable_learning_rate /= 10.0

        return best_model

    def xgb_cv(self,
               cv_X,
               cv_y,
               booster_params=None,
               num_boost_round=100,
               nfold=3,
               stratified=False,
               learning_rate=0.01,
               folds=None,
               metrics=(),
               obj=None,
               feval=None,
               maximize=False,
               early_stopping_rounds=50,
               fpreproc=None,
               as_pandas=True,
               verbose_eval=100,
               show_stdv=True,
               seed=7,
               callbacks=None,
               shuffle=True,
               silent=True
               ):

        try:
            cv_dmatrix = MyXGB.make_dmatrix(data=cv_X, label=cv_y)
            if booster_params is None:
                booster_params = self.__booster_params

            booster_params['learning_rate'] = learning_rate
            booster_params['silent'] = silent
            booster_params['seed'] = seed

            res = xgb.cv(
                    params=booster_params,
                    dtrain=cv_dmatrix,
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
                    seed=seed,
                    callbacks=callbacks,
                    shuffle=shuffle)

            return res
        except Exception as e:
            _error_msg = 'Failed in running XGBoost CV. Error: {}'.format(e)
            print(_error_msg)
            logging.error(_error_msg)
            raise

    def __set_booster_params(self):

        __booster_params = dict(
            max_depth=(4, 9), # verified
            min_child_weight=(0.5, 1.5), # verified.
            reg_alpha=(0, 1), # verified.
            reg_lambda=(0.5, 2.0), # verified.
            subsample=(0.04, 1.5), # verified.
            colsample_bytree=(0.04, 1), # verified.
            colsample_bylevel=(0.04, 1), # verified.
            colsample_bynode=(0.04, 1), # verified.
            min_split_loss=(0, 0.5), # verified.
            # scale_pos_weight=(0.5, 2.5)
        )

        return __booster_params

    def __eval_params_using_cv(
            self,
            max_depth,
            min_child_weight,
            reg_alpha,
            reg_lambda,
            subsample,
            colsample_bytree,
            colsample_bylevel,
            colsample_bynode,
            min_split_loss
            # max_delta_step
            ):

        __booster_params = dict(
            objective=self.__booster_params['objective'],
            # eval_metric=self.__booster_params['eval_metric'],
            max_depth=int(np.round(max_depth)),
            min_child_weight=np.round(min_child_weight, 5),
            reg_alpha=np.round(reg_alpha, 5),
            reg_lambda=np.round(reg_lambda, 5),
            subsample=np.round(subsample, 5),
            colsample_bylevel=np.round(colsample_bylevel, 5),
            colsample_bytree=np.round(colsample_bytree, 5),
            colsample_bynode=np.round(colsample_bynode, 5),
            min_split_loss=np.round(min_split_loss, 5)
            # max_delta_step=int(np.round(max_delta_step))
        )

        _scores = self.xgb_cv(
                           cv_X=self.bayes_opt_X,
                           cv_y=self.bayes_opt_y,
                           booster_params=__booster_params,
                           num_boost_round=10000,
                           nfold=5,
                           stratified=False,
                           learning_rate=0.01,
                           folds=None,
                           metrics=self.bayes_opt_metrics,
                           obj=None,
                           feval=None,
                           maximize=False,
                           early_stopping_rounds=100,
                           fpreproc=None,
                           as_pandas=True,
                           verbose_eval=False,
                           show_stdv=True,
                           seed=7,
                           callbacks=None,
                           shuffle=True,
                           silent=True
                           )

        _score_col = 'test-{}-mean'.format(self.bayes_opt_metrics)

        if self.bayes_is_loss:
            return -_scores[_score_col].min()
        else:
            return _score_col[_score_col].max()

    def bayes_tuning(
            self,
            init_points=5,
            n_iter=25,
            acq='ei',
            xi=0.0,
            eval_func=None,
            pos_eval=True
        ):

        __bayes_opt_params = self.__set_booster_params()

        try:
            if eval_func is None:
                bo = BayesianOptimization(self.__eval_params_using_cv_neg, __bayes_opt_params)
            else:
                bo = BayesianOptimization(eval_func, __bayes_opt_params)

            bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)

            opt_res = pd.DataFrame(bo.res['all']['params'])
            opt_res['values'] = bo.res['all']['values']

            return opt_res
        except Exception as e:
            print('Failed in Bayesian optimization. Error: {}'.format(e))
            raise




















