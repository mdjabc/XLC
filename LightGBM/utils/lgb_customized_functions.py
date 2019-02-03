import numpy as np
import lightgbm as lgb
from sklearn.metrics import *

# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False

# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 init_model=gbm,
#                 fobj=loglikelihood,
#                 feval=binary_error,
#                 valid_sets=lgb_eval)
#
# print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')
#
# print('Starting a new training job...')

# # callback
# def reset_metrics():
#     def callback(env):
#         lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
#         if env.iteration - env.begin_iteration == 5:
#             print('Add a new valid dataset at iteration 5...')
#             env.model.add_valid(lgb_eval_new, 'new_valid')
#     callback.before_iteration = True
#     callback.order = 0
#     return callback
#
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 valid_sets=lgb_train,
#                 callbacks=[reset_metrics()])
#
# print('Finished first 10 rounds with callback function...')