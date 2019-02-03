lgb_default_settings = {
    "multi-class-objectives": ["multiclass", "multiclassova"], 
    "regression-objectives": ["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"], 
    "binary-objectives": ["binary"], 
    "default-metrics": ["l1", "mean_absolute_error", "mae", "regression_l1", "l2", \
                        "mean_squared_error", "mse", "regression_l2", "regression", \
                        "l2_root", "root_mean_squared_error", "rmse", "quantile", \
                        "mape", "mean_absolute_percentage_error", "huber", "fair", \
                        "poisson", "gamma", "gamma_deviance", "tweedie", "ndcg", \
                        "map", "auc", "binary_logloss", "binary_error", "multi_logloss", \
                        "multiclass", "softmax", "multiclassova", "multiclass_ova", \
                        "ova", "ovr", "multi_error", "xentropy", "cross_entropy", \
                        "xentlambda", "cross_entropy_lambda", "kldiv", "kullback_leibler"], 
    "lambdarank-objectives": ["lambdarank"], 
    "cross-entropy-objectives": ["cross_entropy", "cross_entropy_lambda"]
    }