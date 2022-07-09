import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import t
from texttable import Texttable
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from config import *

def create_folds(train, features, target, num_folds, num_repeats=None, shuffle=True, seed=42):
    folds = []
    if num_repeats is None:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(skf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    else:
        rskf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(rskf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    return folds

def seed_basic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seed_everything(seed):
    seed_basic(seed)
    
def evaluate(params, train, features, target, folds, return_res=False, return_preds=False):
    train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=-1, show_stdv=True), lgb.early_stopping(stopping_rounds=params['early_stopping_round'], first_metric_only=False, verbose=True)]
    cv_results = lgb.cv(params=params,
                        train_set=train_lgb,
                        folds=folds,
                        metrics=params['metric'],
                        num_boost_round=params['num_iterations'],
                        stratified=False,
                        callbacks=callbacks,
                        eval_train_metric=True,
                        return_cvbooster=True)
    best_iteration = cv_results['cvbooster'].best_iteration
    oof_preds = np.zeros(train.shape[0])
    columns = ['valid_' + metric for metric in params['metric']] if isinstance(params['metric'], list) else ['valid_' + params['metric']]
    results = pd.DataFrame(index=range(len(folds)), columns=['fold'] + columns)
    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(folds):
        train_fold = train.loc[train_fold_idx].copy()
        valid_fold = train.loc[valid_fold_idx].copy()
        train_fold_lgb = lgb.Dataset(train_fold[features], train_fold[target], reference=train_lgb)
        valid_fold_lgb = lgb.Dataset(valid_fold[features], valid_fold[target], reference=train_lgb)
        model = cv_results['cvbooster'].boosters[n_fold]
        oof_preds[valid_fold_idx] = model.predict(valid_fold[features], num_iteration=best_iteration)
        results.loc[n_fold, 'fold'] = n_fold+1
        results.loc[n_fold, 'valid_auc'] = roc_auc_score(valid_fold[target], oof_preds[valid_fold_idx])
        print('Fold {fold} AUC: {auc:.5f}'.format(fold=n_fold+1, auc=results.loc[n_fold, 'valid_auc']))
    print('Folds AUC: {avg_auc:.5f}+-{std_auc:.5f}'.format(avg_auc=results['valid_auc'].mean(), std_auc=results['valid_auc'].std(ddof=0)))
    print('Total AUC: {auc:.5f}'.format(auc=roc_auc_score(train[target], oof_preds)))
    if return_res:
        return results
    if return_preds:
        return oof_preds
    
def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


