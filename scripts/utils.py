import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import t
from math import factorial
from texttable import Texttable
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
                            average_precision_score,precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

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
    
    
def plot_cv_roc_curve(preds_list, name_list, ci=2):
    fig, ax = plt.subplots(figsize=(7,7))
    mean_auroc_list = []
    std_auroc_list = []
    mean_list = []
    sigma_list = []
    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        tpr_list = []
        auroc_list = []
        base_fpr = np.linspace(0, 1, 101)

        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            fpr, tpr, thresholds = roc_curve(true_fold, preds_fold, pos_label=1)
            auroc = auc(fpr, tpr) # same as roc_auc_score(true_fold, preds_fold)
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_list.append(interp_tpr)
            auroc_list.append(auroc)

        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auroc = auc(base_fpr, mean_tpr)
        std_auroc = np.std(auroc_list)
        mean_auroc_list.append(mean_auroc), std_auroc_list.append(std_auroc)
        std_tpr = np.std(tpr_list, axis=0)
        tpr_upper = np.minimum(mean_tpr + ci*std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - ci*std_tpr, 0)

        mean = ax.plot(base_fpr, mean_tpr, color=f'C{i}', linewidth=3)
        sigma = ax.fill_between(base_fpr, tpr_lower, tpr_upper, color=f'C{i}', alpha=0.2)
        mean_list.append(mean), sigma_list.append(sigma)

    ax.legend([(mean[0], sigma) for mean, sigma in zip(mean_list, sigma_list)],
              [r'Mean $\pm$ {ci} std. dev. ROC {name} (AUROC = {mean:.2f} $\pm$ {std:.2f})'.format(ci=ci, name=name, mean=mean_auroc, std=ci*std_auroc)
               for name, mean_auroc, std_auroc in zip(name_list, mean_auroc_list, std_auroc_list)], loc='lower right')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid()
    
def plot_cv_pr_curve(preds_list, name_list, ci=2):
    fig, ax = plt.subplots(figsize=(7,7))
    mean_auprc_list = []
    std_auprc_list = []
    mean_list = []
    sigma_list = []
    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        precision_list = []
        auprc_list = []
        base_recall = np.linspace(0, 1, 101)

        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            precision, recall, thresholds = precision_recall_curve(true_fold, preds_fold, pos_label=1)
            auprc = average_precision_score(true_fold, preds_fold) # invece di auc(recall, precision) si veda:
            # https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
            precision, recall, thresholds = precision[::-1], recall[::-1], thresholds[::-1]
            interp_precision = np.interp(base_recall, recall, precision)
            precision_list.append(interp_precision)
            auprc_list.append(auprc)

        mean_precision = np.mean(precision_list, axis=0)
        mean_auprc = np.mean(auprc_list)#auc(base_recall, mean_precision)
        std_auprc = np.std(auprc_list)
        mean_auprc_list.append(mean_auprc), std_auprc_list.append(std_auprc)
        std_precision = np.std(precision_list, axis=0)
        precision_upper = np.minimum(mean_precision + ci*std_precision, 1)
        precision_lower = np.maximum(mean_precision - ci*std_precision, 0)

        mean = ax.plot(base_recall, mean_precision, color=f'C{i}', linewidth=3)
        sigma = ax.fill_between(base_recall, precision_lower, precision_upper, color=f'C{i}', alpha=0.2)
        mean_list.append(mean), sigma_list.append(sigma)

    ax.legend([(mean[0], sigma) for mean, sigma in zip(mean_list, sigma_list)],
              [r'Mean $\pm$ {ci} std. dev. PR {name} (AUPRC = {mean:.2f} $\pm$ {std:.2f})'.format(ci=ci, name=name, mean=mean_auprc, std=ci*std_auprc)
               for name, mean_auprc, std_auprc in zip(name_list, mean_auprc_list, std_auprc_list)], loc='lower right')
    #ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid()
    
def classification_report(preds_list, name_list):
    
    report_df = pd.DataFrame(index=name_list, columns=['avg_AUROC', 'avg_AUPRC', 'avg_Precision', 'avg_Recall', 'avg_F1_score',
                                                       'std_AUROC', 'std_AUPRC', 'std_Precision', 'std_Recall', 'std_F1_score'])

    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        auroc_list = []
        auprc_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            auroc_list.append(roc_auc_score(true_fold, preds_fold))
            auprc_list.append(average_precision_score(true_fold, preds_fold))
            precision_list.append(precision_score(true_fold, np.where(preds_fold > 0.5, 1, 0)))
            recall_list.append(recall_score(true_fold, np.where(preds_fold > 0.5, 1, 0)))
            f1_score_list.append(f1_score(true_fold, np.where(preds_fold > 0.5, 1, 0), average='macro'))

        report_df.loc[name, 'avg_AUROC'] = np.mean(auroc_list)
        report_df.loc[name, 'avg_AUPRC'] = np.mean(auprc_list)
        report_df.loc[name, 'avg_Precision'] = np.mean(precision_list)
        report_df.loc[name, 'avg_Recall'] = np.mean(recall_list)
        report_df.loc[name, 'avg_F1_score'] = np.mean(f1_score_list)
        report_df.loc[name, 'std_AUROC'] = np.std(auroc_list)
        report_df.loc[name, 'std_AUPRC'] = np.std(auprc_list)
        report_df.loc[name, 'std_Precision'] = np.std(precision_list)
        report_df.loc[name, 'std_Recall'] = np.std(recall_list)
        report_df.loc[name, 'std_F1_score'] = np.std(f1_score_list)
    
    return report_df.astype(float).round(3)

def statistical_analysis(preds_list, name_list, custom_cv):
    model_scores = {}
    for i, (preds, name) in enumerate(zip(preds_list, name_list)):
        auroc_list = []
        for n_fold, (true_fold, preds_fold) in enumerate(preds):
            auroc_list.append(roc_auc_score(true_fold, preds_fold))
        model_scores[name] = auroc_list
    model_scores = pd.DataFrame(model_scores).T
    
    models_number = len(model_scores)
    n_comparisons = factorial(models_number) / (factorial(2) * factorial(models_number - 2))
    pairwise_t_test = []

    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        n = differences.shape[0]
        df = n - 1
        n_train = len(list(custom_cv)[0][0])
        n_test = len(list(custom_cv)[0][1])
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append([model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val])

    pairwise_comp_df = pd.DataFrame(pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]).round(3)
    return pairwise_comp_df
    
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


def add_subplot_border(ax, width=1, color=None ):

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)
    
def find_bins(observations, width):
    minimmum = np.min(observations)
    maximmum = np.max(observations)
    bound_min = -1.0 * (minimmum % width - minimmum)
    bound_max = maximmum - maximmum % width + width
    n = int((bound_max - bound_min) / width) + 1
    bins = np.linspace(bound_min, bound_max, n)
    return bins


