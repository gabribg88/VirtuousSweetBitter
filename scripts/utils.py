import os
import re
import sys
import random
import colors
import warnings
import matplotlib
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


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']
    
    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]
        
        #Get class
        _class = getattr(module, class_name, None)
        
        if _class is None:
            continue
        
        if isinstance(obj, _class):
            return True

    return False

def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s

def shap_waterfall(shap_values, max_display=10, show=True):
    """ Plots an explantion of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    
    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.

    max_display : str
        The maximum number of features to plot.

    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    
    # Turn off interactive plot
    if show is False:
        plt.ioff
    

    base_values = shap_values.base_values
    
    features = shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first " \
                        "parameter, but you have passed an array as the first parameter! " \
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or " \
                        "for multi-output models try " \
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception("The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])
    
    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width, zorder=10
        )
        
        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=14, zorder=10
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=14, zorder=10
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb,
            width=bar_width,
            head_width=bar_width, zorder=10
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb
            )
        
        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=14, zorder=10
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=14, zorder=10
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    plt.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1/num_features, color="k", linestyle="--", linewidth=1.5, zorder=-1) # era color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="k", linestyle="--", linewidth=1.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=15)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([base_values, base_values+1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$","\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=14, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticklabels(["$f(x)$","$ = "+format_value(np.minimum(fx,0.999), "%0.03f")+"$"], fontsize=14, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
    
    if show:
        plt.show()
        pass
    else:
        return plt.gcf()
