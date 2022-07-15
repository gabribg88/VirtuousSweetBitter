import pickle
import shap
from copy import deepcopy
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
import seaborn as sns
import argparse


__version__ = '0.1.0'
__author__ = 'Gabriele Maroni'


def PlotPredictExplain(shap_values=None, train=None, target=None, smiles=None, fname=None, savefig=False, FEATS_TO_DISPLAY = 10):
    
    """
    Function to plot the prediction results
    :params
    :return: matplotlib figure object
    """  
        
    if not shap_values: 
        sys.exit("Please provide shap values to make fancy plots!")
        
    if train is None: 
        sys.exit("Please provide training data to make fancy plots!")
    
    if target is None: 
        sys.exit("Please provide target column to make fancy plots!")
    
    # retrieve directories tree
    scripts_dir =  os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(scripts_dir)   # retrieve directories tree
    
    shap_values_tmp = deepcopy(shap_values)

    if shap_values_tmp.values.sum()+shap_values_tmp.base_values > 1:
        d1 = 1.0 - shap_values_tmp.base_values
        d2 = d1/shap_values_tmp.values.sum()
        shap_values_tmp.values = d2*shap_values_tmp.values

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 6, figure=fig)

    ### Shap profiles
    ax2 = plt.subplot(gs[0:3, 0:3])
    plt.sca(ax2) # set current axis
    ax2.grid(zorder=0)
    shap_waterfall(shap_values_tmp, max_display=FEATS_TO_DISPLAY, show=False)
    fig.set_size_inches(12,7)

    ### Feature distributions
    indices = list(itertools.product([0,1,2], [3,4,5]))
    tmp = pd.DataFrame({'shap_values': shap_values.values,
                        'data': shap_values.data,
                        'abs_shap_values': np.abs(shap_values.values)},
                        index=shap_values.feature_names).dropna().sort_values('abs_shap_values', ascending=False).iloc[:9]
    feats = tmp.index.tolist()
    datas = tmp.data.values
    axes = []

    for idx,col,dat in zip(indices, feats, datas):
        ax = plt.subplot(gs[idx])
        axes.append(ax)
        train_tmp = train.copy()
        if col == 'BCUTi-1h':
            train_tmp = train.dropna(subset=[col]).copy()
            train_tmp[col] = train_tmp[col].clip(train_tmp[col].quantile(0.01), train_tmp[col].quantile(0.99))
            bins = find_bins(train_tmp[col].values, 1)
            train_tmp['bins'] = pd.cut(train_tmp[col], bins=bins, include_lowest=True)
            train_tmp_grouped = train_tmp.groupby('bins', as_index=False)[col].mean()
            train_tmp = pd.merge(train_tmp, train_tmp_grouped, how='left', on='bins', suffixes=(None, '_binned'))
            sns.histplot(data=train_tmp, x=col+'_binned', hue=target, kde=False, stat='density', edgecolor='white', linewidth=0, alpha=0.4, line_kws=dict(linewidth=3), legend=False)
            ax.set_xlabel(col)
        elif col == 'fr_NH0':
            train_tmp = train.dropna(subset=[col]).copy()
            minx, maxx = train_tmp[col].min(), train_tmp[col].max()
            sns.histplot(data=train_tmp, x=col, hue=target, kde=False, stat='density', edgecolor='white', linewidth=0, alpha=0.4, line_kws=dict(linewidth=3), bins=np.arange(minx-0.25, maxx+0.25, 0.5), legend=False)
        elif col in ['PEOE_VSA8', 'MPC5']:
            dic = {'PEOE_VSA8':5, 'MPC5':10}
            train_tmp = train.dropna(subset=[col]).copy()
            train_tmp[col] = train_tmp[col].clip(train_tmp[col].quantile(0.01), train_tmp[col].quantile(0.99))
            minx, maxx = train_tmp[col].min(), train_tmp[col].max()
            d = dic[col]
            sns.histplot(data=train_tmp, x=col, hue=target, kde=True, stat='density', edgecolor='white', linewidth=0, alpha=0.4, line_kws=dict(linewidth=3), bins=np.arange(minx-d/2, maxx+d/2, d), legend=False)
        else:
            train_tmp[col] = train_tmp[col].clip(train_tmp[col].quantile(0.01), train_tmp[col].quantile(0.99))
            sns.histplot(data=train_tmp, x=col, hue=target, kde=True, stat='density', edgecolor='white', linewidth=0, alpha=0.4, line_kws=dict(linewidth=3), ax=ax, legend=False)

        ax.axvline(dat, color='r')
        ax.set_ylabel(None)
        ax.xaxis.get_label().set_fontsize(13)
        ax.tick_params(axis='y', which='major', labelsize=0, left=False)
        ax.tick_params(axis='y', which='minor', labelsize=0, left=False)
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='x', which='minor', labelsize=8)
    
    fig.set_facecolor("w")
    
    # save results
    pred_prob = shap_values_tmp.base_values + shap_values_tmp.values.sum()
    pred_label = 'Sweet' if pred_prob >= 0.5 else 'Bitter'
    out_prob = pred_prob if pred_label == 'Sweet' else (1-pred_prob)    # actual probability for the two classes
        
    plt.suptitle(fr'$\bfSHAP\,\,values\,\,for:\,\,$ {smiles}''\n'fr'$\bfPrediction:\,\,$ {pred_label} with {out_prob.round(3)} probability',
                     fontsize=14, ha='center')     
    
    if savefig:
        if not fname:
            images_dir = root_dir + os.sep + "images" + os.sep
            Path(images_dir).mkdir(parents=True, exist_ok=True)
            fname = os.path.join(images_dir, f'SHAP_{smiles}.png')   
    
        fig.savefig(fname, dpi=fig.dpi)
        
    return fig




def PredictExplain(sample_df, make_plot=False, savefig=False, fname=None):
    
    """
    Function to predict the Sweet/Bitter taste
    :param sample_df: pandas dataframe (row represent compound, columns represent features) 
    :param make_plot: if True, plot the results of the prediction
    :param savefig: if True save figure with prediction results (default: False)
    :param fname: str or path-like of the output figure with prediction results; if no filename is provided an the savefig is True, the figure will be saved in the "images" folder in the root folder of the VirtuousSweetBitter folder.
    :return: pred_label (string defining the predicted class), out_prob (float defining the probability of the predicted class)
    """    
    
    # retrieve directories tree
    scripts_dir =  os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(scripts_dir)
    data_dir = root_dir + os.sep + "data" + os.sep 
    models_dir = root_dir + os.sep + "models" + os.sep 
    
    ## Load data
    with open(data_dir + 'comb.pickle', 'rb') as handle:
        data = pickle.load(handle)
    df, metadata, features, target, rows = data.values()
    
    ## Load models and explainers
    with open(models_dir + 'models.pickle', 'rb') as handle:
        models_explainers = pickle.load(handle)
    models, explainers = models_explainers.values()
    
    ## Selected features
    features = ['BCUT2D_MRHI','AXp-6dv','piPC4','GATS1d','Kappa3','AATS7i','AATS8i','GATS2v','MATS1v','GATS2m','MATS2s','MATS2d','GATS3dv','GATS4dv',
                'ATSC5c','ATSC5d','GATS6s','ATSC7dv','MPC5','BCUTi-1h','fr_Ndealkylation1','MINssO','MDEC-13','PEOE_VSA8','MINdO','BCUTdv-1l','fr_NH0',
                'naHRing','SlogP_VSA10']
    
    ## Train dataset (for plotting)
    train = df.loc[rows,features+[target]].copy()
    train.reset_index(drop=True, inplace=True)
    train[target].replace({'Bitter': 0, 'Sweet': 1}, inplace=True)
    
    ## Compute oof predictions and explanations
    oof_preds = np.zeros(len(models))
    oof_shap_values = np.zeros((len(models), len(features)))
    base_values = np.zeros(len(models))

    for i, (model, explainer) in enumerate(zip(models, explainers)):
        oof_preds[i] = model.predict(sample_df[features])
        oof_shap_values[i,:] = explainer.shap_values(sample_df[features], check_additivity=True)
        base_values[i] = explainer.expected_value
    
    ## Create Explanation object
    shap_values = shap.Explanation(values=np.mean(oof_shap_values, axis=0),
                                   base_values=np.mean(base_values),
                                   data=sample_df[features].values.squeeze(),
                                   feature_names=features
                                  )
    
    shap_values_tmp = deepcopy(shap_values)
    
    if shap_values_tmp.values.sum()+shap_values_tmp.base_values > 1:
        d1 = 1.0 - shap_values_tmp.base_values
        d2 = d1/shap_values_tmp.values.sum()
        shap_values_tmp.values = d2*shap_values_tmp.values
    
    # save results
    pred_prob = shap_values_tmp.base_values + shap_values_tmp.values.sum()
    pred_label = 'Sweet' if pred_prob >= 0.5 else 'Bitter'
    out_prob = pred_prob if pred_label == 'Sweet' else (1-pred_prob)    # actual probability for the two classes
    
    smiles = sample_df.SMILES.values[0]
    
    # plot if user want to
    if make_plot:
           
        fig = PlotPredictExplain(shap_values=shap_values, train=train, target=target, smiles=smiles, savefig=savefig, fname=fname, FEATS_TO_DISPLAY = 10)       
    
    return pred_label, out_prob

        
if __name__ == "__main__":
    
    # --- Parsing Input ---
    parser = argparse.ArgumentParser(description='testing Sweet/Bitter taste from a pandas dataframe(from VirtuousSweetBitter)')
    parser.add_argument('-f','--file',help="csv file (row: compound, columns: features)",default=None)
    parser.add_argument('-v','--verbose',help="Set verbose mode", default=False, action='store_true')
    args = parser.parse_args()

    # --- Print start message
    if args.verbose:
        print ("\nTesting Sweet/Bitter taste from a csv file (prediction tool from VirtuousSweetBitter)\n")
        
    print("Under development to work as a stand-alone package!")

    