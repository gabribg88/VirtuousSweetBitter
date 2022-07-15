"""
VIRTUOUS SWEET BITTER

The VirtuousSweetBitter tool predict the sweet/bitter taste of quey molecules based on their molecular structures.

This tool is mainly based on:
    1. VirtuousSweetBitter.py: a main script which calls the following functionalities
    2. Virtuous.py: library of preprocessing functionalities
    3. testing_sweetbitter.py: prediction code

To learn how to run, just type:

    python VirtuousSweetBitter.py --help

usage: VirtuousSweetBitter.py [-h] [-s SMILES] [-f FILE] [-v VERBOSE]

VirtuousSweetBitter: ML-based tool to predict the sweet/umami taste

optional arguments:
  -h, --help            show this help message and exit
  -c COMPOUND, --compound COMPOUND
                        query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)
  -f FILE, --file FILE  text file containing SMILES of the query molecules
  -d DIRECTORY, --directory DIRECTORY
                        name of the output directory
  -v VERBOSE, --verbose VERBOSE
                        Set verbose mode (default: False; if True print messagges)

To test the code you can submit an example txt file in the samples fodler (SMILES.txt)

The code will create a log file and an output folder containing:
    1. "best_descriptors.csv": a csv file collecting the 12 best molecular descriptors for each processed smiles on which the prediction relies
    2. "descriptors.csv": a csv file collecting the molecular descriptors for each processed smiles
    3. "predictions.csv": a csv summarising the results of the prediction

"""

__version__ = '0.1.0'
__author__ = 'Virtuous Consortium'


import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.info")
import os
import argparse
import time
import urllib.parse
import urllib.request
import sys
import xmltodict

# # Import Virtuous Library
import Virtuous

# Import the prediction function
import testing_sweetbitter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tstamp = time.strftime('%Y_%m_%d_%H_%M')

if __name__ == "__main__":

    # --- Parsing Input ---
    parser = argparse.ArgumentParser(description='VirtuousSweetBitter: ML-based tool to predict the umami taste')
    parser.add_argument('-c','--compound',help="query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)",default=None)
    parser.add_argument('-f','--file',help="text file containing the query molecules",default=None)
    parser.add_argument('-d','--directory',help="name of the output directory",default=None)
    parser.add_argument('-v','--verbose',help="Set verbose mode", default=False, action='store_true')
    args = parser.parse_args()

    # --- Print start message
    if args.verbose:
        print ("\n\t   *** VirtuousSweetBitter ***\nAn ML-based algorithm to predict the sweet/bitter taste\n")

    # --- Setting Folders and files ---
    # Stting files needed by the code
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(scripts_path)
    data_dir = root_dir + os.sep + "data" + os.sep 
    models_dir = root_dir + os.sep + "models" + os.sep 
    AD_file = data_dir + "bittersweet_AD_train.pkl"
    
    
    # Setting output folders and files
    if args.directory:
        output_folder1 = os.getcwd() + os.sep + args.directory + os.sep
    else:
        output_folder1 = os.getcwd() + os.sep + 'Output_folder_' + str(tstamp) + os.sep
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

    # --- Preprocessing (Virtuous.py) ---

    # 1.1 Defining the SMILES to be processed

    # if user defined only one compound with the --compound directive
    if args.compound:
        query_cpnd = []
        query_cpnd.append(args.compound)

    # if the user defined a txt file collecting multiple molecules
    elif args.file:
        with open(args.file) as f:
            query_cpnd = f.read().splitlines()

    else:
        sys.exit("\n***ERROR!***\nPlease provide a SMILES or a txt file containing a list of SMILES!\nUse python ../VirtuousSweetBitter.py --help for further information\n")

    # 1.2 Import compound as a molecule object
    mol = [Virtuous.ReadMol(cpnd, verbose=args.verbose) for cpnd in query_cpnd]

    # 1.3 Standardise molecule with the ChEMBL structure pipeline (https://github.com/chembl/ChEMBL_Structure_Pipeline)
    standard = [Virtuous.Standardize(m) for m in mol]
    # take only the parent smiles
    issues     = [i[0] for i in standard]
    std_smi    = [i[1] for i in standard]
    parent_smi = [i[2] for i in standard]

    # 1.4 Check the Applicability Domain (AD)
    check_AD = [Virtuous.TestAD(smi, filename=AD_file, verbose = False, sim_threshold=0.21, neighbors = 5, metric = "tanimoto") for smi in parent_smi]
    test       = [i[0] for i in check_AD]
    score      = [i[1] for i in check_AD]
    sim_smiles = [i[2] for i in check_AD]

    # 1.5 Featurization: Calculation of the molecular descriptors
    #DescNames, DescValues = Virtuous.CalcDesc(parent_smi, Mordred=True, RDKit=False, pybel=False)
    descs = [Virtuous.CalcDesc(smi, Mordred=True, RDKit=True, pybel=True) for smi in parent_smi]
    DescValues = []
    for d in descs:
        DescValues.append(d[1])
    DescNames = descs[0][0]
    df = pd.DataFrame(data = DescValues, columns=DescNames)
    df.insert(loc=0, column='SMILES', value=parent_smi)
    df.to_csv(output_folder1 + "descriptors.csv", index=False)
    
    # remove eventual duplicated columns (same values from different libreries, i.e. Mordred, RDKit and pybel)
    df = df.loc[:,~df.columns.duplicated()]

    # save only best descriptors    
    features = ['BCUT2D_MRHI','AXp-6dv','piPC4','GATS1d','Kappa3','AATS7i','AATS8i','GATS2v','MATS1v','GATS2m','MATS2s','MATS2d','GATS3dv','GATS4dv',
                'ATSC5c','ATSC5d','GATS6s','ATSC7dv','MPC5','BCUTi-1h','fr_Ndealkylation1','MINssO','MDEC-13','PEOE_VSA8','MINdO','BCUTdv-1l','fr_NH0',
                'naHRing','SlogP_VSA10']
        
    df_best = df[['SMILES'] + features]
    df_best.to_csv(output_folder1 + "best_descriptors.csv", index=False)

    # --- Run the model (testing_umami.py) ---
    pred_label = []
    pred_prob  = []
    
    for i in range(len(df)): 
        label, prob = testing_sweetbitter.PredictExplain(df.iloc[i:i+1], make_plot=False, savefig=False, fname=None)
        pred_label.append(label)
        pred_prob.append(prob)
        

    # --- Collect results --
    col_names = ["SMILES", "Check AD", "class", "probability"]
    #df = pd.read_csv(output_folder1 + "result_labels.txt", sep="\t", header=None)
    results_df = pd.DataFrame()
    results_df['SMILES'] = parent_smi
    results_df['Check AD'] = test
    results_df['Class'] = pred_label
    results_df['Probability'] = pred_prob
    results_df.to_csv(output_folder1 + "predictions.csv", index=False)

    if args.verbose:
        print("")
        print (results_df)
        print("")
