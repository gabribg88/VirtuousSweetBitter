"""
Python library developed inside the Virtuous Projct (https://virtuoush2020.com/)

The present library collects a series of functions divided into different sections:

- Section a) Preprocessing: import file, Mol2Smiles, Standardize, Fingerprint Calculation
- Section b) Calculating Descriptors
- Section c) Automatic functions for DB processing
- Section d) Function to calculate and verify the Applicability Domin (AD) of a model

----------------
Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement Action (GA No. 872181)

----------
References

[1] Moriwaki, H., Tian, Y.-S., Kawashita, N., & Takagi, T. (2018). Mordred: a molecular descriptor calculator. Journal of Cheminformatics, 10(1), 4. https://doi.org/10.1186/s13321-018-0258-y
[2] Bento, A. P., Hersey, A., Félix, E., Landrum, G., Gaulton, A., Atkinson, F., Bellis, L. J., De Veij, M., & Leach, A. R. (2020). An open source chemical structure curation pipeline using RDKit. Journal of Cheminformatics, 12(1), 51. https://doi.org/10.1186/s13321-020-00456-1
[3] N M O'Boyle, M Banck, C A James, C Morley, T Vandermeersch, and G R Hutchison. "Open Babel: An open chemical toolbox." J. Cheminf. (2011), 3, 33. DOI:10.1186/1758-2946-3-33
[4] RDKit: Open-source cheminformatics; http://www.rdkit.org

----------------
Version history:
- Version 1.0 - 22/10/2021
- Version 1.1 - 27/10/2021: Adding automatic functions for DB processing
- Version 1.2 - 06/12/2021: Removing pybel, Chemopy and Padel descriptors, removing multiprocess
- Version 1.3 - 28/01/2022: Adding function for calculating Morgan Fingerprint using RDKit function and check the Applicability Domain (AD)
- Version 1.4 - 14/03/2022: Re-adding pybel, counters for the standardisations error/removals in DB processing; adding dynamic issue threashold
- Version 1.5 - 16/03/2022: Return the five most similar compounds from the applicability domain
- Version 1.6 - 28/03/2022: Adding nbits and radius parameters for fingerprint calculation in the DefineAD and TestAD function; adding different metrics on the TestAD
- Version 1.7 - 05/05/2022: Adding query to pubchem in case of providing name of a compound; automatic detection of molecular input
- Version 1.8 - 13/07/2022: Changing pybel import
"""

__version__ = '1.8'
__author__ = 'Lorenzo Pallante'

import sys
import os
import pickle
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import shap
from chembl_structure_pipeline import standardizer, checker
from mordred import Calculator, descriptors
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from openbabel import pybel
import urllib.parse
import urllib.request
import sys
import xmltodict
from copy import deepcopy
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
import seaborn as sns

# disable printing warning from RDKit
rdkit.RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# === Section a) PREPROCESSING ===

def pubchem_query(cpnd, verbose = True ):
    """
        Function to search a compound in PubChem based on its name
        :param cpnd: Name of the query compound (e.g. "sucrose")
    """
    if verbose:
        print(f"Querying PubChem for {cpnd}..." )

    # start of the pubchem address
    prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"

    # search by compound name
    input = "name/%s" % urllib.parse.quote(cpnd)

    # link to the cid and SMILES
    operation_cid = "/cids/TXT"

    # generate url to the CID of the compound
    cidurl = prolog + input + operation_cid

    # try to retrieve the CID
    try:
        file = urllib.request.urlopen(cidurl)
        data = file.read()
        file.close()
        cid = int(data)
        if verbose:
            print("NOTE: Closest match on PubChem has CID %s!" % cid)

        # try to retrieve the SMILES
        try:
            input = "cid/%s" % cid
            operation_smi = "/property/CanonicalSMILES/XML"

            url_smi = prolog + input + operation_smi

            file_smi = urllib.request.urlopen(url_smi)
            data_smi = file_smi.read()
            file_smi.close()
            data_smi = xmltodict.parse(data_smi)

            canonical_smiles = data_smi["PropertyTable"]["Properties"]["CanonicalSMILES"]

        except:
            print(f"Cannot retrieve SMILES for {cpnd}")

    except:
        print("Cannot retrieve CID for %s!" % cpnd)

    return canonical_smiles


def ReadMol (file, verbose=True):
    """
    Function to read input file in different file format, run sanification with RDKit and return RDKit mol object
    :param file: molecule file or string
    :return: RDKit molecule object
    """
    # SMILES is expected as input
    mol = Chem.MolFromSmiles(file, sanitize=False)

    if mol:
        type = 'SMILES'
    else:
        mol = Chem.MolFromFASTA(file, sanitize=False)
        if mol:
            type = 'FASTA'
        else:
            mol = Chem.MolFromSequence(file, sanitize=False)
            if mol:
                type = 'SEQUENCE'
            else:
                try:
                    mol = Chem.MolFromInchi(file, sanitize=False)
                except:
                    pass
                if mol:
                    type = 'Inchi'
                else:
                    try:
                        mol = Chem.MolFromPDBFile(file, sanitize=False)
                    except:
                        pass
                    if mol:
                        type = 'PDB'
                    else:
                        try:
                            mol = Chem.MolFromSmarts(file, sanitize=False)
                        except:
                            pass
                        if mol:
                            type = 'Smarts'
                        else:
                            try:
                                smi = pubchem_query(file, verbose=verbose)
                                mol = Chem.MolFromSmiles(smi, sanitize=False)
                            except:
                                pass
                            if mol:
                                type = 'pubchem name'
                            else:
                                sys.exit("\nError while reading your query compound: check your molecule!\nNote that "
                                         "allowed file types are SMILES, FASTA, Inchi, Sequence, Smarts or pubchem "
                                         "name")

    if verbose:
        print(f"Input has been interpeted as {type}")

    # try to sanitiaze the molecule
    try:
        Chem.SanitizeMol(mol)

    # if somithing goes wrong during sanitising
    except Exception as err:
        sys.exit("Error in the RDKit Sanification process: %s" %err)

    return mol


def Mol2Smiles (mol):
    """
    Function to write SMILES from molecule object
    :param mol: RDKit molecule object
    :return: smiles of the input molecule
    """
    smiles = Chem.MolToSmiles(mol)

    return smiles

def Standardize (mol):
    """
    Function to preprocess molecules with the ChEMBL structure pipeline (https://github.com/chembl/ChEMBL_Structure_Pipeline)
    [2] Bento, A. P., Hersey, A., Félix, E., Landrum, G., Gaulton, A., Atkinson, F., Bellis, L. J., De Veij, M., & Leach, A. R. (2020). An open source chemical structure curation pipeline using RDKit. Journal of Cheminformatics, 12(1), 51. https://doi.org/10.1186/s13321-020-00456-1
    :param mol: RDKit molecule object
    :return: issues (issue score from checker), std_smi (standard smiles from standardizer), parent_smi (parent smiles from get parent)
    """

    # define molblock from smiles
    molblock = Chem.MolToMolBlock(mol)

    # launch the checker
    issues = checker.check_molblock(molblock)

    # standardise molecule
    std_mol = standardizer.standardize_mol(mol)
    std_smi = Chem.MolToSmiles(std_mol)

    # get parent molecule
    parent, _ = standardizer.get_parent_mol(std_mol)
    parent_smi = Chem.MolToSmiles(parent)

    return issues, std_smi, parent_smi


def Calc_fps (smiles, nbits=1024, radius=2):
    """
    Function to calculate Morgan fingerprint using the RDKit function
    :param smiles: smiless of the query molecule
    :param nbits: number of bits to compute the fingerprint
    :param radius: radius of the circular fingerprint
    :return: Morgan fingerprint of the query smiles
    """
    fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),radius,nBits=nbits,useChirality=True)
    return fps



# ==============================================================================
# === Section b) DESCRIPTORS CALCULATION ===

def Calc_Mordred (smiles, ignore_3D=False):
    """
    Function to calculate Mordred Descriptors (https://github.com/mordred-descriptor/mordred)
    [1] Moriwaki, H., Tian, Y.-S., Kawashita, N., & Takagi, T. (2018). Mordred: a molecular descriptor calculator. Journal of Cheminformatics, 10(1), 4. https://doi.org/10.1186/s13321-018-0258-y

    :param smiles: smiles of the query molecule
    :param ignore_3D: ignore the 3D descriptors calculation
    :return: values and names of the Mordred descriptors (1613 2D and 213 3D)
    """
    # check smile and generating RDKit molecule from smiles
    mol = ReadMol(smiles, verbose=False)

    #######################################################
    # TO BE REVISED --> 7 molecules of the DB failed with "Bad Conformed Id"
    # build 3D coordinates if you want to compute 3D features as well
    #if not ignore_3D:
    #    mol=Chem.AddHs(mol)
    #    Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=5000)
    #    Chem.AllChem.UFFOptimizeMolecule(mol)
    ######################################################

    # Mordred descriptors
    calc = Calculator(descriptors, ignore_3D=ignore_3D)

    # create string vector containing all the Mordred descriptors
    descNames = [str(Desc) for j, Desc in enumerate(calc.descriptors)]

    try:

        # calculating all descriptors
        out = calc(mol)
        descValues = out[:]

    except Exception as err:

        print ("Error during Mordred descriptors calculation on %s: %s" %(smile, err))
        # fill descValues with Nan
        descValues = np.empty((len(descNames),))
        descValues[:] = np.nan

    return np.array(descNames, dtype=str), np.array(descValues, dtype=float)


def Calc_RDKit (smile):
    """
    Function to calculate RDKit Descriptors
    [4] RDKit: Open-source cheminformatics; http://www.rdkit.org

    :param smiles: smiles of the query molecule
    :return: values and names of the RDKit descriptors (208)
    """
    # create string vector containing all the RDKit descriptors
    DescNames = [str(Desc[0]) for j, Desc in enumerate(Chem.Descriptors.descList)]

    # generating molecules from smile
    mol = Chem.MolFromSmiles(smile)

    # initialising vector for descriptors of the i-esime moleucles
    desc = []

    # calculating all descriptors
    for j, Desc in enumerate(Chem.Descriptors.descList):

        try:
            desc.append(Desc[1](mol))

        except Exception as err:

            print ("Error during RDKit descriptors calculation on %s: %s. %s failed" %(smile, err, str(Desc[0])))
            desc.append(np.nan)

    return np.array(DescNames, dtype=str), np.array(desc, dtype=float)


def Calc_pybel (smile):
    """
    Function to calculate pybel Descriptors
    [3] N M O'Boyle, M Banck, C A James, C Morley, T Vandermeersch, and G R Hutchison. "Open Babel: An open chemical toolbox." J. Cheminf. (2011), 3, 33. DOI:10.1186/1758-2946-3-33

    :param smiles: smiles of the query molecule
    :return: values and names of the pybel descriptors (25)
    """
    # read smile input with pybel
    mymol = pybel.readstring("smi", smile)

    # calculate descriptors
    desc  = mymol.calcdesc()

    DescValues = list(desc.values())
    DescNames  = list(desc.keys())

    return np.array(DescNames, dtype=str), np.array(DescValues, dtype=float)


def CalcDesc(smi, Mordred=True, RDKit=True, pybel=True):
    """
    Function to calculate different types of descriptors

    :param smi: smiles of the query molecule
    :param Mordred: if True calculate Mordred descriptors, if False not
    :param RDKit: if True calculate RDKit descriptors, if False not
    :param pybel: if True calculate pybel descriptors, if False not
    :return: values and names of the required descriptors
    """
    # initialise lists for descriptors names and values
    all_desc_names = []
    all_desc_values = []

    if Mordred:

        names, values   = Calc_Mordred(smi)
        all_desc_names.extend(names)
        all_desc_values.extend(values)

    if RDKit:

        names, values   = Calc_RDKit(smi)
        all_desc_names.extend(names)
        all_desc_values.extend(values)

    if pybel:

        names, values   = Calc_pybel(smi)
        all_desc_names.extend(names)
        all_desc_values.extend(values)

    return np.array(all_desc_names, dtype=str), np.array(all_desc_values, dtype=float)



# ==============================================================================
# === Section c) AUTOMATIC FUNCTIONS FOR DB PROCESSING ===

def Import_DB (DB_filename, smile_field = "SMILES", output = "out.csv", sep = ",", header = "infer", standardize = True, max_len = None, remove_issues = True, remove_duplicates = True):
    """
    Function to import, check, standardize and preprocess a database of molecules collected in a csv file

    :param DB_filename: filename of the csv file collecting data
    :param smile_field: file of the dataset containing the smiles of the compounds to be processed
    :param output: output filename of the outputfile
    :param sep: separator character which divides entries in the csv
    :param header: header of the csv file
    :param standardize: if True, run the standardisation from the ChEMBL_Structure_Pipeline
    :param max_len: if specified, remove compounds with smiles longer than the defined maximum length
    :param remove_issues: if True, remove compouds with issue scores from the ChEMBL_Structure_Pipeline higher than 5
    :param remove_duplicates: if True, remove duplicates in the dataset

    :return: preprocessed dataset in pandas and in csv format accoring to the defined parameters
    """
    # read DB from csv and create a pandas dataframe
    DB = pd.read_csv(DB_filename, sep = sep, header = header, low_memory=False)

    # ===== CHECK MOLECULES =====
    print ("Checking Molecules... ")

    # create log file
    f = open("Check_DB.log", "w")

    # create list for RDKit output smiles
    RDKit_smiles = []
    errors = 0
    errors_mol = []

    # create progress bar
    p_bar = tqdm(range(len(DB)))

    # loop over molecules in starting DB
    for entry in p_bar:

        # isolate smi from the DB
        smi = DB[smile_field].iloc[entry]

        # Check Smiles in the DB using RDKit and save smile
        try:
            mol = ReadMol(smi, verbose=False)
            RDKit_smiles.append(Mol2Smiles(mol))

        # if errors encountered
        except SystemExit as se:
            errors += 1
            f.write ("Error on entry #%s: %s\n" %(entry, se))
            errors_mol.append(entry)

    # if no errors in the RDKit sanitisazion process, write the RDKit Smiles and remove the log file
    if errors == 0:

        DB ["SMILES_check"] = RDKit_smiles
        os.remove("Check_DB.log")
        print ("OK: Check results in no errors!")

    # if errors encountered, print error message and preserve log file
    else:
        f.write ("\nSummary of entries generating errors: \n%s" %(errors_mol))
        f.close()
        sys.exit ("%d errors encounterd! Please read file Check_DB.log\nEntries with errors: %s" %(errors, errors_mol))

    # start the standardisation if the standardize is True
    if standardize:
    #######################
    # STANDARDISE MOLECULES
        print ("Standardizing molecules using the ChEMBL Structure Pipeline ... ")

        # arrays for ChEMBL structure pipeline standardisation
        issues        = []
        std_smiles    = []
        parent_smiles = []
        # array to store molecules with smiles longer than a threshold
        long_smi      = []
        # array to store molecules with errors in the Standardization (Checker values higher than 5)
        error_mol     = []

        # create progress bar
        p_bar = tqdm(range(len(DB)))

        # loop over molecules in starting DB
        for entry in p_bar:

            # isolate smi from the DB
            smi = DB[smile_field].iloc[entry]

            # check if smiles is greater than the maxium allowed length (Default: None)
            if max_len and len(smi) > max_len:

                long_smi.append(entry)

            # create mol
            mol = ReadMol(smi, verbose=False)

            # Standardise
            issue, std_smi, parent_smi = Standardize (mol)

            issues.append(issue)
            std_smiles.append(std_smi)
            parent_smiles.append(parent_smi)

            # select molecules with score issue from checker greater than 5 as in the original publication
            issue_threshold = 5
            if remove_issues and issue and issue[0][0]>issue_threshold:
                error_mol.append(entry)

        # add columns for standard smiles and Parent Smiles to the original DB
        DB ["Std_SMILES"]    = pd.DataFrame(data=std_smiles)
        DB ["Parent_SMILES"] = pd.DataFrame(data=parent_smiles)

        # CLEANING DB:
        if long_smi and error_mol:
            # remove molecules longer than max_len characters and issues from checker
            print ("Removing %d molecules with SMILES longer than %d characters \nRemoving %d molecules with issue scores from Checker higher than %s ..." %(len(long_smi), max_len, len(error_mol), issue_threshold))
            DB.drop(long_smi + error_mol, inplace=True)

        if long_smi and not error_mol:
            # remove molecules longer than max_len characters
            print ("Removing %d molecules with SMILES longer than %d characters ..." %(len(long_smi), max_len))
            DB.drop(long_smi, inplace=True)

        if error_mol and not long_smi:
            # remove error molecules found by the Checker
            print ("Removing %d molecules with issue scores from Checker higher than 5 ..." %(len(error_mol)))
            DB.drop(error_mol, inplace=True)

        if remove_duplicates:
            # remove duplicates according to the Parent SMILES
            print ("Removing %d duplicates ... " %DB.duplicated(subset=smile_field).sum())
            DB.drop_duplicates(subset=['Parent_SMILES'], inplace=True)

    # reset index after removing duplicates
    DB.reset_index(inplace=True, drop=True)
    # save file
    DB.to_csv(output, sep = sep)
    print ("All Done!")

    return DB

def CalcDesc_DB (DB, smile_field = "Parent_SMILES", output = "desc_out.csv", sep = ",", Mordred=True, RDKit=True, pybel=True):
    """
    Function to calculate descriptors of a database of molecules collected in a pandas df (suggested to be performed after ImportDB function)

    :param DB: pandas dataframe collecting data
    :param smile_field: file of the dataset containing the smiles of the compounds to be processed
    :param output: output filename of the outputfile
    :param sep: separator character which divides entries in the csv
    :param Mordred: if True calculate Mordred descriptors, if False not
    :param RDKit: if True calculate RDKit descriptors, if False not
    :param pybel: if True calculate pybel descriptors, if False not

    :return: dataset with relative molecular descriptors dataset in pandas and in csv format accoring to the defined parameters
    """
    # create empty list for descriptors values
    desc_values = []

    # create progress bar
    p_bar = tqdm(range(len(DB)))

    # loop over molecules in starting DB
    for entry in p_bar:

        # isolate smi from the DB
        smi = DB[smile_field].iloc[entry]

        # calcolate descriptors
        names, values    = CalcDesc(smi, Mordred=Mordred, RDKit=RDKit, pybel=pybel)

        # append descriptors values
        desc_values.append(values)

    # saving into pandas df
    DB_desc = pd.DataFrame(data = desc_values,   columns=names)
    DB_desc_complete = pd.concat([DB, DB_desc], axis=1)

    # save file
    DB_desc_complete.to_csv(output, sep = sep)

    return DB_desc_complete



# ========================================================================================
# === Section d) VERIFY THE APPLICABILITY DOMAIN OF THE MODEL ===

def DefineAD (DB, smile_field = "Parent_SMILES", output_filename = "AD.pkl", nbits=1024, radius=2):
    """
    Function to define the AD of the model

    :param DB: compounds database in a pandas dataframe
    :param smile_field: field of the pandas DB containing compounds Smiles
    :param output_filename: filename for the pickle file collecting the AD
    :param nbits: number of bits to compute the Morgan Fingerprint using RDKit
    :param radius: radius parameter for the Morgan Fingerprint Calculation

    :return: pandas dataframe and csv with smiles and relative fingerprints
    """
    # create empty list for fingerprints
    fps = []

    # loop over molecules in starting DB
    for entry in range(len(DB)):

        # isolate smi from the DB
        smi = DB[smile_field].iloc[entry]

        fps.append(Calc_fps(smi, nbits=nbits, radius=radius))

    # Store the applicability domain (list of fingerprints and relative smiles) into a pandas df and a pickle file:
    smiles_and_fps = pd.DataFrame()
    smiles_and_fps["Smiles"] = DB[smile_field]
    smiles_and_fps["fps"] = fps
    smiles_and_fps.to_pickle(output_filename)

    return smiles_and_fps

def TestAD (query_smile, filename="AD.pkl", metric = "tanimoto", neighbors = 5, sim_threshold = 0.1, verbose = True, nbits=1024, radius=2):
    """
    Function to test if a query compound is inside a pre-defined Applicability Domain (AD)

    :param query_smile: query smiles to be compared with the defined AD
    :param filename: filename of the pickle file cointaining the AD of interest
    :param neighbors: number of neighbors to calculate the similarity using the Tanimoto Score (Default: 5 compounds)
    :param sim_threshold: threshold applied to the average Tanimoto score between the query compound and the 5 most similar (default: 0.1 as in e-Sweet)
    :param verbose: if True, it gives the output message reporting if the query compound is inside/outsite the AD
    :param nbits: number of bits to compute the Morgan Fingerprint using RDKit
    :param radius: radius parameter for the Morgan Fingerprint Calculation

    :return: True/False (compound inside or outside the AD), similarity_score (average similarity scores of the query compound), similarity_smiles (5 most similar smiles in the AD)
    """

    # standardize the smile
    issues, std_smi, parent_smi = Standardize(ReadMol(query_smile, verbose=False))
    # calculate Morgan Fingerprint
    fp_query = Calc_fps(parent_smi, nbits=nbits, radius=radius)

    # Open Applicability Domain in pandas
    AD = pd.read_pickle(filename)

    # calculate similarities between the query compound and all the other compounds in the dataset using fingerprints
    if metric == "tanimoto":
        AD["similarity"] = DataStructs.BulkTanimotoSimilarity(fp_query, AD["fps"])
    elif metric == "dice":
        AD["similarity"] = DataStructs.BulkDiceSimilarity(fp_query, AD["fps"])
    elif metric == "cosine":
        AD["similarity"] = DataStructs.BulkCosineSimilarity(fp_query, AD["fps"])
    else:
        print ("Metric %s is not a valid similarity metric: choose tanimoto, dice or cosine!\n" %metric)

    # the similarity score is the average similiraty between the query compound and the five closest neighbors (defined with the Tanimoto Similarity)
    similarity_score  = np.mean(AD.sort_values(by=["similarity"], ascending=False).head(neighbors)["similarity"])
    similarity_smiles = AD.sort_values(by=["similarity"], ascending=False).head(neighbors)["Smiles"]

    # if similarity > threashold --> the compound is inside the applicability domain!
    if similarity_score >sim_threshold:
        if verbose:
            print ("OK: The query compound is inside the Applicability Domain!")
        return True, similarity_score, np.array(similarity_smiles)
    else:
        if verbose:
            print ("*** WARNING ***\nThe query compound is OUTSIDE the Applicability Domain!")
        return False, similarity_score, np.array(similarity_smiles)

    
    
def PredictExplain(sample_df, savefig=False):
    ## Load data
    with open('../data/comb.pickle', 'rb') as handle:
        data = pickle.load(handle)
    df, metadata, features, target, rows = data.values()
    
    ## Load models and explainers
    with open('../models/models.pickle', 'rb') as handle:
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
    ## Plotting
    FEATS_TO_DISPLAY = 10

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
    
    pred_prob = shap_values_tmp.base_values + shap_values_tmp.values.sum()
    pred_label = 'Sweet' if pred_prob >= 0.5 else 'Bitter'
    if pred_label == 'Sweet':
        plt.suptitle(fr'$\bfSHAP\,\,values\,\,for:\,\,$ {sample_df.SMILES.values[0]}''\n'fr'$\bfPrediction:\,\,$ {pred_label} with {pred_prob.round(3)} probability',
                     fontsize=14, ha='center')
    else:
        plt.suptitle(fr'$\bfSHAP\,\,values\,\,for:\,\,$ {sample_df.SMILES.values[0]}''\n'fr'$\bfPrediction:\,\,$ {pred_label} with {1-pred_prob.round(3)} probability',
                     fontsize=14, ha='center');
    if savefig:
        PATH = os.path.join('..', 'images')
        Path(PATH).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(PATH, f'SHAP_{sample_df.SMILES.values[0]}.png'), dpi=fig.dpi)
