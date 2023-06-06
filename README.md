# VirtuousSweetBitter

The **VirtuosSweetBitter** tool predict the sweet/bitter taste of query molecules based on their molecular structures. 
>Maroni, G., Pallante, L., Di Benedetto, G., Deriu, M. A., Piga, D., & Grasso, G. (2022). Informed classification of sweeteners/bitterants compounds via explainable machine learning. Current Research in Food Science, 5, 2270–2280. https://doi.org/10.1016/j.crfs.2022.11.014

VirtuousSweetBitter is also implemented into a web-service interface at https://virtuous.isi.gr/#/sweetbitter

This tool was developed within the Virtuous Project (https://virtuoush2020.com/)

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/


----------------
### Repo Structure
The repository is organized in the following folders:

- notebook/
> - ***"VirtuousSweetBitter.ipynb"*** allows to test the developed model by predicting the bitter/sweet taste for a compound of interest from its SMILES representation. Each of the steps needed for the predictions is commented and briefly explained. 
> - ***Notebooks 0-6*** provide the steps followed in the model implementation in order to allow the users to replicate the obtained results.

- scripts/
>Collecting python codes and sources files 

- data/
> Collecting datasets of the work and related figures


### Authors
1. [Gabriele Maroni](https://github.com/gabribg88)
2. [Lorenzo Pallante](https://github.com/lorenzopallante)

----------------
## Prerequisites
----------------

1. Create conda environment:

        conda create -n myenv python=3.8
        conda activate myenv

2. Clone the `VirtuousSweetBitter` repository from GitHub

        git clone https://github.com/gabribg88/VirtuousSweetBitter

3. Install required packages:

        conda install -c conda-forge rdkit chembl_structure_pipeline openbabel
        conda install -c mordred-descriptor mordred
        pip install -r requirements.txt

4. Export Environment variable (please adapt according to your installation path)

        export BABEL_DATADIR=~/anaconda3/pkgs/openbabel-3.1.1-py38h3d1cf2f_4/share/openbabel/3.1.0

Enjoy! 


---------------------------------
## How to use VirtuousSweetBitter
---------------------------------

The main code is `VirtuousSweetBitter.py` within the scripts folder.

To learn how to run, just type:

    python VirtuousSweetBitter.py --help

And this will print the help message of the program:

    usage: VirtuousSweetBitter.py [-h] [-c COMPOUND] [-f FILE] [-d DIRECTORY] [-v]

    VirtuousUmami: ML-based tool to predict the sweet/bitter taste

    optional arguments:
      -h, --help            show this help message and exit

      -c COMPOUND, --compound COMPOUND
                        query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)

      -f FILE, --file FILE  text file containing the query molecules

      -d DIRECTORY, --directory DIRECTORY name of the output directory

      -v, --verbose         Set verbose mode

To test the code you can submit an example txt file in the "samples" fodler (test.txt)      

The code will create a log file and an output folder containing:

    1. "best_descriptors.csv": a csv file collecting the 29 best molecular descriptors for each processed smiles on which the prediction relies
    2. "descriptors.csv": a csv file collecting all the calculated molecular descriptors for each processed smiles
    3. "predictions.csv": a csv summarising the results of the prediction

------------------
## Acknowledgement
------------------

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).
