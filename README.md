# VirtuousSweetBitter

The **VirtuosSweetBitter** tool predict the sweet/bitter taste of query molecules based on their molecular structures. This tool was developed within the Virtuous Project (https://virtuoush2020.com/)

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/


### Repo Structure
The repository is organized in the following folders:

- scripts/
>Collecting python codes and sources files 

- data/
> Collecting datasets of the work and related figures

- notebook/
> Collecting jupyter notebooks to explain the functionality of the VirtuousSweetBitter code and the processing steps followed in the work development


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

        pip install -r requiremets.txt
        conda install -c conda-forge rdkit chembl_structure_pipeline openbabel
        conda install -c mordred-descriptor mordred

Enjoy! 

------------------
## Acknowledgement
------------------

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).
