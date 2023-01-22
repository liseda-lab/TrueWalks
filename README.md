# TrueWalks: Biomedical Knowledge Graph Embeddings with Negative Statements

## Overview

An overview of TrueWalks, the method we propose, is shown below. The first step is the transformation of the KG into an RDF Graph. Next, our novel random walk generation strategy that is aware of positive and negative statements is applied to the graph to produce a set of entity sequences. The positive and negative entity walks are fed to neural language models to learn a dual latent representation of the entities. TrueWalks has two variants: one that employs the classical skip-gram model to learn the embeddings (TrueWalks), and one that employs a variation of skip-gram that is aware of the order of entities in the walk (TrueWalksOA, i.e. order-aware). 

<img src="https://github.com/liseda-lab/TrueWalks/blob/main/TrueWalks.png" width="450"/>


## Pre-requesites
* install python 3.6.8;
* install python libraries by running the following command:  ```pip install -r req.txt```.


## Usage

Run the command:
```
python3 run_TrueWalksEmbeddings.py  
```

This command will create an save TrueWalks embeddings. It can be configured by the [configuration file](https://github.com/liseda-lab/TrueWalks/blob/main/configuration.py).


## Evaluation and Datasets

TrueWalks was evaluated on two biomedical tasks: protein-protein interaction (PPI) prediction and gene-disease association (GDA) prediction. The data is available in [Data](https://github.com/liseda-lab/TrueWalks/blob/main/Data) folder. Since the ontology files are too big, they should be downloaded and used:
* [Gene Ontology](http://release.geneontology.org/2021-09-01/ontology/index.html)
* [Human Phenotype Ontology](https://hpo.jax.org/app/data/ontology)

To use TrueWalks embeddings as input to a random forest classifier, run the command:
```
python3 run_embedML.py  
```

To use TrueWalks embeddings to compute similarity, run the command
```
python3 run_embedSS.py  
```
