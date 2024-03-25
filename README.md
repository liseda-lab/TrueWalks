# TrueWalks: Biomedical Knowledge Graph Embeddings with Negative Statements

This repository provides an implementation described in the paper: https://link.springer.com/chapter/10.1007/978-3-031-47240-4_23.

## Overview

An overview of TrueWalks, a method to learn Knowledge Graph (KG) embeddings taking into account negative statements is shown below. The first step is the transformation of the KG into an RDF Graph. Next, our novel random walk generation strategy that is aware of positive and negative statements is applied to the graph to produce a set of entity sequences. The positive and negative entity walks are fed to neural language models to learn a dual latent representation of the entities. TrueWalks has two variants: one that employs the classical skip-gram model to learn the embeddings (TrueWalks), and one that employs a variation of skip-gram that is aware of the order of entities in the walk (TrueWalksOA, i.e. order-aware). 

<img src="https://github.com/liseda-lab/TrueWalks/blob/main/TrueWalks.png" width="450"/>


## Pre-requisites
* install python 3.6.8;
* install python libraries by running the following command:  ```pip install -r req.txt```.


## Usage

Run the command:
```
python3 run_TrueWalksEmbeddings.py  
```

This command will create and output TrueWalks embeddings to a file (the description of the embedding file is in [Embeddings_format.txt](https://github.com/liseda-lab/TrueWalks/blob/main/Embeddings_format.txt)). The parameters and input files can be changed in the [configuration file](https://github.com/liseda-lab/TrueWalks/blob/main/configuration.py).


## Evaluation and Datasets

TrueWalks was evaluated on two biomedical tasks: protein-protein interaction (PPI) prediction and gene-disease association (GDA) prediction. The data to build the KG and subsequiently train the ML approaches or compute similarities can be found in [Data](https://github.com/liseda-lab/TrueWalks/blob/main/Data) except for the Gene Ontology data, which can be downloaded [here](http://release.geneontology.org/2021-09-01/ontology/index.html).

To use TrueWalks embeddings as input to a random forest classifier, run the command:
```
python3 run_embedML.py  
```

To use TrueWalks embeddings to compute similarity, run the command:
```
python3 run_embedSS.py  
```


## How to Cite

```
@inproceedings{10.1007/978-3-031-47240-4_23,
author = {Sousa, Rita T. and Silva, Sara and Paulheim, Heiko and Pesquita, Catia},
title = {Biomedical Knowledge Graph Embeddings with&nbsp;Negative Statements},
year = {2023},
isbn = {978-3-031-47239-8},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-47240-4_23},
doi = {10.1007/978-3-031-47240-4_23},
booktitle = {The Semantic Web – ISWC 2023: 22nd International Semantic Web Conference, Athens, Greece, November 6–10, 2023, Proceedings, Part I},
pages = {428–446},
numpages = {19},
keywords = {Biomedical Applications, Negative Statements, Knowledge Graph Embedding, Knowledge Graph},
location = {Athens, Greece}
}
```
