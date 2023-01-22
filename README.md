# TrueWalks: Biomedical Knowledge Graph Embeddings with Negative Statements

## Overview

An overview of TrueWalks, the method we propose, is shown below. The first step is the transformation of the KG into an RDF Graph. Next, our novel random walk generation strategy that is aware of positive and negative statements is applied to the graph to produce a set of entity sequences. The positive and negative entity walks are fed to neural language models to learn a dual latent representation of the entities. TrueWalks has two variants: one that employs the classical skip-gram model to learn the embeddings (TrueWalks), and one that employs a variation of skip-gram that is aware of the order of entities in the walk (TrueWalksOA, i.e. order-aware). 

<img src="https://github.com/liseda-lab/TrueWalks/blob/main/TrueWalks.png"/>

## Pre-requesites
* install python 3.6.8;
* install python libraries by running the following command:  ```pip install -r req.txt```.

## Usage

## Datasets
