import os
import sys
import argparse

import rdflib
from rdflib.namespace import RDF, OWL, RDFS

from gensim.models.word2vec import Word2Vec as W2V
import gensim

import sampler
import kg
import walker

from configuration import EMBEDDING_SIZE,N_WALKS,WALK_DEPTH,LANGUAGE_MODEL,PATH_OUTPUT_EMBEDDING,PATH_ENTITY_FILE,ONTOLOGY_FILE_PATH,ANNOTATIONS_FILE_PATH


def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: a path-like object representing a file system path
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def process_annotations(annotations_file_path):
    """
    Process an annotations file and returns a dictionary where the key is the entities (dic_annotations = {ent1:{'pos':[annot1, annot2,...], 'neg':[annot1, annot2,...], ent2:{...}}
    :param annotations_file_path: file path in tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation
    """
    dic_annotations = {}
    with open(annotations_file_path, "r") as annotations_file:
        for line in annotations_file:
            ent, annots = line[:-1].split('\t')
            dic_annotations[ent] = {"pos":[], "neg":[]}
            list_annots = annots.replace(" ","").replace("(P)","").replace("(F)","").replace("(C)", "").split(";")
            for annot in list_annots:
                if len(annot) > 0:
                    if annot[0]=="+":
                        dic_annotations[ent]["pos"].append(annot[1:])
                    elif annot[0]=="-":
                        dic_annotations[ent]["neg"].append(annot[1:])
    return dic_annotations


def add_annotations(g, dic_annotations, ents):
    """
    Adds to the ontology rdflib graph edges corresponding to positive and negative annotations.
    :param g: rdflib graph
    :param dic_annotations: dictionary where the key is the entities (dic_annotations = {ent1:{'pos':[annot1, annot2,...], 'neg':[annot1, annot2,...], ent2:{...}}
    """
    for ent in ents:
        for a in dic_annotations[ent]["pos"]:
            g.add((rdflib.term.URIRef(ent), rdflib.term.URIRef("http://hasPositiveAnnotation"),rdflib.term.URIRef(a)))
        for a in dic_annotations[ent]["neg"]:
            g.add((rdflib.term.URIRef(ent), rdflib.term.URIRef('http://hasNegativeAnnotation'),rdflib.term.URIRef(a)))
    return g


def construct_kg(ents, ontology_file_path, annotations_file_path):
    """
    Construct an rdflib graph after processing the ontology owl file and annotations file.
    :param ontology_file_path: owl ontology file
    :param annotations_file_path: file path in tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation
    """
    dic_annotations = process_annotations(annotations_file_path)
    g = rdflib.Graph()
    g.parse(ontology_file_path, format="xml")
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        g.add((o, rdflib.term.URIRef("https://www.w3.org/2000/01/rdf-schema#superClassOf"), s))
    return add_annotations(g, dic_annotations, ents)


def write_paths(walks, path_walks):
    """
    Processes the list of walks in the graph and writes it to a file.
    :param walks: list of list, where each list corresponds to a walk in the graph and each element to a node in the graph
    :param path_walks: path file where the walks will be saved
    """
    walk_strs = []
    for _, walk in enumerate(walks):
        s = ""
        for i in range(len(walk)):
            s += f"{walk[i]} "
            if i < len(walk) - 1:
                s += " "
        walk_strs.append(s)
    with open(path_walks, "w+") as f:
        for s in walk_strs:
            f.write(s)
            f.write("\n")


def write_embeddings(embeddings, ents, path_output_embeddings):
    """
    Writes the embedding of each entity in the file.
    :param embeddings: list where each element corresponds to an embedding
    :param ents: list of entities
    :param path_output_embeddings: File where the embeddings will be stored in the format {ent1:[v1,v2,...v200], ent2:[v1,v2,...]}
    """
    with open(path_output_embeddings, "w") as embedding_file:
        embedding_file.write("{")
        first = False
        for i in range(len(ents)):
            if first:
                embedding_file.write(", '%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
            else:
                embedding_file.write("'%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
                first = True
            embedding_file.flush()
        embedding_file.write("}")


def run_wang2vec(walks_file_path, path_model_file, vector_size, ents):
    """
    Run order-aware model to generate embeddings for entities.
    :param walks_file_path: path file where the walks are saved
    :param path_model_fiel: bin file where the language model will be stored
    :param vector_size: number of dimensions that the generated embeddings have
    :param ents: list of entities
    """
    cmd = './wang2vec/word2vec -train "' + walks_file_path + '" -output "' + path_model_file + '" -type 1 -size ' + str(
        vector_size) + ' -window 5 -negative 5 -nce 0 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 5 -cap 0'
    os.system(cmd)
    model = gensim.models.KeyedVectors.load_word2vec_format(path_model_file, binary=True)
    embeddings = [model.wv.get_vector(str(entity)) for entity in ents]
    return embeddings


def run_word2vec(walks, vector_size, ents):
    """
    Run word2vec model to generate embeddings for entities.
    :param walks: list of list, where each list corresponds to a walk in the graph and each element to a node in the graph
    :param vector_size: number of dimensions that the generated embeddings have
    :param ents: list of entities
    """
    corpus = [list(map(str, x)) for x in walks]
    model = W2V(corpus)
    embeddings = [model.wv.get_vector(str(entity)) for entity in ents]
    return embeddings


def compute_embeddings(g, ents, path_output, vector_size, n_walks, walk_depth, language_model):
    """
    Compute TrueWalks embeddings for entities and save them to a file.
    :param g: rdflib graph that integrates the ontology and annotations
    :param ents: list of entities
    :param path_output: folder where the embeddings, walks and models files will be stored
    :param vector_size: number of dimensions that the generated embeddings have
    :param n_walks: maximum number of walks per entity
    :param walk_depth: maximum depth of walks
    :param language_model: options are "word2vec" e "wang2vec" (order-aware)
    """
    graph = kg.rdflib_to_kg(g)
    sg_value = 1

    emb_sampler = sampler.Sampler()
    emb_walker = walker.Walker(depth=walk_depth, walks_per_graph=n_walks, sampler = emb_sampler)

    positive_walks, negative_walks = emb_walker.extract(graph, ents)
    write_paths(positive_walks, path_output + "PositivePaths.txt")
    write_paths(negative_walks, path_output + "NegativePaths.txt")

    if language_model == "word2vec":

        positive_embeddings = run_word2vec(positive_walks, vector_size, ents)
        negative_embeddings = run_word2vec(negative_walks, vector_size, ents)

    elif language_model == "wang2vec":

        positive_embeddings = run_wang2vec(path_output + "PositivePaths.txt", path_output + "PositiveModel.bin", vector_size, ents)
        negative_embeddings = run_wang2vec(path_output + "NegativePaths.txt", path_output + "NegativeModel.bin", vector_size, ents)

    write_embeddings(positive_embeddings, ents, path_output + "Emb100_TrueWalks_pos.txt")
    write_embeddings(negative_embeddings, ents, path_output + "Emb100_TrueWalks_neg.txt")


def run_TrueWalks(ontology_file_path, annotations_file_path, path_entity_file, path_output, vector_size=100, n_walks=100, walk_depth=4, language_model="word2vec"):
    """
    Build the rdflib graph from the ontology and annotations. Then compute TrueWalks embeddings for entities and save them to a file.
    :param ontology_file_path: owl ontology file
    :param annotations_file_path: file path in tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation.
    :param path_output: folder where the embeddings, walks and models files will be stored
    :param vector_size: number of dimensions that the generated embeddings have
    :param n_walks: maximum number of walks per entity
    :param walk_depth: maximum depth of walks
    :param language_model: options are "word2vec" e "wang2vec" (order-aware)
    :param path_entity_file: path file where we have one entity per line
    """
    ents = [line.strip() for line in open(path_entity_file).readlines()]
    g = construct_kg(ents, ontology_file_path, annotations_file_path)
    compute_embeddings(g, ents, path_output, vector_size, n_walks, walk_depth, language_model)


if __name__== '__main__':

    ####################################
    vector_size = EMBEDDING_SIZE
    n_walks = N_WALKS
    walk_depth = WALK_DEPTH
    language_model = LANGUAGE_MODEL
    path_output = PATH_OUTPUT_EMBEDDING
    path_entity_file = PATH_ENTITY_FILE
    ontology_file_path = ONTOLOGY_FILE_PATH
    annotations_file_path = ANNOTATIONS_FILE_PATH
    run_TrueWalks(ontology_file_path, annotations_file_path, path_entity_file, path_output, vector_size, n_walks, walk_depth, language_model)



