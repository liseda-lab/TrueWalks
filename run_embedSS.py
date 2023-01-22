import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import os
import sys
from scipy.stats import rankdata

from configuration import FILE_DATASET,EMBEDDING_FILES,PATH_OUTPUT_SS


def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def process_dataset(file_dataset_path):
    """
    Process the dataset file and returns a list with the pairs of entities, a list of labels, and a list of entities
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1\tEnt2\tLabel\n"
    """
    dataset = open(file_dataset_path, "r")
    pairs_ents, labels, ents = [], [], []
    for line in dataset:
        split1 = line[:-1].split('\t')
        ent1, ent2, label = split1[0], split1[1], float(split1[2])
        pairs_ents.append((ent1, ent2))
        labels.append(label)
        if ent1 not in ents:
            ents.append(ent1)
        if ent2 not in ents:
            ents.append(ent2)
    dataset.close()
    return pairs_ents, labels, ents


def process_embedding_files(list_embeddings_files, ents):
    """
    Compute cosine similarity between embeddings for each posssible pair and sabe them in a dic_SS = {(ent1,ent2):0.5, (ent3,ent2):0.8}.
    :param list_embeddings_files: list of the embeddings files (positive and negative)
    :param ents: list of entities of the dataset
    """

    list_dict = []
    for embedding_file in list_embeddings_files:
        dict_embeddings = eval(open(embedding_file, 'r').read())
        list_dict.append(dict_embeddings)

    dict_ss = {}
    for name_ent1 in ents:

        for name_ent2 in ents:

            begin = True
            for dict_embeddings in list_dict:

                ent1 = np.array(dict_embeddings[name_ent1])
                ent1 = ent1.reshape(1, len(ent1))

                ent2 = np.array(dict_embeddings[name_ent2])
                ent2 = ent2.reshape(1, len(ent2))

                if begin:
                    ent1_concat = ent1
                    ent2_concat = ent2
                    begin = False
                else:
                    ent1_concat = np.hstack((ent1_concat, ent1))
                    ent2_concat = np.hstack((ent2_concat, ent2))

            sim = cosine_similarity(ent1, ent2)[0][0]
            dict_ss[(name_ent1, name_ent2)] = sim
            
    return dict_ss


def compute_roc_auc(dict_ss, ents, pairs_ents, labels):
    """

    :param dict_ss: dictionary that stores the similarities between all possible pairs and it is in the format dic_SS = {(ent1,ent2):0.5, (ent3,ent2):0.8}
    :param ents: list of entities of the dataset
    :param pairs_ents: list of entity pairs. Each element represents a tuple (ent1, ent2)
    :param labels: list of labels. The label label[i] corresponds to the label of the pair pairs_ents[i]
    """
    labels_pred, preds = [], []
    positive_pairs = {pairs_ents[i] for i in range(len(pairs_ents)) if labels[i]==1}
    for ent1 in ents:
        for ent2 in ents:
            if ent1 != ent2:
                if (ent1,ent2) in positive_pairs:
                    labels_pred.append(1)
                    preds.append(dict_ss[(ent1,ent2)])
                else:
                    labels_pred.append(0)
                    preds.append(dict_ss[(ent1, ent2)])
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = metrics.roc_curve(labels_pred, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def compute_metrics(ents, pairs_ents, labels, dict_ss):
    """
    Computes recall at top 10, recall at top 100, roc-auc and mean rank
    :param ents: list of entities of the dataset
    :param pairs_ents: list of entity pairs. Each element represents a tuple (ent1, ent2)
    :param labels: list of labels. The label label[i] corresponds to the label of the pair pairs_ents[i]
    :param dict_ss: dictionary that stores the similarities between all possible pairs and it is in the format dic_SS = {(ent1,ent2):0.5, (ent3,ent2):0.8}
    """
    top10,top100,mean_rank = 0,0,0
    ranks = {}

    dic_index = {}
    for i in range(len(ents)):
        dic_index[ents[i]] = i

    labels_matrix = np.zeros((len(ents), len(ents)), dtype=np.int32)
    sim_matrix = np.zeros((len(ents), len(ents)), dtype=np.float32)
    for i1 in range(len(ents)):
        ent1 = ents[i1]
        for i2 in range(len(ents)):
            ent2 = ents[i2]
            sim_matrix[i1, i2] = dict_ss[(ent1,ent2)]

    for i in range(len(pairs_ents)):
        e1,e2 = pairs_ents[i]
        label = labels[i]
        pairs_ents.append((e2,e1))
        labels.append(label)

    for i in range(len(pairs_ents)):
        e1,e2 = pairs_ents[i]
        i1 = dic_index[e1]
        i2 = dic_index[e2]
        if labels[i]==1:
            labels_matrix[i1,i2] =1

            index = rankdata(-sim_matrix[i1, :], method='average')
            rank = index[i2]
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

    n = len(pairs_ents)
    top10 /= n
    top100 /= n
    mean_rank /= n
    roc_auc = compute_roc_auc(dict_ss, ents, pairs_ents, labels)
    return top10, top100, mean_rank, roc_auc


def run_sim_metrics(list_embeddings_files, file_dataset_path, path_results):
    """
    Uses cosine similarity to estimate similarity between entities and computes metrics.
    :param list_embeddings_files: list of the embeddings files (positive and negative)
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1\tEnt2\tLabel\n"
    :param path_results: folder where metrics are stored
    """
    pairs_ents, labels, ents = process_dataset(file_dataset_path)
    dict_ss = process_embedding_files(list_embeddings_files, ents)

    with open(path_results + "PerformanceMetrics.txt", "w") as file_sim:
        file_sim.write("Hits@10\tHits@100\tMeanRank\tROCAUC\n")
        top10, top100, mean_rank, roc_auc = compute_metrics(ents, pairs_ents, labels, dict_ss)
        file_sim.write(str(top10) + "\t" + str(top100) + "\t" + str(mean_rank) + "\t" + str(roc_auc) + "\n")


if __name__ == "__main__":

    file_dataset_path = FILE_DATASET
    list_embedding_files = EMBEDDING_FILES
    path_results = PATH_OUTPUT_SS
    run_sim_metrics(list_embedding_files, file_dataset_path, path_results)