import numpy as np
import pandas as pd
import os
import sys
from statistics import mean, median

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit

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
    Process the dataset file and returns a list with the pairs of entities and a list of labels.
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1\tEnt2\tLabel\n"
    """
    dataset = open(file_dataset_path, "r")
    pairs_ents, labels = [], []
    for line in dataset:
        split1 = line[:-1].split('\t')
        ent1, ent2, label = split1[0], split1[1], float(split1[2])
        pairs_ents.append([ent1, ent2])
        labels.append(label)
    dataset.close()
    return pairs_ents, labels


def process_embedding_files(pairs_ents, list_embeddings_files, operation):
    """
    Compute pair representation using a operator.
    :param pairs_ents: list of entity pairs. Each element represents a tuple (ent1, ent2)
    :param list_embeddings_files: list of the embeddings files (positive and negative)
    :param operation: operator that will be used to combine the embeddings of the pair
    """
    list_dict = []
    for embedding_file in list_embeddings_files:
        dict_embeddings = eval(open(embedding_file, "r").read())
        list_dict.append(dict_embeddings)

    list_emb_op = []
    for name_ent1, name_ent2 in pairs_ents:
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
        if operation == "concat":
            emb_op = np.hstack((ent1_concat, ent2_concat))
        elif operation == "avg":
            emb_op = 0.5 * ent1_concat + 0.5 * ent2_concat
        elif operation == "hada":
            emb_op = np.multiply(ent1_concat, ent2_concat)
        elif operation == "wl1":
            emb_op = np.absolute(np.subtract(ent1_concat, ent2_concat))
        elif operation == "wl2":
            emb_op = np.power(np.absolute(np.subtract(ent1_concat, ent2_concat)), 2)

        list_emb_op.append(emb_op.tolist()[0])
    return list_emb_op


def run_MonteCarlopartitions(pairs_ents, labels, path_partition, n_partition, test_size):
    """
    Split the dataset n_partition times and save the test set indexes for each split.
    :param pairs_ents: list of entity pairs. Each element represents a tuple (ent1, ent2)
    :param labels: list of labels. The label label[i] corresponds to the label of the pair pairs_ents[i]
    :param path_partition: folder where the split indexes will be stored
    :param n_partition: number of splits
    :param test_size: fraction of the dataset that will correspond to the test set
    """
    n_pairs = len(labels)
    for run in range(n_partition):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for index_train, index_test in sss.split(pairs_ents, labels):
            with open(path_partition + str(run + 1) + ".txt", "w") as file_partition:
                for index in index_test:
                    file_partition.write(str(index) + "\n")


def process_indexes_partition(file_partition):
    """
    Process the split indexes file and returns a list of indexes.
    :param file_partition: partition file path (each line is a index);
    """
    indexes_partition = []
    with open(file_partition, "r") as partition:
        for line in partition:
            indexes_partition.append(int(line[:-1]))
    return indexes_partition


def predictions(preds, y):
    """
    Get the predictions and expected results to calculate the weighted average f-measure, recall and precision.
    :param preds: list of predictions (predicted values)
    :param y: list of labels (expected values). The expect value of the prediction preds[i] is y[i]
    """
    waf = metrics.f1_score(y, preds, average="weighted")
    precision = metrics.precision_score(y, preds)
    recall = metrics.recall_score(y, preds)
    return waf, precision, recall


def writePredictions(preds, y, path_output):
    """
    Write the predictions to a file.
    :param predictions: list of predictions (predicted values)
    :param y: list of labels (expected values). The expect value of the prediction preds[i] is y[i]
    :param path_output: file path where the predictions will be saved. Each line represents a prediction "predicted_value\texpected_value\n"
    """
    with open(path_output, "w") as file_predictions:
        file_predictions.write("Predicted_output" + "\t" + "Expected_Output" + "\n")
        for i in range(len(y)):
            file_predictions.write(str(preds[i]) + "\t" + str(y[i]) + "\n")


def run_RF(X_train, X_test, y_train, y_test, path_output_predictions):
    """
    Applies Random Forest Algorithm.
    :param X_train: the training input samples. The shape of the list is (n_samplesTrain, embedding_pair_size)
    :param X_test: the testing input samples. The shape of the list is (n_samplesTest, embedding_pair_size)
    :param y_train: labels of the training set. The shape of the list is (n_samplesTrain)
    :param y_test: labels of the test set. The shape of the list is (n_samplesTest)
    :param path_output_predictions: file path where the predictions will be saved. Each line represents a prediction "predicted_value\texpected_value\n"
    """
    model = RandomForestClassifier()
    clf = GridSearchCV(model, {'max_depth': [2, 4, 6, None], 'n_estimators': [50, 100, 200]})
    clf.fit(X_train, y_train)
    preds_test = clf.predict(X_test)
    preds_train = clf.predict(X_train)
    writePredictions(preds_train, y_train, path_output_predictions + '_TrainSet')
    writePredictions(preds_test, y_test, path_output_predictions + '_TestSet')
    return predictions(preds_test, y_test)


def run_cross_validation(list_embeddings_files, file_dataset_path, operation, path_results, path_partition, n_partition=30, test_size=0.3, make_partitions=False):
    """

    :param list_embeddings_files: list of the embeddings files (positive and negative)
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1\tEnt2\tLabel\n";
    :param operation: operator that will be used to combine the embeddings of the pair
    :param path_results: folder where predictions and metrics are stored
    :param path_partition: folder where the split indexes are stored
    :param n_partition: number of splits
    :param test_size: fraction of the dataset that will correspond to the test set
    :param make_partitions: boolean. If True splits are generated. If False are used already used splits
    """
    pairs_ents, labels = process_dataset(file_dataset_path)
    list_emb_op = process_embedding_files(pairs_ents, list_embeddings_files, operation)

    if make_partitions:
        run_MonteCarlopartitions(pairs_ents, labels, path_partition, n_partition, test_size)

    file_ML = open(path_results + "/" + "PerformanceResults.txt", 'w')
    file_ML.write("Run\tWAF\tPrecision\tRecall\n")
    list_results = []

    n_pairs = len(labels)
    for run in range(1, n_partition + 1):

        file_partition = path_partition + str(run) + '.txt'
        test_index = process_indexes_partition(file_partition)
        train_index = list(set(range(0, n_pairs)) - set(test_index))

        array_labels = np.array(labels)
        y_train, y_test = array_labels[train_index], array_labels[test_index]
        y_train, y_test = list(y_train), list(y_test)

        array_emb_op = np.array(list_emb_op)
        X_train, X_test = array_emb_op[train_index], array_emb_op[test_index]
        X_train, X_test, = list(X_train), list(X_test)


        path_output_predictions = path_results + "/Predictions__Run" + str(run)
        waf, precision, recall = run_RF(X_train, X_test, y_train, y_test, path_output_predictions)
        file_ML.write(str(run) + "\t" + str(waf) + "\t" + str(precision) + "\t" + str(recall) + "\n")
        list_results.append([waf, precision, recall])

    wafs = [i[0] for i in list_results]
    precisions = [i[1] for i in list_results]
    recalls = [i[2] for i in list_results]

    with open(path_results + "/PerformanceResults(median).txt", "w") as file_algorithm_median:
        file_algorithm_median.write("WAF\tPrecision\tRecall\n")
        file_algorithm_median.write(str(median(wafs)) + "\t" + str(median(precisions)) + "\t" + str(median(recalls)) + "\n")

    file_ML.close()


            
if __name__ == "__main__":


    #################################### PPInegAnnot_datasets v2 ####################################
    file_dataset_path = "Data/PPI/pairs_PPIdataset.txt"
    operation = "hada"
    list_embedding_files = ["Embeddings/PPI/Emb100_TrueWalks_pos.txt", "Embeddings/PPI/Emb100_TrueWalks_neg.txt"]
    path_results = "Results/PPI-ML"
    path_partition = "Results/PPI-ML/IndexTest_MCCV_Run"
    n_partition = 30
    test_size = 0.3
    run_cross_validation(list_embedding_files, file_dataset_path, operation, path_results, path_partition, n_partition, test_size, True)



