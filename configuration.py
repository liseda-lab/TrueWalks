#number of dimensions of the embeddings
EMBEDDING_SIZE = 100

#maximum number of walks per entity
N_WALKS = 100

#maximum depth of walks
WALK_DEPTH = 4

#options are "word2vec" e "wang2vec" (order-aware)
LANGUAGE_MODEL = "word2vec"

#folder where the embeddings, walks and models files will be saved
PATH_OUTPUT_EMBEDDING = "Embeddings/PPI/"

#path file with one entity per line
PATH_ENTITY_FILE = "Data/PPI/Prots.txt"

#ontology file path in the OWL file
ONTOLOGY_FILE_PATH = "Data/GO/go.owl"

#annotations file path in the tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation
ANNOTATIONS_FILE_PATH = "Data/GO/GO_annotations.tsv"


##### Evaluation 

# dataset file path, where each line of the dataset files is in the format "Ent1\tEnt2\tLabel\n" 
FILE_DATASET = "Data/PPI/pairs_PPIdataset.txt"

#list of the embeddings files (one file for the embeddings generated with positive statements and the other for the embeddings generated with negative statements)
EMBEDDING_FILES = ["Embeddings/PPI/Emb100_TrueWalks_pos.txt", "Embeddings/PPI/Emb100_TrueWalks_neg.txt"]

#operator to combine the embeddings of a pair
OPERATION = "hada"

#folder where predictions and metrics are saved
PATH_OUTPUT_ML = "Results/PPI-ML"

#folder where the MCCV split indexes are saved
PATH_PARTITION_ML = "Results/PPI-ML/IndexTest_MCCV_Run"

#number of splits for MCCV
N_PARTITIONS = 30

#fraction of the dataset that will correspond to the test set in the MCCV
TEST_SIZE = 0.3

#folder where similarity metrics are saved
PATH_OUTPUT_SS = "Results/PPI-sim/"
