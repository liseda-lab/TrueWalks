#Number of dimensions of the embeddings
EMBEDDING_SIZE = 100

#Maximum number of walks per entity
N_WALKS = 100

#Maximum depth of walks
WALK_DEPTH = 4

#Options are "word2vec" e "wang2vec" (order-aware)
LANGUAGE_MODEL = "word2vec"

#Folder where the embeddings, walks and models files will be saved
PATH_OUTPUT_EMBEDDING = "Embeddings/PPI/"

#Path file with one entity per line
PATH_ENTITY_FILE = "Data/PPI/Prots.txt"

#Ontology file path in the OWL file
ONTOLOGY_FILE_PATH = "Data/GO/go.owl"

#Annotations file path in the tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation
ANNOTATIONS_FILE_PATH = "Data/GO/GO_annotations.tsv"


##### Evaluation 

#Dataset file path, where each line of the dataset files is in the format "Ent1\tEnt2\tLabel\n" 
FILE_DATASET = "Data/PPI/pairs_PPIdataset.txt"

#List of the embeddings files (one file for the embeddings generated with positive statements and the other for the embeddings generated with negative statements)
EMBEDDING_FILES = ["Embeddings/PPI/Emb100_TrueWalks_pos.txt", "Embeddings/PPI/Emb100_TrueWalks_neg.txt"]

#Operator to combine the embeddings of a pair. The options are: "concat", "avg", "hada", "wl1", "wl2"
OPERATION = "hada"

#Folder where predictions and metrics are saved
PATH_OUTPUT_ML = "Results/PPI-ML"

#Folder where the MCCV split indexes are saved
PATH_PARTITION_ML = "Results/PPI-ML/IndexTest_MCCV_Run"

#Number of splits for MCCV
N_PARTITIONS = 30

#Fraction of the dataset that will correspond to the test set in the MCCV
TEST_SIZE = 0.3

#Folder where similarity metrics are saved
PATH_OUTPUT_SS = "Results/PPI-sim/"
