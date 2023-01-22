#number of dimensions that the generated embeddings have
EMBEDDING_SIZE = 200
#maximum number of walks per entity
N_WALKS = 100
#maximum depth of walks
WALK_DEPTH = 4
#options are "word2vec" e "wang2vec" (order-aware)
LANGUAGE_MODEL = "word2vec"
#folder where the embeddings, walks and models files will be stored
PATH_OUTPUT_EMBEDDING = "Embeddings/PPI/"
#path file where we have one entity per line
PATH_ENTITY_FILE = "Data/PPI/Prots.txt"
#owl ontology file
ONTOLOGY_FILE_PATH = "Data/GO/go.owl"
#file path in tsv format with one protein per line and with '+' and '-' to indicate if it is a positive or negative annotation
ANNOTATIONS_FILE_PATH = "Data/GO/GO_annotations.tsv"


# dataset file path. The format of each line of the dataset files is "Ent1\tEnt2\tLabel\n" 
FILE_DATASET = "Data/PPI/pairs_PPIdataset.txt"
#list of the embeddings files (positive and negative)
EMBEDDING_FILES = ["Embeddings/PPI/Emb100_TrueWalks_pos.txt", "Embeddings/PPI/Emb100_TrueWalks_neg.txt"]
#operator that will be used to combine the embeddings of the pair
OPERATION = "hada"
#folder where predictions and metrics are stored
PATH_OUTPUT_ML = "Results/PPI-ML"
#folder where the split indexes are stored
PATH_PARTITION_ML = "Results/PPI-ML/IndexTest_MCCV_Run"
#number of splits
N_PARTITIONS = 30
#fraction of the dataset that will correspond to the test set
TEST_SIZE = 0.3
#folder where similarity metrics are stored
PATH_OUTPUT_SS = "Results/PPI-sim/"