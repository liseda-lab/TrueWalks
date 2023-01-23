##### TrueWalk Embeddings Generation
##### TrueWalks takes as input an ontology file, an instance annotation file, and an entity file to generate the knowledge graph. 

#ontology_file_path: str, ontology file in the OWL format 
ONTOLOGY_FILE_PATH = "Data/GO/go.owl"

#annotations_file_path: str, annotations file path in the tsv format with one entity per line and with '+' and '-' to indicate if it is a positive or negative annotation
ANNOTATIONS_FILE_PATH = "Data/GO/GO_annotations.tsv"

#path_entity_file: str, file with the entities for which we want to generate the embeddings (each line represents one entity and it is in the format 'Ent\n')
PATH_ENTITY_FILE = "Data/PPI/Prots.txt"

#embedding_size: int, dimensionality of the node embeddings
EMBEDDING_SIZE = 100

#n_walks: int, maximum number of walks per entity
N_WALKS = 100

#walk_depth: int, maximum depth of walks
WALK_DEPTH = 4

#language_model: {"word2vec","wang2vec"}, language model used to learn the embeddings
LANGUAGE_MODEL = "word2vec"

#path_output_embedding: str, folder where the embeddings, walks and models files are saved
PATH_OUTPUT_EMBEDDING = "Embeddings/PPI/"


##### TrueWalk Embeddings Evaluation
##### TrueWalk embeddings can be used as features for a random forest classifier and directly for similarity-based prediction.

#file_dataset: str, dataset file, where each line is in the format "Ent1\tEnt2\tLabel\n" 
FILE_DATASET = "Data/PPI/pairs_PPIdataset.txt"

#embedding_files: [str, str], list of the embeddings files (one file for the embeddings generated with positive statements and the other for the embeddings generated with negative statements)
EMBEDDING_FILES = ["Embeddings/PPI/Emb100_TrueWalks_pos.txt", "Embeddings/PPI/Emb100_TrueWalks_neg.txt"]

#operation: {"concat", "avg", "hada", "wl1", "wl2"}, operator to combine the embeddings of a pair.
OPERATION = "hada"

#path_output_ml: str, folder where predictions and metrics are saved
PATH_OUTPUT_ML = "Results/PPI-ML"

#path_partition_ml: folder where the MCCV split indexes are saved
PATH_PARTITION_ML = "Results/PPI-ML/IndexTest_MCCV_Run"

#n_partitions: int, number of splits for MCCV
N_PARTITIONS = 30

#test_size: float, fraction of the dataset that corresponds to the test set in the MCCV
TEST_SIZE = 0.3

#path_output_ss: str, folder where similarity metrics are saved
PATH_OUTPUT_SS = "Results/PPI-sim/"
