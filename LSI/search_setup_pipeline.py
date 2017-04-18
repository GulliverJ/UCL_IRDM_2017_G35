# ==== IMPORTS ====

# Code to create the inverted index
from HTML2Index import HTML2Index

# Code to create the term-document matrix
from TDMatCreator import TDMatCreator

# Code to setup the latent semantic indexer
from LatentSemanticIndexer import LatentSemanticIndexer

# For timing code
import time

# ==================

# ==== PATHS ====
# Here are specified all paths necessary to run the pipeline

# The base directory of the project on the system
BASE_DIR = "/Users/edcollins/Documents/CS/4thYear/IRDM/CW/Code/"

# The path to the web pages stored in JSON format
PATH_TO_STORED_WEBPAGES_AS_JSON = BASE_DIR + "LSI/pages/"

# Where to save the inverted index
PATH_TO_SAVE_INVERTED_INDEX = BASE_DIR + "LSI/"

# The full path to the inverted index
FULL_PATH_TO_SAVED_INVERTED_INDEX = PATH_TO_SAVE_INVERTED_INDEX + "inverted_index.pkl"

# The location in which to save the term-document matrix
PATH_TO_SAVE_TERM_DOCUMENT_MATRIX = BASE_DIR + "LSI/Term_Document_Matrix/"

# Path to the mapping from filenames to URLs
PATH_TO_FILENAME2URL_MAPPING = BASE_DIR + "LSI/filename2url.pkl"

# The path to save the ranking matricies
PATH_TO_SAVE_RANKING_MATRICIES = BASE_DIR + "LSI/SVD/"

# ==== MAIN PIPELINE =====

print("\n\n====> BEGINNING SEARCH SETUP <====")
start_time = time.time()

# Create the index
print("\n----> Creating and saving the inverted index...")
index_start_time = time.time()
index_creator = HTML2Index(PATH_TO_STORED_WEBPAGES_AS_JSON)
index_creator.save_index(PATH_TO_SAVE_INVERTED_INDEX)
print("----> Done. Took ", time.time() - index_start_time, " seconds.")

# Create the term-document matrix
print("\n----> Creating and saving the term-document matrix...")
td_start_time = time.time()
tdmat_creator = TDMatCreator(FULL_PATH_TO_SAVED_INVERTED_INDEX, filename2url_path=PATH_TO_FILENAME2URL_MAPPING)
tdmat_creator.save_objects(PATH_TO_SAVE_TERM_DOCUMENT_MATRIX)
print("----> Done. Took ", time.time() - td_start_time, " seconds.")

# Create the latent semantic indexer
print("\n----> Creating the latent semantic indexer...")
lsi_start_time = time.time()

lsi = LatentSemanticIndexer(
    path_to_td_matrix=PATH_TO_SAVE_TERM_DOCUMENT_MATRIX,
    path_to_inverted_index=FULL_PATH_TO_SAVED_INVERTED_INDEX,
    path_to_save_ranking_matricies=PATH_TO_SAVE_RANKING_MATRICIES,
    path_to_filename2url_mapping=PATH_TO_FILENAME2URL_MAPPING,
    path_to_crawled_files=PATH_TO_STORED_WEBPAGES_AS_JSON,
    searching=False
)

print("----> Done. Took ", time.time() - lsi_start_time, " seconds.")

print("\n====> PIPELINE COMPLETE, TOTAL TIME ", (time.time() - start_time) / 60, " MINUTES <====")