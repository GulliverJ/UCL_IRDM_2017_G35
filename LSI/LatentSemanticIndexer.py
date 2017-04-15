# ======== IMPORTS ========

# To construct the bag of words representation of each paper
from collections import defaultdict

# For reading the inverted index
import pickle

# To create the term-document matrix
import numpy as np

# For timing
import time

# For loading the Term-Document matrix and associated maps
from TDMatCreator import TDMatCreator

# =========================


class LatentSemanticIndexer:

    def __init__(self, path_to_td_matrix):
        """
        A ranker that ranks documents by their cosine similarity after transforming the document into a representation
        using Singular Value Decomposition. Queries are likewise transformed into this semantic space and their vectors
        compared to that of each document.
        :param path_to_td_matrix: the path to the Term-Document matrix needed for Latent Semantic Indexing.
        """
        self.term2rowindex, self.filename2colindex, self.vocab, self.filenames, self.term_document_matrix = \
            TDMatCreator(create=False).load_objects(path_to_td_matrix)

    def create_ranker(self):
        """
        Creates the ranker using singular value decomposition.
        :return: the ranker matricies.
        """
        pass

if __name__ == '__main__':
    lsi = LatentSemanticIndexer("LSI/Term_Document_Matrix/")