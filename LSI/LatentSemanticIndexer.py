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

# Import the HTML parser
from HTMLParser import HTMLParser

# For computing the cosine similarity between vectors
from scipy import spatial

# For sorting the similarities when ranking
from operator import itemgetter

# For truncated SVD
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

# For sparse martricies
from scipy import sparse
from scipy.sparse.linalg import svds

# For computing similarity
from math import sqrt

# =========================


class LatentSemanticIndexer:

    def __init__(self, path_to_td_matrix, path_to_inverted_index, path_to_save_ranking_matricies,
                 path_to_filename2url_mapping, path_to_crawled_files, searching):
        """
        A ranker that ranks documents by their cosine similarity after transforming the document into a representation
        using Singular Value Decomposition. Queries are likewise transformed into this semantic space and their vectors
        compared to that of each document.
        :param path_to_td_matrix: the path to the Term-Document matrix needed for Latent Semantic Indexing.
        :param path_to_inverted_index: the path to the inverted index mapping terms to documents they occur in.
        :param path_to_save_ranking_matricies: the path to save / load the ranking matrices
        :param path_to_filename2url_mapping: path to filename2url mapping
        :param searching: True if the directory of documents is being queried for a document. If this is true,
                          singular value decomposition will not be performed and the matrix will be loaded from disk.
                          The loaded matrices will then be used to rank and retreive documents. If it is false, then
                          the term document matrix will be decomposed with singular value decomposition to create the
                          ranking matrices.
        """

        # Load the Term-to-Row Mapping, Filename-to-Column Mapping, Vocab, Filenames and Term-Document Matrix
        self.term2rowindex, self.filename2colindex, self.vocab, self.filenames, self.term_document_matrix = \
            TDMatCreator(create=False).load_objects(path_to_td_matrix)

        print("----> Is loaded matrix sparse? ", sparse.issparse(self.term_document_matrix))

        with open(path_to_inverted_index, "rb") as f:
            self.inverted_index, self.file_count = pickle.load(f)
            print("----> No. Files to Process: ", self.file_count)

        # The rank to use to approximate the Term-Document matrix
        self.rank = 100

        if not searching:

            # Create the rank-k approximation matrices
            self.term_matrix_k, self.singular_values_k, self.document_matrix_k = self.create_ranker()

            # Save the matrices
            self.save_ranker_matrices(path=path_to_save_ranking_matricies)

        if searching:

            # A HTML parser to parse documents
            self.parser = HTMLParser()

            # Turn the filename2index mapping into an index2filename mapping
            self.index2filename = {val: key for key, val in self.filename2colindex.items()}

            # Mapping from filenames to URLs
            with open(path_to_filename2url_mapping, "rb") as f:
                self.filenames2urls = pickle.load(f)

            # Load the rank-k versions of: term matrix, singular values and document matrix
            self.term_matrix_k, self.singular_values_k, self.document_matrix_k = \
                self.load_ranker_matrices(path=path_to_save_ranking_matricies)

            self.html_dir_path = path_to_crawled_files

    def save_ranker_matrices(self, path=""):
        """
        Saves the ranker matrices at the specified location.
        :param path: the location to save the matrices at.
        """
        # Save the term matrix
        with open(path + "term_matrix_k=" + str(self.rank) + ".npy", "wb") as f:
            np.save(f, self.term_matrix_k)

        # Save the singular value matrix
        with open(path + "singular_values_k=" + str(self.rank) + ".npy", "wb") as f:
            np.save(f, self.singular_values_k)

        # Save the document matrix
        with open(path + "document_matrix_k=" + str(self.rank) + ".npy", "wb") as f:
            np.save(f, self.document_matrix_k)

    def load_ranker_matrices(self, path=""):
        """
        Loads the ranker matrices at the specified location.
        :param path: the location to save the matrices at.
        :return: the rank-k term matrix, singular value matrix and document matrix.
        """
        # Save the term matrix
        with open(path + "term_matrix_k=" + str(self.rank) + ".npy", "rb") as f:
            tm_k = np.load(f)

        # Save the singular value matrix
        with open(path + "singular_values_k=" + str(self.rank) + ".npy", "rb") as f:
            sv_k = np.load(f)

        # Save the document matrix
        with open(path + "document_matrix_k=" + str(self.rank) + ".npy", "rb") as f:
            dm_k = np.load(f)

        return tm_k, sv_k, dm_k

    def create_ranker(self):
        """
        Creates the ranker using singular value decomposition.
        :return: the ranker matricies.
        """

        # Perform singular value decomposition
        print("----> Decomposing Matrix")
        t2 = time.time()
        #term_matrix, singular_values, document_matrix = self.decompose_matrix(self.term_document_matrix)
        tm_k_c, sv_k_c, dm_k_c = self.decompose_matrix_truncated(self.term_document_matrix)
        print("----> Matrix Decomposition Complete, took ", time.time() - t2, " seconds")

        # Create the rank-k approximation
        #tm_k, sv_k, dm_k = self.rank_approximate(term_matrix, singular_values, document_matrix, rank=self.rank)

        return tm_k_c, sv_k_c, dm_k_c

    def rank_approximate(self, term_matrix, singular_values, document_matrix, rank=100):
        """
        Implements rank approximation by keeping the first "rank" columns of the term and document matrices, and the
        first "rank" rows and columns of the singular value matrix.
        :param term_matrix: the full term matrix (U in SVD)
        :param singular_values: the full singular value matrix (S in SVD)
        :param document_matrix: the full document matrix (V in SVD)
        :param rank: the rank to reduce the matrices to. Defaults to 100 which was recommended by the literature.
        :return: rank-approximated term, document and singular value matricies.
        """
        # Sanity check
        rows, cols = singular_values.shape
        assert(rank < rows and rank < cols)

        # Create the rank approximation matrices
        tm_k = term_matrix[:, :rank]
        dm_k = document_matrix[:, :rank]
        sv_k = singular_values[:rank, :rank]

        return tm_k, sv_k, dm_k

    def decompose_matrix_truncated(self, sparse_td_mat):
        """
        Performs singular value decomposition on the term-document matrix (or any given matrix).
        :param td_mat: the term document matrix to decompose.
        :return: term matrix (U), singular values (S), document matrix (V) such that td_mat = US(V.transpose)
        """
        #sparse_td_mat = sparse.csc_matrix(td_mat.astype(float))

        # Use numpy to perform the singular value decomposition
        term_matrix, singular_vals, doc_matrix = svds(sparse_td_mat, k=self.rank)

        #print(term_matrix.shape, " ", singular_vals.shape, " ", doc_matrix.shape)
        #input()

        # The sparse SVD gives back V_t but we just want V
        doc_matrix = np.transpose(doc_matrix)

        # Transform the singular vals into a matrix rather than list
        tm_rows, tm_cols = term_matrix.shape
        dm_rows, dm_cols = doc_matrix.shape

        S = np.zeros((tm_cols, dm_cols), dtype=np.float32)
        S[:tm_cols, :dm_cols] = np.diag(singular_vals)

        #print(term_matrix.shape, " ", S.shape, " ", doc_matrix.shape)
        #input()

        #for i in range(S.shape[0]):
        #    for j in range(S.shape[1]):
        #        if i == j:
        #            print(S[i, j])

        #input()

        return term_matrix, S, doc_matrix

    def decompose_matrix(self, td_mat):
        """
        Performs singular value decomposition on the term-document matrix (or any given matrix).
        :param td_mat: the term document matrix to decompose.
        :return: term matrix (U), singular values (S), document matrix (V) such that td_mat = US(V.transpose)
        """

        # Use numpy to perform the singular value decomposition
        term_matrix, singular_vals, doc_matrix = np.linalg.svd(td_mat)

        for item in singular_vals:
            print(item)
        #input()

        # Transform the singular vals into a matrix rather than list
        tm_rows, tm_cols = term_matrix.shape
        dm_rows, dm_cols = doc_matrix.shape

        S = np.zeros((tm_rows, dm_cols), dtype=np.float32)
        S[:dm_rows, :dm_rows] = np.diag(singular_vals)

        return term_matrix, S, doc_matrix

    def search(self, query):
        """
        Searches the document base for documents matching the query.
        :param query: a query string.
        :return: ranked documents matching the query.
        """

        # Sanity check
        assert(type(query) is str)

        # Parse the query into a list of words
        query = self.parser.parse_query(query)

        # Print information about the query
        print("====> QUERY ANALYSIS <====")
        print("----> Are query terms indexed?")
        for word in query:
            if word in self.vocab:
                print("--> ", word, " : ", "YES")
            else:
                print("--> ", word, " : ", "YES")

        # Find the list of files which contain these query terms
        allowed_filenames = self.check_index(query)

        # Turn the query into a vector
        query_vec, query_count = self.query2vec(query)

        # Check if any of the query was in the vocab
        if query_count == 0:
            return []

        # Calculate the new query vector in the k-dimensional space
        query_vec = self.calculate_new_query_vec(query_vec)

        print("Ranking document indicies...")

        # Compare the query vector to every document to rank them
        ranked_doc_indices = self.rank_documents(query_vec)

        print("Pruning the ranked files...")

        # Only keep files that the words actually occur in
        pruned_ranked_indices = self.prune_ranking(ranked_doc_indices, allowed_filenames)

        print("Changing indicies to files including parsing...")

        # Turn the ranked document indicies into actual file information
        files = self.ranked_indices2files(pruned_ranked_indices)

        print("Adding URLs to parsed results...")

        # Add URLs to the search results
        files = self.add_urls(files)

        to_print = []
        for f in files:
            f_url = f["url"]
            if "mobile" not in f_url:
                to_print.append(f)
        to_print = to_print[:10]
        for item in to_print:
            print(item["filename"], " -> ", item["url"])
        input()
        prev_html = ""
        for f in files:
            print(f["title"].strip(), " --> ", f["filename"], " --> ", f["url"])
            print(f["file_text"] == prev_html)
            prev_html = f["file_text"]
            input()

        return files

    def prune_ranking(self, ranked_indices, allowed_filenames):
        """
        Prunes the ranking so that only files which actually contain the terms are kept.
        :param ranked_indices: the ranked indices of files as a list
        :param allowed_filenames: a set of the filenames which contain terms from the search query
        :return: the ranked indices excluding files that don't contain any of the search terms.
        """
        # Maximum number of returned files is 100
        max_files = 100

        final_ranking = []
        for ranked_i in ranked_indices:
            fname = self.index2filename[ranked_i]

            if fname in allowed_filenames and len(final_ranking) < max_files:
                final_ranking.append(ranked_i)

        return final_ranking

    def check_index(self, query):
        """
        Takes the query as a list of words and returns the list of filenames that contain the words from that query.
        :param query: the search query
        :return: list of files containing one or more query terms
        """
        allowed_filenames = set()

        for word in query:

            files_word_occurs_in = self.inverted_index[word]

            for fname, count in files_word_occurs_in:
                if count > 1:
                    allowed_filenames.add(fname)

        return allowed_filenames

    def add_urls(self, files):
        """
        Adds URLs to the search results.
        :param files: the files to add URLs to.
        :return: the file dicts with URLs.
        """

        new_files = []
        for f in files:
            fname = f["filename"]
            url = self.filenames2urls[fname]
            f["url"] = url
            new_files.append(f)

        return new_files

    def query2vec(self, query):
        """
        Takes a query from the user and transforms it into a one-hot vector.
        :param query: the query as a list of words.
        :return: the query as a one-hot vector and the query count (number of words from the query that were in the
                 vocab. If it is zero, return no results)
        """

        # Sanity check
        assert(type(query) is list)

        # Create a vector of zeros to store the query as a vector.
        query_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)

        # Counter of how many words from the query were in the vocab. If none, return no results.
        query_count = 0

        # Iterate over each word in the query
        for word in query:

            if word in self.vocab:
                index = self.term2rowindex[word]
                query_vec[0, index] = 1
                query_count += 1

        return query_vec, query_count

    def calculate_new_query_vec(self, query_vec):
        """
        Transforms the one-hot query vector into the k-dimensional space given by the decomposition of the term-document
        matrix.
        :param query_vec: the one hot vector version of the query.
        :return: the query as a dense vector of dimension k.
        """
        # The formula is q_k = q * Term_Matrix_k * inverse(Singular_Values_Matrix_k)

        # Invert the k-dimensional singular value matrix
        inverse_singular_values = np.linalg.inv(self.singular_values_k)

        # Dot the query vector with the term matrix
        q_k = np.dot(query_vec, self.term_matrix_k)

        # Dot the result of the previous operation with the inverted singular values
        q_k = np.dot(q_k, inverse_singular_values)

        return q_k

    def rank_documents(self, query_vec):
        """
        Ranks the documents using the decomposed term-document matrix and the query vector.
        :param query_vec: the query vector embedded into the k-dimensional space.
        :return: a list of the indices of the documents in order of their relevance to the query.
        """

        # Get the number of rows in the document matrix
        rows_dm, cols_dm = self.document_matrix_k.shape

        # List of the similarities of the query vector to each document and the index of the document that they corres-
        # pond to.
        similarities = []

        # Compare the query to each document. Each document is represented by a row in the document matrix of rank k.
        for i in range(rows_dm):

            # Get the document vector
            document_vector = self.document_matrix_k[i, :]

            # Get the query vector into a suitable form
            q_vec = query_vec[0, :]

            # Calculate the cosine distance. We do 1 minus the scipy implementation because the scipy implementation
            # computes distance rather than similarity
            similarity = 1 - spatial.distance.cosine(q_vec, document_vector)

            similarities.append((similarity, i))

        # Sort the documents by greatest similarity
        similarities = [x for x in reversed(sorted(similarities, key=itemgetter(0)))]

        # Return the document indices
        indices = [x for _, x in similarities]

        return indices

    def similarity(self, bow1, bow2):
        """
        Compares file similarity on two bag of words representations
        :param bow1: first bag of words
        :param bow2: second bag of words
        :return: similarity between 0 and 1
        """
        dot_prod, file1_norm, file2_norm = 0.0, 0.0, 0.0
        for word, count1 in bow1.items():
            count2 = bow2[word]
            dot_prod += count1 * count2
            file1_norm += count1 ** 2
            file2_norm += count2 ** 2
        return dot_prod / (sqrt(file1_norm) * sqrt(file2_norm))

    def ranked_indices2files(self, ranked_indices):
        """
        Turns ranked document indices into parsed HTML files.
        :param ranked_indices: the indices of files in the TD matrix ranked according to relevance to the query
        :return: parsed HTML files corresponding to the ranked indices
        """
        # List of the filenames
        filenames = []

        # Iterate over each of the indices
        for index in ranked_indices:
            filename = self.index2filename[index]
            filenames.append(filename)

        # List to hold parsed files
        parsed_files = []

        # Parse each file
        for filename in filenames:
            path = self.html_dir_path + filename
            parsed_html = self.parser.parse(path, bag_of_words=True)

            # Check if the HTML file is already present
            already_present = False
            for parsed_file in parsed_files:
                sim = self.similarity(parsed_html["bag_of_words"], parsed_file["bag_of_words"])
                if sim > 0.9:
                    already_present = True
                    break

            if not already_present:
                parsed_files.append(parsed_html)

        return parsed_files

if __name__ == '__main__':
    create_new = False

    if create_new:
        t = time.time()
        lsi = LatentSemanticIndexer("LSI/Term_Document_Matrix/", False)
        print("----> Decomposed the Term-Document matrix, time taken: ", time.time() - t, " seconds")
    else:

        # The base directory of the project on the system
        BASE_DIR = "/Users/edcollins/Documents/CS/4thYear/IRDM/CW/Code/"

        # Where to save the inverted index
        PATH_TO_SAVE_INVERTED_INDEX = BASE_DIR + "LSI/"

        # The path to the web pages stored in JSON format
        PATH_TO_STORED_WEBPAGES_AS_JSON = BASE_DIR + "LSI/pages/"

        # The full path to the inverted index
        FULL_PATH_TO_SAVED_INVERTED_INDEX = PATH_TO_SAVE_INVERTED_INDEX + "inverted_index.pkl"

        # The location in which to save the term-document matrix
        PATH_TO_SAVE_TERM_DOCUMENT_MATRIX = BASE_DIR + "LSI/Term_Document_Matrix/"

        # Path to the mapping from filenames to URLs
        PATH_TO_FILENAME2URL_MAPPING = BASE_DIR + "LSI/filename2url.pkl"

        # The path to save the ranking matricies
        PATH_TO_SAVE_RANKING_MATRICIES = BASE_DIR + "LSI/SVD/"

        lsi = LatentSemanticIndexer(
            path_to_td_matrix=PATH_TO_SAVE_TERM_DOCUMENT_MATRIX,
            path_to_inverted_index=FULL_PATH_TO_SAVED_INVERTED_INDEX,
            path_to_save_ranking_matricies=PATH_TO_SAVE_RANKING_MATRICIES,
            path_to_filename2url_mapping=PATH_TO_FILENAME2URL_MAPPING,
            path_to_crawled_files=PATH_TO_STORED_WEBPAGES_AS_JSON,
            searching=True
        )

        lsi.search("machine learning")
