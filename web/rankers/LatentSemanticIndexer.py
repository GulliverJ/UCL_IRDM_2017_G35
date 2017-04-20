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
from rankers.TDMatCreator import TDMatCreator

# Import the HTML parser
from rankers.LSIParser import LSIParser

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

    def __init__(self, path_to_td_matrix, path_to_inverted_index, path_to_save_ranking_matricies, searching):
        """
        A ranker that ranks documents by their cosine similarity after transforming the document into a representation
        using Singular Value Decomposition. Queries are likewise transformed into this semantic space and their vectors
        compared to that of each document.
        :param path_to_td_matrix: the path to the Term-Document matrix needed for Latent Semantic Indexing.
        :param path_to_inverted_index: the path to the inverted index mapping terms to documents they occur in.
        :param path_to_save_ranking_matricies: the path to save / load the ranking matrices
        :param searching: True if the directory of documents is being queried for a document. If this is true,
                          singular value decomposition will not be performed and the matrix will be loaded from disk.
                          The loaded matrices will then be used to rank and retreive documents. If it is false, then
                          the term document matrix will be decomposed with singular value decomposition to create the
                          ranking matrices.
        """

        # Load the Term-to-Row Mapping, Filename-to-Column Mapping, Vocab, Filenames and Term-Document Matrix
        self.term2rowindex, self.filename2colindex, self.vocab, self.filenames = \
            TDMatCreator(create=False).load_objects(path_to_td_matrix)

        with open(path_to_inverted_index, "rb") as f:
            self.inverted_index, self.file_count = pickle.load(f)
            print("----> No. Files to Process: ", self.file_count)

        # The rank to use to approximate the Term-Document matrix
        self.rank = 100

        if not searching:

            # Load the TD matrix seperately because it is so large
            self.term_document_matrix = TDMatCreator(create=False).load_td_matrix(path_to_td_matrix)

            # Create the rank-k approximation matrices
            self.term_matrix_k, self.singular_values_k, self.document_matrix_k = self.create_ranker()

            # Save the matrices
            self.save_ranker_matrices(path=path_to_save_ranking_matricies)

        if searching:

            # A HTML parser to parse documents
            self.parser = LSIParser()

            # Turn the filename2index mapping into an index2filename mapping
            self.index2filename = {val: key for key, val in self.filename2colindex.items()}

            # Load the rank-k versions of: term matrix, singular values and document matrix
            self.term_matrix_k, self.singular_values_k, self.document_matrix_k = \
                self.load_ranker_matrices(path=path_to_save_ranking_matricies)

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
        allowed_filenames, best_filenames, amazing_filenames = self.check_index(query)

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
        pruned_ranked_indices = self.prune_ranking(ranked_doc_indices, allowed_filenames, best_filenames, amazing_filenames)

        # Turn the indicies into the correct file PIDs
        final_indicies = self.rankings_to_pids(pruned_ranked_indices)

        return final_indicies

    def rankings_to_pids(self, pruned_ranked_indicies):
        """
        Turns the indices used by this ranker into the correct file PIDs
        :param pruned_ranked_indicies: ranked term document matrix indices
        :return: file PIDs
        """
        # List of the filenames
        final_indices = []

        # Iterate over each of the indices
        for index in pruned_ranked_indicies:
            filename = self.index2filename[index]
            final_indices.append(int(filename.strip(".json")))

        return final_indices

    def prune_ranking(self, ranked_indices, allowed_filenames, best_filenames, amazing_filenames):
        """
        Prunes the ranking so that only files which actually contain the terms are kept.
        :param ranked_indices: the ranked indices of files as a list
        :param allowed_filenames: a set of the filenames which contain terms from the search query
        :return: the ranked indices excluding files that don't contain any of the search terms.
        """
        # Maximum number of returned files is 100
        max_files = 100

        final_ranking = []
        okay_files = []
        for ranked_i in ranked_indices:
            fname = self.index2filename[ranked_i]

            if len(final_ranking) < max_files:

                if fname in amazing_filenames:
                    final_ranking.insert(0, ranked_i)
                elif fname in best_filenames:
                    final_ranking.append(ranked_i)
                elif fname in allowed_filenames:
                    okay_files.append(ranked_i)

        final_ranking = final_ranking + okay_files

        return final_ranking

    def check_index(self, query):
        """
        Takes the query as a list of words and returns the list of filenames that contain the words from that query.
        :param query: the search query
        :return: list of files containing one or more query terms
        """
        allowed_filenames = set()
        best_filenames = set()
        amazing_files = set()

        for word in query:

            files_word_occurs_in = self.inverted_index[word]

            for fname, count in files_word_occurs_in:
                if count > 1:
                    if fname in best_filenames:
                        amazing_files.add(fname)
                    if fname in allowed_filenames:
                        best_filenames.add(fname)
                    else:
                        allowed_filenames.add(fname)

        return allowed_filenames, best_filenames, amazing_files

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
    listing_conversion = pickle.load(open("./Parser/filenames2results.pkl", "rb"))

    def similarity(bow1, bow2):
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

    def translate_ranking(ranking):
        """
        Translate numerical page IDs to readable info for display - only keeping pages which are not already present.
        :param ranking: a list of document IDs, with the ID at position 0 corresponding to the most highly ranked file.
        :return: dictionaries of information corresponding to the IDs, used for display
        """
        # The final results to display
        listings = []

        for item in ranking:

            # Get the result for the first PID
            file_result = listing_conversion["%d.json" % item]

            # This will be true if the current item is already in the ranking
            already_present = False

            # Check the existing list of files
            for existing_listing in listings:

                # Calculate similarity using vector methods
                sim = similarity(file_result["bag_of_words"], existing_listing["bag_of_words"])

                # If similarity is more than 90%, it is likely the file is already present
                if sim > 0.9:
                    already_present = True
                    break

            # Only append a file if it is not already there
            if not already_present:
                listings.append(file_result)

        # NOTE: Enforcing only top 20 listings for now?
        num_results = 20

        if len(listings) > num_results:
            return listings[0:num_results]
        else:
            return listings

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
        PATH_TO_SAVE_TERM_DOCUMENT_MATRIX = BASE_DIR + "LSI/LSIData/"

        # The path to save the ranking matricies
        PATH_TO_SAVE_RANKING_MATRICIES = BASE_DIR + "LSI/LSIData/"

        lsi = LatentSemanticIndexer(
            path_to_td_matrix=PATH_TO_SAVE_TERM_DOCUMENT_MATRIX,
            path_to_inverted_index=FULL_PATH_TO_SAVED_INVERTED_INDEX,
            path_to_save_ranking_matricies=PATH_TO_SAVE_RANKING_MATRICIES,
            searching=True
        )

        ranked_indicies = lsi.search("evil cate nation")

        files = translate_ranking(ranked_indicies)

        for f in files:
            print(f["url"])
        input()
