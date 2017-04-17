# ======== IMPORTS ========

# To construct the bag of words representation of each paper
from collections import defaultdict

# For reading the inverted index
import pickle

# To create the term-document matrix
import numpy as np

# For timing
import time

# =========================


class TDMatCreator:

    def __init__(self, inverted_index_path="", create=True):
        """
        Creates a term-document matrix which has a row for every term in the vocabulary, and a column for every
        document. There is a 1 in each entry if that term occurs in that document.
        :param inverted_index_path: the path to the inverted index of the documents - mapping words to the documents
                                    that they occur in.
        :param create: if True, will create the term-document matrix. If false, will not.
        """
        if create and inverted_index_path == "":
            raise ValueError("If you want to create a term-document matrix, you must provide a path to the inverted" +
                             "index")

        if create:
            with open(inverted_index_path, "rb") as f:
                self.inverted_index, self.file_count = pickle.load(f)
                print("----> No. Files to Process: ", self.file_count)

            # A vocabulary as a set
            self.vocab = set()

            # Filename to URL mapping
            self.filename2url = self.read_filename2url()

            # List of the filenames already seen
            self.filenames = set()

            # Dictionary which will hold a mapping from words to row indicies in the term-document matrix
            self.term2rowindex = defaultdict(int)
            self.create_term2rowindex_mapping()

            # Dictionary which will hold a mapping from filenames to column indicies in the term-document matrix
            self.filename2colindex = defaultdict(int)
            self.create_filename2colindex_mapping()

            # Number of files
            self.num_documents = len(self.filename2url)

            # The term-document matrix
            self.term_document_matrix = self.create_TD_matrix()

            print(np.shape(self.term_document_matrix))

    def read_filename2url(self):
        """
        Reads filename2url file.
        :return: filename2url file.
        """
        with open("LSI/filename2url.pkl", "rb") as f:
            filename2url = pickle.load(f)

        return filename2url

    def create_filename2colindex_mapping(self):
        """
        Creates filename2colindex mapping.
        :return: mapping
        """
        file_count = 0
        for key, _ in self.filename2url.items():
            self.filename2colindex[key] = file_count
            self.filenames.add(key)
            file_count += 1

    def create_term2rowindex_mapping(self):
        """
        Term2rowindex mapping creator
        :return:
        """
        term_count = 0
        for key, _ in self.inverted_index.items():
            self.term2rowindex[key] = term_count
            self.vocab.add(key)
            term_count += 1

    def create_TD_matrix(self):
        """
        Creates the term-document matrix by first mapping each word and document name to a unique integer ID.
        :return: the term document matrix.
        """

        # Calculate the size of the vocab (which will be the number of rows in the TD matrix. +1 for OOV token.
        vocab_size = len(self.inverted_index.keys())

        # Create the initial term-document matrix
        td_matrix = np.zeros((vocab_size, self.num_documents), dtype=np.float32)

        # For logging
        count = 0

        # Iterate over all of the terms in the inverted index
        for term, doc_list in self.inverted_index.items():

            print("Processing Term: ", count, end="\r")
            count += 1

            # A list of document IDs for all documents that this term maps to
            document_IDs = [(self.filename2colindex[fname], tc) for fname, tc in doc_list]

            # Get the index of the current term
            term_index = self.term2rowindex[term]

            # Get the number of files which this word occurs in (for calculating TF-IDF)
            num_file_occurences = len(doc_list)

            # Add to the term-document matrix
            td_matrix = self.populate_td_matrix(td_matrix, term_index, document_IDs, num_file_occurences)

        return td_matrix

    def add_to_term_index(self, term, term_count):
        """
        Adds a term to the term index and increments term_count
        :param term: the term to add
        :param term_count: the ID for the term to add
        :return: the incremented term_count
        """

        # Add to the vocab
        self.vocab.add(term)

        # Add to the term-to-id mapping
        self.term2rowindex[term] = term_count

        # Increment the term counter
        term_count += 1

        return term_count

    def add_to_file_index(self, filename, file_count):
        """
        Adds the filename to the mapping from filenames to columns, and increments the file_count.
        :param filename: the name of the file to add.
        :param file_count: the ID to assign to filename
        :return: file_count incremented
        """

        # Add to the set of filenames already seen
        self.filenames.add(filename)

        # Add to the filename-to-id mapping
        self.filename2colindex[filename] = file_count

        # Increment the file counter
        file_count += 1

        return file_count

    def populate_td_matrix(self, td_matrix, term_index, document_IDs, num_file_occurences):
        """
        Adds entries for the current term to the TD matrix.
        :param td_matrix: the current TD matrix.
        :param term_index: index of the current term.
        :param document_IDs: list of tuples of document IDs that the term occurs in and the count of the term in that
                             document.
        :param num_file_occurences: the number of different documents which the term occurs in.
        :return: the updated td_matrix
        """

        # Measure the size of the TD matrix
        rows, cols = np.shape(td_matrix)

        # Get the column indexes to assign
        col_indexes = [x for x, _ in document_IDs]

        # Calculate the TF-IDFs for each entry
        tf_idfs = [self.calculate_tf_idf(num_file_occurences, y) for _, y in document_IDs]

        # Update the TD matrix with new values
        td_matrix[term_index, col_indexes] = tf_idfs

        return td_matrix

    def calculate_tf_idf(self, num_file_occurences, term_frequency):
        """
        Calculates the TF-IDF score of the term in the current file.
        :param num_file_occurences: the number of different files that the term occurs in.
        :param term_frequency: the number of times the term occurs in each document.
        :return: the TF-IDF score of the term in this document.
        """
        idf = np.log((self.file_count / num_file_occurences))
        tf = term_frequency
        return tf * idf

    def expand_td_matrix(self, td_matrix, cols):
        """
        Expands the TD matrix to have "cols" columns.
        :param td_matrix: the current td matrix
        :param cols: the number of columns the new matrix should have
        :return: the TD matrix with extra columns full of zeros added.
        """

        # Get the number of rows in the existing matrix
        rows, old_cols = np.shape(td_matrix)

        # Create the new TD matrix
        new_td_matrix = np.zeros((rows, cols), dtype=np.float32)

        # Make sure the new matrix has all the elements of the old one
        new_td_matrix[:, :old_cols] = td_matrix

        return new_td_matrix

    def save_objects(self, path=""):
        """
        Saves the TD matrix, mappings, vocabular and set of filenames.
        :param path: directory in which to save the information.
        """
        # Save the term2rowindex mapping first with Pickle
        with open(path + "term2rowindex.pkl", "wb") as f:
            pickle.dump(self.term2rowindex, f)

        # Save the filename2colindex next with Pickle
        with open(path + "filename2colindex.pkl", "wb") as f:
            pickle.dump(self.filename2colindex, f)

        # Save the vocab with Pickle
        with open(path + "vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

        # Save the filename set with Pickle
        with open(path + "filenames.pkl", "wb") as f:
            pickle.dump(self.filenames, f)

        # Finally save the TF matrix with numpy.save()
        with open(path + "td_matrix.npy", "wb") as f:
            np.save(f, self.term_document_matrix)

    def load_objects(self, path=""):
        """
        Loads the TD matrix, mappings, vocabular and set of filenames.
        :param path: directory in which to save the information.
        :return: TD matrix, mappings, vocabulary and set of filenames
        """
        # Save the term2rowindex mapping first with Pickle
        with open(path + "term2rowindex.pkl", "rb") as f:
            term2rowindex = pickle.load(f)

        # Save the filename2colindex next with Pickle
        with open(path + "filename2colindex.pkl", "rb") as f:
            filename2colindex = pickle.load(f)

        # Save the vocab with Pickle
        with open(path + "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        # Save the filename set with Pickle
        with open(path + "filenames.pkl", "rb") as f:
            filenames = pickle.load(f)

        # Finally save the TF matrix with numpy.save()
        with open(path + "td_matrix.npy", "rb") as f:
            term_document_matrix = np.load(f)

        return term2rowindex, filename2colindex, vocab, filenames, term_document_matrix


if __name__ == '__main__':
    t = time.time()
    tdmat_creator = TDMatCreator("LSI/inverted_index.pkl")
    print("\nTook ", time.time() - t, " seconds")
    input("Press enter to save...")
    tdmat_creator.save_objects("LSI/Term_Document_Matrix/")

    t2 = TDMatCreator(create=False)
    print(type(t2.load_objects("LSI/Term_Document_Matrix/")))
