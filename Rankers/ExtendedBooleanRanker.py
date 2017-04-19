from Rankers.Ranker import Ranker
from collections import defaultdict
import pickle


INVERTED_INDEX = "data/inverted_index_boolean.pkl"
FILENAMES2URL = "data/filename2url_boolean.pkl"


class ExtendedBooleanRanker(Ranker):

    def __init__(self, index, filenames, p):
        """
        Create an object for boolean retrieval with an inverted index and list of filenames.
        :param index: an inverted index created by IndexBoolean.
        :param filenames: a list of filenames created by IndexBoolean.
        :param p: parameter for the p-norm.
        """
        self.inverted_index = index
        self.filenames = filenames
        self.p = p

        self.term_to_files = defaultdict()

        for term, entries in self.inverted_index.items():
            files_containing_term = []
            for entry in entries:
                files_containing_term.append(entry[0])
            self.term_to_files[term] = files_containing_term

    def t2d(self, term):
        """
        Check if a term exists in the vocabulary.
            Yes -> then return the list of files it can be found in.
            No -> return an empty list.
        :param term: the term to be checked.
        :return: list of files containing the term.
        """
        if term in self.term_to_files:
            return self.term_to_files[term]
        else:
            return []

    def basic_retrieval(self, query, qtype):
        """
        Performs the most basic boolean retrieval
        :param query: the query is a list of terms.
        :param qtype: the query type must be 'OR' or 'AND'.
        :return: a set of files that satisfy the query.
        """
        return_set = None
        if qtype == 'OR':
            return_set = set()
        elif qtype == 'AND':
            return_set = set(self.filenames)
        for term in query:
            if qtype == 'OR':
                return_set |= set(self.t2d(term))
            elif qtype == 'AND':
                return_set &= set(self.t2d(term))
        return return_set

    def similarity(self, query, file, weights, qtype):
        """
        Compute the similarity score of the p-norm model.
        :param query: the query is a list of terms.
        :param file: the filename of a document.
        :param weights: assigned weight for each query term.
        :param qtype: the query type must be 'OR' or 'AND'.
        :return: similarity score between query and file.
        """
        nom = 0
        denom = sum([w ** p for w in weights])
        if qtype == 'OR':
            for i in range(len(query)):
                if file in self.t2d(query[i]):
                    nom += weights[i] ** self.p
            return (nom / denom) ** (1/self.p)
        elif qtype == 'AND':
            for i in range(len(query)):
                if file not in self.t2d(query[i]):
                    nom += weights[i] ** self.p
            return 1 - ((nom / denom) ** (1/self.p))

    def search(self, query, num_results):

        query = query.lower().split()
        num_results = int(num_results)
        weights = [1]*len(query)

        and_scores = sorted(
            [(self.similarity(query, file, weights, 'AND'), file) for file in self.filenames],
            reverse=True)

        ranked_list = []
        for (score, file) in and_scores[:num_results]:
            ranked_list.append(file)

        return ranked_list



# ================ MAIN ================

if __name__ == '__main__':

    index = None
    filenames = None

    print("Loading index and filenames from file.")
    with open(INVERTED_INDEX, "rb") as f:
        filename2url = pickle.load(f)
        filenames = filename2url.keys()
    with open(FILENAMES2URL, "rb") as f:
        load = pickle.load(f)
        index = load[0]

    p = 2
    print("Creating model with p-norm =", p)
    model = ExtendedBooleanRanker(index, filenames, p)
    print("Model created.")