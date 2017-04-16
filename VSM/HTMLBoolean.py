from HTML2Index import HTML2Index
from sys import exit
from collections import defaultdict


class HTMLBoolean:

    def __init__(self, index_creator, p):
        """
        Create an object for boolean retrieval with a HTML2Index object.
        :param index_creator: HTML2Index object.
        :param p: parameter for the p-norm.
        """
        self.inverted_index = index_creator.get_index()
        self.filenames = index_creator.get_filenames()
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

    def search(self):
        """
        Prompt the user for input query.
        First does basic boolean retrieval for both AND and OR queries.
        Then uses the p-norm model to produce a ranking.
        User can manually specify the weights of each term.
        Example:
            query: unexamined game of life
            weights: 5 1 1 1
            result: quote2.html ranks highest
        """
        query = input("Search for: ").lower().split()
        weights = [float(x) for x in input("Enter term weights as space separated numbers: ").split()]
        if not query:
            exit()
        if len(weights) != len(query):
            print("Number of terms weights different to number of query terms -> Weighting all terms equally")
            weights = [1]*len(query)
        print('at least one term found in:', self.basic_retrieval(query, 'OR'))
        print('all terms found in:', self.basic_retrieval(query, 'AND'))
        print("-----------------------------------")

        or_scores = sorted(
            [(self.similarity(query, file, weights, 'OR'), file) for file in self.filenames],
            reverse=True)
        print("Score: filename [OR query]")
        for (score, file) in or_scores:
            print(str(score) + ": " + file)
        print("-----------------------------------")

        and_scores = sorted(
            [(self.similarity(query, file, weights, 'AND'), file) for file in self.filenames],
            reverse=True)
        print("Score: filename [AND query]")
        for (score, file) in and_scores:
            print(str(score) + ": " + file)


# ================ MAIN ================

if __name__ == '__main__':

    index_creator = HTML2Index("htmldocs/")

    p = 2
    print("Creating model with p-norm =", p)
    model = HTMLBoolean(index_creator, p)
    print("Model created.")
    print("-----------------------------------")

    while True:
        model.search()
