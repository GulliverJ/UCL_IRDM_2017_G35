from math import log, sqrt
from collections import defaultdict
import pickle
#from Rankers.Ranker import Ranker


INVERTED_INDEX = "data/inverted_index_vsm.pkl"
FILENAMES2URL = "data/filename2url_vsm.pkl"
WORD_COUNTS = "data/word_counts_vsm.pkl"


class VSMRanker:

    def __init__(self, index, word_counts, filenames, variant, pivot=0.0, slope=1.0):
        """
        Create a vector space model.
        :param index: an inverted index created by IndexVSM.
        :param word_counts: a dict of word_counts for each file created by IndexVSM.
        :param filenames: a list of filenames created by IndexVSM..
        :param variant: String using the qqq.ddd notation.
        :param pivot: pivot parameter, unused unless specified.
        :param slope: slope parameter, unused unless specified.
        """

        # References to variables of the HTML2Index object
        self.inverted_index = index
        self.word_counts = word_counts
        self.filenames = filenames
        self.file_count = len(filenames)


        # Parse the variant string
        self.variant = list(variant)

        # Variables for pivoted normalization
        self.pivoted_norm = False
        if pivot > 0:
            self.pivoted_norm = True
            self.pivot = pivot
            self.slope = slope

        # Initialize and normalize the term frequencies
        self.term_freq = defaultdict(dict)
        self.init_and_norm_tf()

    def init_and_norm_tf(self):
        """
        Calculate the log frequencies of terms from the word_counts.
        Then length normalize the term frequency vectors for each file.
        """

        for file, term_list in self.word_counts.items():
            for term, count in term_list.items():
                if self.variant[4] == 'b':
                    self.term_freq[file][term] = 1
                elif self.variant[4] == 'n':
                    self.term_freq[file][term] = count
                elif self.variant[4] == 'l':
                    self.term_freq[file][term] = 1.0 + log(count, 2)

        if self.variant[6] == 'c':
            for file in self.term_freq:
                norm = sqrt(sum([_ ** 2 for _ in self.term_freq[file].values()]))
                for term in self.term_freq[file]:
                    self.term_freq[file][term] /= norm

    def idf(self, term):
        """
        Calculate the inverse document frequency for a term.
        :param term: a word in a query or document.
        :return: log ( count of all documents / number of documents with this term )
                 returns zero if the term is not in the vocabulary.
        """
        if term in self.inverted_index:
            return log(self.file_count / len(self.inverted_index[term]), 2)
        return 0.0

    def tf_idf(self, term, file):
        """
        Calculate the tf-idf "importance" of a term.
        :param term: a word in a query or document.
        :param file: the filename of the document.
        :return: tf * idf
                 returns zero if the term is not in the vocabulary.
        """
        if term in self.inverted_index:
            if term in self.term_freq[file]:
                return self.term_freq[file][term] * self.idf(term)
        return 0.0

    def cos_sim(self, query, file):
        """
        Calculate the cosine similarity between query and document vectors.
        :param query: a list of words.
        :param file: the filename of the document.
        :return: similarity score between query and file.
        """
        dot_prod, q_norm, d_norm, cos = 0.0, 0.0, 0.0, 0.0

        q_tf_norm = 1
        if self.variant[2] == 'c':
            q_tf_norm = sqrt(sum([query.count(term) ** 2 for term in query]))

        for term in query:
            q_tf = 0
            if self.variant[0] == 'b':
                q_tf = 1
            if self.variant[0] == 'n':
                q_tf = query.count(term)
            elif self.variant[0] == 'l':
                q_tf = 1.0 + log(q_tf, 2)
            q_idf = 1
            if self.variant[1] == 't':
                q_idf = self.idf(term)
            q_tf_idf = q_tf / q_tf_norm * q_idf

            d_tf_idf = self.tf_idf(term, file)

            dot_prod += q_tf_idf * d_tf_idf
            q_norm += q_tf_idf ** 2
            d_norm += d_tf_idf ** 2

        q_norm = sqrt(q_norm)
        d_norm = sqrt(d_norm)
        if self.pivoted_norm:
            s = self.slope
            p = self.pivot
            q_norm = (1.0 - s) * p + s * q_norm
            d_norm = (1.0 - s) * p + s * d_norm
        if q_norm > 0 and d_norm > 0:
            cos = dot_prod / (q_norm * d_norm)

        return cos

    def search(self, query):

        query = query.lower().split()

        scores = sorted(
            [(self.cos_sim(query, file), file) for file in self.filenames],
            reverse=True)

        ranked_list = []
        for (score, file) in scores:
            ranked_list.append(int(file.strip(".json")))

        return ranked_list

# ================ MAIN ================

if __name__ == '__main__':

    index = None
    word_counts = None
    filenames = None

    print("Loading index and filenames from file.")
    with open(FILENAMES2URL, "rb") as f:
        filename2url = pickle.load(f)
        filenames = filename2url.keys()
    with open(INVERTED_INDEX, "rb") as f:
        load = pickle.load(f)
        index = load[0]
    with open(WORD_COUNTS, "rb") as f:
        word_counts = pickle.load(f)

    # Each variant uses the notation qqq.ddd
    # Term weighting:
    #   n: natural
    #   l: logarithmic
    #   b: boolean
    # Document weighting:
    #   n: none
    #   t: inverse document frequency
    # Normalization:
    #   n: none
    #   c: cosine
    # Note: not all combinations are valid.
    # Example: ntc.lnc
    #   query: natural term frequency, idf weighting, cosine normalization
    #   document: logarithmic term frequency, no df weighting, cosine normalization
    variant = "ntc.ltc"

    # Create model
    print("Creating model with variant =", variant)
    model = VSMRanker(index, word_counts, filenames, variant)
    print("Model created.")
