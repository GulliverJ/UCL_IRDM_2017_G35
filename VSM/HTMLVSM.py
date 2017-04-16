from HTML2Index import HTML2Index
from math import log, sqrt
from collections import defaultdict
from sys import exit


class HTMLVSM:

    def __init__(self, index_creator, variant, pivot=0.0, slope=1.0):
        """
        Create a vector space model for use with a HTML2Index object.
        :param index_creator: HTML2Index object.
        :param variant: String using the qqq.ddd notation.
        """

        # References to variables of the HTML2Index object
        self.inverted_index = index_creator.get_index()
        self.word_counts = index_creator.get_word_counts()
        self.file_count = index_creator.get_file_count()
        self.filenames = index_creator.get_filenames()

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

    def search(self):
        """
        Prompt the user for input query.
        Calculate the cosine similarity between query and all documents.
        Sort and print the scores for all files.
        """
        query = input("Search for: ").lower().split()
        if not query:
            exit()
        scores = sorted(
            [(self.cos_sim(query, file), file) for file in self.filenames],
            reverse=True)
        print("Score: filename")
        for (score, file) in scores:
            print(str(score) + ": " + file)

# ================ MAIN ================

if __name__ == '__main__':

    index_creator = HTML2Index("htmldocs/")

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
    print("Creating model.")
    model = HTMLVSM(index_creator, variant)
    print(variant, "model created.")

    while True:
        model.search()
