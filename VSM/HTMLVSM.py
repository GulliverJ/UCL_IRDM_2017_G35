from HTML2Index import HTML2Index
from math import log, sqrt
from collections import defaultdict
from sys import exit

class HTMLVSM:

    def __init__(self, index_creator):
        """
        Create a vector space model for use with a HTML2Index object.
        :param index_creator: HTML2Index object.
        """

        # References to variables of the HTML2Index object
        self.inverted_index = index_creator.get_index()
        self.word_counts = index_creator.get_word_counts()
        self.file_count = index_creator.get_file_count()
        self.filenames = index_creator.get_filenames()

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
                self.term_freq[file][term] = 1.0 + log(count, 2)

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
        :return: decimal value between 0 and 1.
        """
        dot_prod, q_norm, d_norm, cos = 0.0, 0.0, 0.0, 0.0
        q_tf_norm = sqrt(sum([query.count(term) ** 2 for term in query]))
        for term in query:
            q_tf_idf = query.count(term) / q_tf_norm * self.idf(term)
            d_tf_idf = self.tf_idf(term, file)
            dot_prod += q_tf_idf * d_tf_idf
            q_norm += q_tf_idf ** 2
            d_norm += d_tf_idf ** 2
        q_norm = sqrt(q_norm)
        d_norm = sqrt(d_norm)
        if q_norm != 0 and d_norm != 0:
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

    index_creator = HTML2Index("html/")

    model = HTMLVSM(index_creator)

    while True:
        model.search()
