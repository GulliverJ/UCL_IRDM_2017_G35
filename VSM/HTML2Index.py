# ======== IMPORTS ========

# For parsing the HTML
from bs4 import BeautifulSoup

# To iterate over files
import os

# To construct the bag of words representation of each paper
from collections import defaultdict

# To save the index
import pickle

# =========================


class HTML2Index:

    def __init__(self, html_dir):
        """
        An object which will turn a directory of HTML files into an inverted index with counts. It will create a
        dictionary with the keys being the vocabulary of all of the HTML files, and the values being a list of the form
        [(document_word_occurs_in, count_occurences_in_document)] for each document that the word occurs in.

        Usage example:

        >>> index_creator = HTML2Index(directory_to_parse)
        >>> index = index_creator.get_index()
        >>> index["CS"]
        [("research.html", 3), ("index.html", 1)]

        :param html_dir: the directory to parse html files from. This should just be the directory name of the HTML
                         files, assuming that this file is in a directory with the HTML files as a subdirectory.
                         As type string.
        """
        # Sanity check
        assert(type(html_dir) is str)

        if html_dir.endswith("/"):
            html_dir = html_dir.strip("/")

        self.word_counts = defaultdict()
        self.file_count = 0
        self.filenames = []
        self.html_directory = html_dir
        self.index = self.build_index()

    def get_filenames(self):
        """
        Returns a list of filenames
        :return: the value of filenames
        """
        return self.filenames

    def get_file_count(self):
        """
        Returns the the number of files processed.
        :return: the value of file_count
        """
        return self.file_count

    def get_word_counts(self):
        """
        Returns the word counts for each file in the HTML directory file.
        :return: the word_counts constructed in build_index()
        """
        return self.word_counts

    def get_index(self):
        """
        Returns the inverted index for the HTML directory file.
        :return: the inverted index constructed by build_index()
        """
        return self.index

    def save_index(self):
        """
        Saves the index using Python's pickle.
        """
        with open("inverted_index.pkl", "wb") as f:
            pickle.dump(self.index, f)

    def build_index(self):
        """
        Builds the inverted index of HTML files.
        :return: the inverted index.
        """

        # The inverted index will be a defaultdict
        inverted_index = defaultdict(list)

        # Iterate over each HTML file
        for filename in os.listdir(self.html_directory):

            if filename.endswith(".html"):

                # Read the HTML file
                with open(self.html_directory + "/" + filename, "r") as f:
                    html = f.read().lower()

                # Turn the raw HTML string into a dictionary which is a bag of words representation of that file.
                html_bag_of_words = self.process_html_file(html)

                # Merge the existing inverted index with the information just gathered from this HTML file
                inverted_index = self.merge_index(inverted_index, html_bag_of_words, filename)

                # Update file counter, filename list and word counts
                self.file_count += 1
                self.filenames.append(filename)
                self.word_counts[filename] = html_bag_of_words

        return inverted_index

    def process_html_file(self, html):
        """
        Processes a string of HTML to return a bag of words representation of that HTML.
        :param html: the HTML page to parse, as a string
        :return: a bag of words representation of the file, which is a dictionary of the form
                {word: count_of_word_in_file}, for every word in the file.
        """

        # Sanity check
        assert(type(html) is str)

        # Parse the HTML string with Beautiful Soup
        soup = BeautifulSoup(html, "lxml")

        # Get the page text
        page_text = soup.get_text()

        # Split the page text into individual tokens
        page_words = page_text.split()

        # Bag of words representation will be a defaultdict
        bag_of_words = defaultdict(int)

        # Construct the bag of words representation
        for word in page_words:
            bag_of_words[word] += 1

        return bag_of_words

    def merge_index(self, existing_index, bag_of_words, filename):
        """
        Merges the existing index that we have constructed so far with the bag of words representation of the latest
        HTML page.
        :param existing_index: the existing inverted index, as a dictionary of form {word: documents_and_counts}
        :param bag_of_words: the bag of words representation of the HTML file given by filename.
        :param filename: the name of the HTML file which is being added to the index.
        :return: the new and extended inverted index.
        """

        # Iterate over each words in the bag of words representation and add it to the existing index with the document
        # that it appears in, plus the count of the word in that document.
        for word, count in bag_of_words.items():
            existing_index[word].append((filename, count))

        return existing_index


# ================ MAIN ================

if __name__ == '__main__':

    index_creator = HTML2Index("HTML/")

    inverted_index = index_creator.get_index()
    index_creator.save_index()

    for key, val in inverted_index.items():
        print(key, " ", val)
