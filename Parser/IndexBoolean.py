# ======== IMPORTS ========

# For parsing the HTML
from bs4 import BeautifulSoup

# To iterate over files
import os

# To construct the bag of words representation of each paper
from collections import defaultdict

# To save the index
import pickle

# Import the HTML parser
from Parser.ParserBoolean import ParserBoolean

# To time how fast this executes
import time

# For reading pages
import json

# Measuring length and standard deviation
import numpy as np

# =========================

PAGES = "data/pages_good"
INVERTED_INDEX = "data/inverted_index_boolean.pkl"
FILENAMES2URL = "data/filename2url_boolean.pkl"


class IndexBoolean:

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

        self.html_directory = html_dir

        # A parser object to process words
        self.parser = HTMLParser()

        # Counter for the number of files
        self.file_count = 0

        # A mapping from filenames to URLs
        self.filenames2urls = {}

        self.index = self.build_index()

    def get_index(self):
        """
        Returns the inverted index for the HTML directory file.
        :return: the inverted index constructed by build_index()
        """
        return self.index

    def save_index(self, path=""):
        """
        Saves the index using Python's pickle. Saves the index and the number of files as a tuple.
        """
        # Save the inverted index
        to_save = (self.index, self.file_count)
        with open(path + INVERTED_INDEX, "wb") as f:
            pickle.dump(to_save, f)

        # Save the filename to url mapping
        with open(path + FILENAMES2URL, "wb") as f:
            pickle.dump(self.filenames2urls, f)

    def build_index(self):
        """
        Builds the inverted index of HTML files.
        :return: the inverted index.
        """

        # The inverted index will be a defaultdict
        inverted_index = defaultdict(list)

        # For logging
        count = 0

        # Iterate over each HTML file
        for filename in os.listdir(self.html_directory):

            if filename.endswith(".json"):

                print("Processing File: ", count, end="\r")

                # Read the JSON file that contains the HTML
                with open(self.html_directory + "/" + filename, "r") as f:
                    html_dict = json.load(f)

                # Extract the HTML
                html = html_dict["html"]

                # Extract the URL
                url = html_dict["url"]

                # Add to the filename-to-url mapping
                self.filenames2urls[filename] = url

                # Turn the raw HTML string into a dictionary which is a bag of words representation of that file.
                html_bag_of_words = self.process_html_file(html)

                # Merge the existing inverted index with the information just gathered from this HTML file
                inverted_index = self.merge_index(inverted_index, html_bag_of_words, filename)

                # Increment the file counter
                self.file_count += 1

                # Increment the logging counter
                count += 1

                if count % 1000 == 0:
                    print(count, "files processed.")

        inverted_index = self.prune_index(inverted_index)

        return inverted_index

    def prune_index(self, index):
        """
        Removes keys and values that are irrelevant by being more than 3 standard deviations from the average length.
        :param index: the index to prune
        :return: the pruned index
        """
        # Collect the lengths
        lens = []
        for key, _ in index.items():
            lens.append(len(key))

        # Measure the mean and standard deviation
        mean_len = np.mean(lens)
        std_dev = np.std(lens)

        # Calculate the max length
        max_len = mean_len + 3 * std_dev

        # New index
        new_index = defaultdict(list)

        for key, val in index.items():
            if len(key) < max_len and len(val) > 1:
                new_index[key] = val

        return new_index

    def process_html_file(self, html, ignore_stopwords=True):
        """
        Processes a string of HTML to return a bag of words representation of that HTML.
        :param html: the HTML page to parse, as a string.
        :param ignore_stopwords: if true will not add stopwords to the bag of words representation.
        :return: a bag of words representation of the file, which is a dictionary of the form
                {word: count_of_word_in_file}, for every word in the file.
        """

        # Sanity check
        assert(type(html) is str)

        # Parse the HTML string with Beautiful Soup
        soup = BeautifulSoup(html, "lxml")

        # Remove all script tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the page text
        page_text = soup.get_text()

        # Split the page text into individual tokens
        page_words = page_text.split()

        # Bag of words representation will be a defaultdict
        bag_of_words = defaultdict(int)

        # Construct the bag of words representation
        for word in page_words:

            # Remove punctuation from the word and translate it to lowercase
            new_word = self.parser.process_word(word)

            # Check if it is a stopword, if so don't add it to the bag of words
            if ignore_stopwords and new_word in self.parser.stopwords:
                continue

            bag_of_words[new_word] += 1

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

    index_creator = IndexBoolean(PAGES)
    index_creator.save_index()
