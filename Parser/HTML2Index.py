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
from HTMLParser import HTMLParser

# To time how fast this executes
import time

# For reading pages
import json

# Measuring length and standard deviation
import numpy as np

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

        self.html_directory = html_dir

        self.num_files = len([x for x in os.listdir(html_dir) if x.endswith(".json")])

        # A parser object to process words
        self.parser = HTMLParser()

        # Counter for the number of files
        self.file_count = 0

        # A mapping from filenames to URLs
        self.filenames2urls = {}

        # The number of files a term must occur in to be included in the inverted index
        self.min_occurences = 5

        self.index = self.build_index()

    def get_index(self):
        """
        Returns the inverted index for the HTML directory file.
        :return: the inverted index constructed by build_index()
        """
        return self.index

    def has_numbers(self, string):
        """
        Returns true if "string" has any numbers.
        :param string: string to check for numbers.
        :return: true if there are numbers in "string".
        """
        return any(char.isdigit() for char in string)

    def is_number(self, string):
        """
        Returns true if "string" is a number.
        :param string: string to check for being a number.
        :return: true if string is a number.
        """
        try:
            float(string)
            return True
        except ValueError:
            return False

    def save_index(self, path=""):
        """
        Saves the index using Python's pickle. Saves the index and the number of files as a tuple.
        """
        # Save the inverted index
        to_save = (self.index, self.file_count)
        with open(path + "inverted_index.pkl", "wb") as f:
            pickle.dump(to_save, f)

        # Save the filename to url mapping
        with open(path + "filename2url.pkl", "wb") as f:
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

                print("Processing File: ", count, " / ", self.num_files, end="\r")

                # Read the JSON file that contains the HTML
                with open(self.html_directory + filename, "r") as f:
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

        inverted_index = self.prune_index(inverted_index)

        return inverted_index

    def prune_index(self, index):
        """
        Removes keys and values that are irrelevant by being more than 3 standard deviations from the average length,
        and ignores terms that only appear in a single document.
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
            if len(key) < max_len and len(val) > self.min_occurences and not self.is_number(key):
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

    def has_numbers(string):
        return (any(char.isdigit() for char in string))


    def is_number(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    t = time.time()
    index_creator = HTML2Index("../data/pages/pages/")
    print("\nTook: ", time.time() - t, " seconds to create the index")

    inverted_index = index_creator.get_index()
    index_creator.save_index("../data/")
    key_count = len(inverted_index.keys())
    print("NUMBER OF KEYS: ", key_count)
    #input()
    num_count = 0
    pure_num_count = 0
    short_count = 0
    num_docs = 3
    for key, val in inverted_index.items():
        if has_numbers(key):
            num_count += 1
        if is_number(key):
            pure_num_count += 1
        if len(val) <= num_docs:
            short_count += 1

    print("Items with numbers: ", num_count)
    print("Items that are only numbers: ", pure_num_count)
    print("Items appearing in ", num_docs, " or less documents: ", short_count)

    input()

    from operator import itemgetter
    key_val_list = sorted([(key, len(val)) for key, val in inverted_index.items()], key=itemgetter(1))
    print("\n\n\n\n\n\n\n\n\n\n\n")

    for item in key_val_list:
        print(item)

    print("\n\n\n\n\n\n\n\n\n\n\n")
