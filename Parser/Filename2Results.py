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


class Filename2Results:

    def __init__(self, json_dir, save_location=""):
        """
        An object which will turn the directory of JSON objects into a dictionary mapping filenames to dictionaries
        which will be used to display results, with each dictionary having the page title, URL and first line of text.

        :param json_dir: the directory to parse JSON files from.
        :param save_location: the place to save the filename to results mapping.
        """

        # Sanity check
        assert(type(json_dir) is str)

        self.html_directory = json_dir

        self.num_files = len([x for x in os.listdir(json_dir) if x.endswith(".json")])

        self.file_count = 0

        self.filenames2results = self.build_mapping()

        self.save_mapping(save_location)

        for key, val in self.filenames2results.items():
            print()
            print("KEY: ", key)
            print("VAL: ")
            print(val)
            input()

    def save_mapping(self, path=""):
        """
        Saves the index using Python's pickle. Saves the index and the number of files as a tuple.
        """
        with open(path + "filenames2results.pkl", "wb") as f:
            pickle.dump(self.filenames2results, f)

    def build_mapping(self):
        """
        Builds the filename to results mapping.
        :return: the mapping.
        """

        # The inverted index will be a defaultdict
        filename2results = defaultdict(list)

        # For logging
        count = 0

        # Iterate over each HTML file
        for filename in os.listdir(self.html_directory):

            if filename.endswith(".json"):

                info_dict = {}

                print("Processing File: ", count, " / ", self.num_files, end="\r")

                # Read the JSON file that contains the HTML
                with open(self.html_directory + filename, "r") as f:
                    html_dict = json.load(f)

                # Extract the HTML
                html = html_dict["html"]

                # Extract the URL
                url = html_dict["url"]

                # Add to the info-dict
                info_dict["url"] = url

                # Extract the title and first line of text from the html
                title, first_line_of_text = self.process_html(html)

                # Add to the info-dict
                info_dict["title"] = title
                info_dict["blurb"] = first_line_of_text

                # Increment the file counter
                self.file_count += 1

                # Increment the logging counter
                count += 1

                filename2results[filename] = info_dict

        return filename2results

    def process_html(self, html):
        """
        Processes an HTML string to extract the title and first line of text.
        :param html: the html string.
        :return title of the HTML and its first line of text.
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

        # Split the page text into sentences
        page_sents = page_text.split(".")

        first_line = ""

        # Get the title
        if soup.title is not None and soup.title.string is not None:

            title = soup.title.string.strip()
            if len(page_sents) > 1:
                first_line = page_sents[0].strip()

        else:
            if len(page_sents) > 0 and len(page_sents) <= 1:
                title = page_sents[0].strip()

            elif len(page_sents) > 0 and len(page_sents) <= 2:
                title = page_sents[0].strip()
                first_line = page_sents[1].strip()

            elif len(page_sents) > 0 and len(page_sents) > 2:
                title = page_sents[1].strip()
                first_line = page_sents[2].strip()

        first_line = first_line.replace("\n", "")
        first_line = first_line.replace("\t", "")

        return title, first_line


# ================ MAIN ================

if __name__ == '__main__':

    t = time.time()
    index_creator = Filename2Results(json_dir="LSI/pages/", save_location="Parser/")
    print("\nTook: ", time.time() - t, " seconds to create the index")
