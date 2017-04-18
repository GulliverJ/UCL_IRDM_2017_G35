import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict
from HTMLParser import HTMLParser
from math import sqrt

directory = "json_test/"
parser = HTMLParser()
word_counts = defaultdict()


def process_html_file(html, ignore_stopwords=True):

    # Sanity check
    assert (type(html) is str)

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
        new_word = parser.process_word(word)

        # Check if it is a stopword, if so don't add it to the bag of words
        if ignore_stopwords and new_word in parser.stopwords:
            continue

        bag_of_words[new_word] += 1

    return bag_of_words


def similarity(file1, file2):
    dot_prod, file1_norm, file2_norm = 0.0, 0.0, 0.0
    for word, count1 in word_counts[file1].items():
        count2 = word_counts[file2][word]
        dot_prod += count1 * count2
        file1_norm += count1 ** 2
        file2_norm += count2 ** 2
    return dot_prod / (sqrt(file1_norm) * sqrt(file2_norm))

for filename in os.listdir(directory):
    if filename.endswith(".json"):

        with open(directory + "/" + filename, "r") as f:
            html_dict = json.load(f)

        html = html_dict["html"]

        html_bag_of_words = process_html_file(html)

        word_counts[filename] = html_bag_of_words

for file1, bag1 in word_counts.items():
    for file2, bag2, in word_counts.items():
        x = similarity(file1, file2)
        print("{0:.3f}".format(x), end=" ")
    print()
