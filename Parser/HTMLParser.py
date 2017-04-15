# ======== IMPORTS ========

# For parsing the HTML
from bs4 import BeautifulSoup

# To construct the bag of words representation of each paper
from collections import defaultdict

# For the punctuation list
import string

# =========================


class HTMLParser:

    def __init__(self):
        """
        Creates a HTMLParser object, which can quickly parse HTML files. Will also tokenize queries.
        """
        self.stopwords = self.read_stopwords()

        # This will be used to remove punctuation from strings
        self.translator = str.maketrans("", "", string.punctuation)

    def process_word(self, word):
        """
        Takes a word the comes from a webpage or query and preprocesses it by removing punctuation and translating to
        lowercase.
        :param word: the word to preprocess, as a string.
        :return: the word in lowercase with punctuation removed.
        """
        # Sanity check
        assert(type(word) is str)

        # Remove punctuation
        word_no_punctuation = word.translate(self.translator)

        # Lowercase word
        word_lower = word_no_punctuation.lower()

        return word_lower

    def read_stopwords(self):
        """
        Reads a list of stopwords from a text file and turns it into a set.
        :return: a set of stopwords with no punctuation
        """
        stopwords = []

        with open("stopwords_no_punct.txt", "r") as f:
            for row in f.readlines():
                stopwords.append(row.strip("\n"))

        # Return as a set for fast access
        return set(stopwords)

    def parse_query(self, query):
        """
        Takes a query as a string and parses it into its constituent tokens
        :param query: the query to parse as a string.
        :return: a list of the words that make up the query.
        """

        # Sanity check
        assert(type(query) is str)

        # Split into words - this can be replaced with a more advanced function
        query_words = query.split()

        # Process each word in the query
        new_words = map(self.process_word, query_words)

        return new_words

    def parse(self, filename, bag_of_words=False):
        """
        Parses an HTML file to return various features about it.
        :param filename: the name or path from current working directory to the HTML file to parse.
        :param bag_of_words: if true, return a bag of words representation of the HTML file as well.
        :return: a dictionary of information about the HTML page.
        """

        # Sanity check
        assert(type(filename) is str)

        # Read in the file
        html = self.read_file(filename)

        # Use Beautiful Soup to parse the file
        soup = BeautifulSoup(html, "lxml")

        # Get the title of the HTML file
        title = soup.title.string

        # Get all of the links in the HTML file
        links = soup.find_all("a")

        # Get specific URLs from links
        urls = []
        for link in links:
            urls.append(link.get("href"))

        # Get the raw text of the page
        text = soup.get_text()

        # Create the dictionary which will hold the information parsed from the HTML file
        file_info = {
            "filename": filename,
            "title": title,
            "links": links,
            "urls": urls,
            "file_text": text
        }

        # Optionally create a bag of words representation from the text
        if bag_of_words:
            b_o_w = self.create_bag_of_words(text)
            file_info["bag_of_words"] = b_o_w

        return file_info

    def create_bag_of_words(self, text):
        """
        Creates a bag of words representation of an HTML page.
        :param text: the text of an html page, not containing any markup.
        :return: a dictionary represention the bag of words representation of the text.
        """

        # Sanity check
        assert(type(text) is str)

        # Split the text into words
        words = text.split()

        # The bag of words will be defaultdict
        bag_of_words = defaultdict(int)

        # Create the bag of words
        for word in words:
            bag_of_words[word] += 1

        return bag_of_words

    def read_file(self, filename):
        """
        Reads a local HTML file.
        :param filename: the name or path from current working directory to the HTML file to parse.
        :return: the contents of the HTML file as a string.
        """
        with open(filename, "r") as f:
            html = f.read()

        return html


# ================ MAIN ================

if __name__ == '__main__':

    parser = HTMLParser()

    info = parser.parse("HTML/index.html")

    for key, val in info.items():
        print("KEY: ", key)
        print("VAL: ", type(val))
