from HTML2Index import HTML2Index
from sys import exit


class HTMLBoolean:

    def __init__(self, index_creator):
        """
        Create an object for boolean retrieval with a HTML2Index object.
        :param index_creator: HTML2Index object.
        """
        self.inverted_index = index_creator.get_index()
        self.filenames = index_creator.get_filenames()

    def and_search(self, query):
        return_set = set(self.filenames)
        for term in query:
            term_found_in = []
            for item in self.inverted_index[term]:
                term_found_in.append(item[0])
            return_set &= set(term_found_in)
        return return_set

    def or_search(self, query):
        return_set = set()
        for term in query:
            term_found_in = []
            for item in self.inverted_index[term]:
                term_found_in.append(item[0])
            return_set |= set(term_found_in)
        return return_set

    def search(self):
        query = input("Search for: ").lower().split()
        if not query:
            exit()
        print('all terms found in:', self.and_search(query))
        print('at least one term found in:', self.or_search(query))



# ================ MAIN ================

if __name__ == '__main__':

    index_creator = HTML2Index("htmldocs/")

    print("Creating model.")
    model = HTMLBoolean(index_creator)
    print("Model created.")

    while True:
        model.search()
