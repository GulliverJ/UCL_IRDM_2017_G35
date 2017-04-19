# ==== IMPORTS ====

import abc

# =================

class Ranker:
    """
    Ranker represents the interface for ranking algorithms for this project. Each ranking algorithm should extend this
    class and implement the "search" method.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def search(self, query):
        """
        Searches the locally indexed documents which match the string "query".
        :param query: a search string.
        :return: a list of document IDs, ranked according to their relevance to the query with the MOST RELEVANT
                 DOCUMENT ID FIRST, I.E. AT POSITION 0 in the list.
        """
        pass