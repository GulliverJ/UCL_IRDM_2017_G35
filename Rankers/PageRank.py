import numpy as np
import pickle
from Ranker import Ranker
from collections import defaultdict

ADJACENCY_FILE = "data/adjacency.pkl"
PAGERANK_SCORES = "data/pagerank.pkl"
INVERTED_INDEX = "data/inverted_index.pkl"

class PageRank(Ranker):

    
    
    def transProb(self,Adj):
        """
        Changes the adjacency matric into a transition probability matrix
        """
        line_sum = Adj.sum(axis=1)
        for i in range(len(line_sum)):
            if line_sum[i] == 0:
                line_sum[i] = 1
            else: 
                line_sum[i] = 1/line_sum[i]
        P = Adj*line_sum[:,np.newaxis]
        return P

    def damping(self,P,d):
        """
        Applies the damping factor
        """
        n = P.shape[0]
        P = np.multiply(P,d)
        P += (1-d)/n
        return P

    def preprocess(self,Adj,d):
        """
        Runs the preprocessing stage of pagerank (ie getting the damped TPM
        and initial pagerank scores (from the eigenvalues)
        """
        print('Damping')
        P = self.damping(self.transProb(Adj),d)
        print('getting eigenvalues')
        a,v = np.linalg.eig(P)
        print('preprocessed')
        return (a,P)

    def run(a,P,threshold):
        """
        Runs the pagerank algorithm until the difference between
        the iterations is below the threshold value, set by default
        as 0.001
        """
        delta = 1
        while delta>threshold:
            a_prev =  a
            a = np.matmul(a,P)
            delta = np.linalg.norm(a-a_prev)
            print(delta)
        return a
            
    def getAdjacencyMatrix(self,data):
        """
        Gets the adjacency matrix by parsing the pickle file containing
        the inlinks and outlinks
        """
        size = len(data)
        Adj = np.zeros((size,size),dtype=bool)
        for i in range(size):
            for link in data[i]["outlinks"]:
                Adj[i,int(link)] = True
        print(size)
        return Adj

    def compute(self,d = 0.85,threshold = 0.001):
        """
        The method used to run the pagerank algorithm and compute the scores
        which can be applied at search time.
        :param d: damping factor, set by default to 0.85 as recommended
        by Page
        :param threshold: the threshold value used to stop iterating,
        a considered acceptable amount of convergence
        :return PageRank scores
        """
        data = pickle.load(open(ADJACENCY_FILE,"rb"))
        print('Getting Adjacency Matrix')
        Adj = self.getAdjacencyMatrix(data)
        print('Preprocessing')
        (a,P) = self.preprocess(Adj,d)
        print('running')
        print(a.shape)
        print(P.shape)
        a = run(a,P,threshold)
        pickle.dump(a,open(PAGERANK_SCORES,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        return a

    def getBaseSet(self,query):
        """
        Used in order to obtain the relevant results, which will next be ranked
        """
        # Load inverted index
        inverted_index = pickle.load( open( INVERTED_INDEX, "rb"))

        term_to_files = defaultdict()

        for term, entries in inverted_index[0].items():
            files_containing_term = []
            for entry in entries:
                files_containing_term.append(entry[0])
            term_to_files[term] = files_containing_term
        # Tokenise the query and get a list of filenames for the root set
        tokens = query.split()
        root_set_str = []
        for token in tokens:
            root_set_str = list(set(root_set_str + term_to_files[token]))

        # Convert the filenames to ints
        root_set = []
        for i in range(0, len(root_set_str)):
            root_set.append(int(root_set_str[i].strip(".json")))
        return root_set        

    def search(self,query):
        """
        Used to rank the query results by using the precompiled scores
        :param query: The query string
        :return Ranked order of relevant results
        """
        scores = pickle.load(open(PAGERANK_SCORES,"rb"))
        base_set = self.getBaseSet(query)
        base_score = []
        for page in base_set:
            base_score.append(int(scores[page]))
        ranked = [x for (y,x) in sorted(zip(base_scores,base_set))]
        return ranked












            
            

