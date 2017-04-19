import pickle
import operator
from collections import defaultdict
import math

# HITS implementation

def search(query):

    # NOTE: adjacency and index should be global!

    # Load adjacency matrix
    adjacency = pickle.load( open( "./data/adjacency.pkl", "rb"))

    # Load inverted index
    inverted_index = pickle.load( open( "./data/inverted_index.pkl", "rb"))

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

    # For the base set, include outlinks and inlinks for each page
    # Also build base_adj
    base_adj = {}
    base_set = root_set

    for page in root_set:

        base_adj[page] = {'outlinks':adjacency[page]['outlinks'],'inlinks':adjacency[page]['inlinks']}

        for link in adjacency[page]['outlinks']:
            if link not in root_set:
                if link not in base_adj:
                    base_adj[link] = {'outlinks':[], 'inlinks':[page]}
                else:
                    base_adj[link]['inlinks'].append(page)

        for link in adjacency[page]['inlinks']:
            if link not in root_set:
                if link not in base_adj:
                    base_adj[link] = {'outlinks':[page], 'inlinks':[]}
                else:
                    base_adj[link]['outlinks'].append(page)

        base_set = list(set(base_set + adjacency[page]['outlinks'] + adjacency[page]['inlinks']))


    # h and a will store hub and authority scores respectively
    h = {}
    a = {}

    # Initialise hub scores
    for page in base_set:
        h[page] = 1;

    iterations = 1
    for i in range(0, iterations):
        #maxH = 0
        #maxA = 0
        norm = 0

        # Update authority score for each
        for page in base_set:
            a[page] = 0
            for inlink in adjacency[page]['inlinks']:
                if inlink in base_set:
                    a[page] = a[page] + h[inlink]
                norm += (a[page] * a[page])

        # Apply normalisation
        norm = math.sqrt(norm)
        for page in base_set:
            a[page] = a[page] / norm

        print ("Done authorities for iteration %d" % (i))

        norm = 0
        # Update hub score for each
        for page in base_set:
            h[page] = 0
            for outlink in adjacency[page]['outlinks']:
                if outlink in base_set:
                    h[page] = h[page] + a[outlink]

                norm += (h[page] * h[page])

        # normalise
        norm = math.sqrt(norm)
        for page in base_set:
            h[page] = h[page] / norm

    # Sort dictionaries to list of tuples
    sortedH = sorted(h.items(), key=operator.itemgetter(1), reverse=True)
    sortedA = sorted(a.items(), key=operator.itemgetter(1), reverse=True)

    # Convert to lists of pages
    hubs = []
    auths = []
    for i in range (0, len(sortedH)):
        hubs.append(sortedH[i][0])
        auths.append(sortedA[i][0])

    return hubs, auths
