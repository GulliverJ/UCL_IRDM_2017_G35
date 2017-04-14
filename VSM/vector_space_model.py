import os
import sys
import math
from collections import defaultdict


docnames = {}  # the file names of documents
N = 0  # number of documents

# for each term, store a dict of the documents containing the term
# term_to_docs structure:
#   key = term
#   value = doc_dict
# doc_dict structure:
#   key = document ID
#   value = frequency of term appearing in document
term_to_docs = defaultdict(dict)

# the vocabulary is a set of the unique words over all documents
vocab = set()

# for each document, store a dict of terms in the document
# term frequency = 1 + log( tf(term, doc) )
#   key = document ID
#   value = term_dict
# term_dict structure:
#   key = term
#   value = frequency
term_freq = defaultdict(dict)

# how many documents a term appears in
#   key = term
#   value = frequency
doc_freq = defaultdict(int)


def tokenize(doc):
    remove = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"
    terms = doc.lower().split()
    return [term.strip(remove) for term in terms]


def init_docs(path):
    global docnames
    if not os.path.isdir(path):
        print("Invalid path specified: ", path)
        return
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.txt'):
                docnames[len(docnames)] = os.path.join(root, f)


def init_tf():
    global N, vocab, term_to_docs, term_freq
    N = len(docnames)
    for doc_id in docnames:
        with open(docnames[doc_id], 'r') as f:
            doc = f.read()
        terms = tokenize(doc)
        unique_terms = set(terms)
        vocab |= unique_terms
        for term in unique_terms:
            tf = terms.count(term)  # the frequency of the term in the doc
            term_to_docs[term][doc_id] = tf
            term_freq[doc_id][term] = 1.0 + math.log(tf, 2)


def init_df():
    global doc_freq
    for term in vocab:
        doc_freq[term] = len(term_to_docs[term])


def norm_tf():
    global term_freq
    for doc_id in term_freq:
        values = []
        for value in term_freq[doc_id].values():
            values.append(value)
        norm = math.sqrt(sum([x**2 for x in values]))
        for term in term_freq[doc_id]:
            term_freq[doc_id][term] /= norm


def idf(term):
    if term in vocab:
        return math.log(N/doc_freq[term], 2)
    return 0.0


def tf_idf(term, doc_id):
    if term in vocab:
        if term in term_freq[doc_id]:
            return term_freq[doc_id][term] * idf(term)
    return 0.0


def cos_sim(query, doc_id):
    dot_prod, q_norm, d_norm, cos = 0.0, 0.0, 0.0, 0.0
    q_tf = [query.count(term) for term in query]
    q_tf_norm = math.sqrt(sum([x**2 for x in q_tf]))
    for term in query:
        q_tf_idf = query.count(term) / q_tf_norm * idf(term)
        d_tf_idf = tf_idf(term, doc_id)
        dot_prod += q_tf_idf * d_tf_idf
        q_norm += q_tf_idf ** 2
        d_norm += d_tf_idf ** 2
    q_norm = math.sqrt(q_norm)
    d_norm = math.sqrt(d_norm)
    if q_norm != 0 and d_norm != 0:
        cos = dot_prod / (q_norm * d_norm)
    return cos


def search():
    query = tokenize(input("Search for: ").lower())
    if not query:
        sys.exit()
    scores = sorted([(cos_sim(query, doc_id), doc_id) for doc_id in docnames],
                    reverse=True)
    print("Score: filename")
    for (score, doc_id) in scores:
        print(str(score) + ": " + docnames[doc_id])


if __name__ == '__main__':
    print("Running Vector Space Model")

    init_docs(path='minidocs/')
    init_tf()
    init_df()
    norm_tf()
    print("Initialization completed")

    #print(docnames)
    #print(term_freq)
    #print(doc_freq)

    while True:
        search()
