import numpy as np
import scipy.sparse as mat

def transProb(Adj):
    line_sum = Adj.sum(axis=1)
    for i in range(len(line_sum)):
        if line_sum[i] == 0:
            line_sum[i] = 1
        else: 
            line_sum[i] = 1/line_sum[i]
    P = Adj.transpose().multiply(line_sum).transpose()
    return P

def damping(P,d):
    n = P.shape[0]
    P = np.multiply(P,d)
    P = P.todense()
    P += (1-d)/n
    return P

def preprocess(Adj,d):
    print('damping')
    P = damping(transProb(Adj),d)
    print('getting eigenvalues')
    a,v = np.linalg.eig(P)
    print('preprocessed')
    return (a,P)

def run(a,P,threshold):
    delta = 1
    while delta>threshold:
        a_prev =  a
        a = np.matmul(a,P)
        delta = np.linalg.norm(a-a_prev)
        print(delta)
    return a
        
def getAdjacencyMatrix(data):
    return mat.load_npz(data)

def PageRank(data,d,threshold):
    print('Getting Adjacency Matrix')
    Adj = getAdjacencyMatrix(data)
    print('Preprocessing')
    (a,P) = preprocess(Adj,d)
    print('running')
    print(a.shape)
    print(P.shape)
    a = run(a,P,threshold)
    return a
