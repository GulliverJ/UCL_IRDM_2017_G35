# Usage Guide

## PageRank
The PageRank algorithm requires an input of an adjacency matrix stored in a .npz file (Scipy sparse matrix).
In order to run use the following:
```
>>> pagerank_scores = PageRank(directory_of_npz, damping_factor, threshold)

```

The two parameters to the PageRank algorithm are the damping factor, recommended to be set to 0.85, and a threshold value, in order to establish the tolerence levels in the convergence.
