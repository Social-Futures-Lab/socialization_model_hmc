from collections import defaultdict
import numpy as np

"""## Data Model and Preprocessing Funcs ##"""

def read2D(filename: str) -> list[list[int]]:
    with open(filename, 'r') as f:
        num_rows = int(f.readline().strip())
        row_sizes = [int(f.readline().strip()) for _ in range(num_rows)]

        v = [[] for _ in range(num_rows)]
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            if line.strip():
                v[i] = [int(x) for x in line.split()[:row_sizes[i]]]
    return v

def read1D(filename: str) -> list[int]:
    with open(filename, 'r') as f:
        size = int(f.readline().strip())
        v = [int(x) for line in f for x in line.split()][:size]
    return v

def write1D(filename: str, v: list[int]) -> None:
    with open(filename, 'w') as f:
        f.write(f"{len(v)}\n")
        f.write(" ".join(str(x) for x in v) + "\n")

def write2D(filename: str, v: list[list[int]]) -> None:
    with open(filename, 'w') as f:
        f.write(f"{len(v)}\n")
        for row in v:
            f.write(f"{len(row)}\n")
        for row in v:
            f.write(" ".join(str(x) for x in row) + "\n")

def write3D(filename: str, v: list[list[list[int]]]) -> None:
    with open(filename, 'w') as f:
        f.write(f"{len(v)}\n")
        for matrix in v:
            f.write(f"{len(matrix)}\n")
            for row in matrix:
                f.write(f"{len(row)}\n")
        for matrix in v:
            for row in matrix:
                f.write(" ".join(str(x) for x in row) + "\n")

def flatten_docs(docs):
    """
    docs: np.ndarray of shape (M, L)
          docs[i, j] = word id of jth word in doc i

    Returns:
        w   : (N,) word indices
        doc : (N,) document indices
    """
    M, L = docs.shape

    w = docs.reshape(-1)
    doc = np.repeat(np.arange(M), L)

    return w, doc

def flatten_docs_ragged(docs):
    """
    docs: list of lists (or np.ndarray of shape (M, L))
          docs[i][j] = word id of jth word in doc i (must be integer-valued)

    Returns:
        w   : (N,) int word indices
        doc : (N,) int document indices

    Raises:
        ValueError if any element in docs is not integer-valued
    """
    int_docs = []
    for i, row in enumerate(docs):
        row_arr = np.asarray(row)
        if not np.issubdtype(row_arr.dtype, np.integer):
            if np.all(row_arr == row_arr.astype(int)):
                row_arr = row_arr.astype(int)
            else:
                raise ValueError(f"Row {i} contains non-integer values: {row_arr}")
        int_docs.append(row_arr)

    w   = np.concatenate(int_docs).astype(int)
    doc = np.repeat(np.arange(len(docs)), [len(row) for row in int_docs])
    return w, doc


def group_by_edge_count(adj_list):
    """
    Parameters
    ----------
    adj_list : list of lists
        adj_list[i] is a list of edge ids for document i

    Returns
    -------
    out : list of dicts
        Each dict has:
            "num_edges" : int
            "doc_ids"   : np.ndarray shape (n_docs_in_group,)
            "edges"     : np.ndarray shape (n_docs_in_group, num_edges)
    """
    # Bucket doc indices by edge count
    buckets = defaultdict(list)
    for doc_id, edges in enumerate(adj_list):
        k = len(edges)
        if k > 0:
            buckets[k].append(doc_id)

    out = []
    for k, doc_ids in buckets.items():
        doc_ids_arr = np.array(doc_ids, dtype=int)
        edges_mat = np.array([adj_list[i] for i in doc_ids], dtype=int)  # shape (n_docs, k)

        out.append({
            "num_edges": k,
            "doc_ids": doc_ids_arr,
            "edges": edges_mat
        })

    return out

def has_edges_bool_mask(adj_list):
    """
    Returns a boolean mask indicating which docs have ≥1 edge.

    Parameters
    ----------
    adj_list : list of lists

    Returns
    -------
    np.ndarray of shape (num_docs,), dtype=bool
    """
    return np.array([len(edges) > 0 for edges in adj_list], dtype=bool)

def build_edge_mask(adj_list, num_src_subs):
    """
    Convert adjacency list into dense edge mask.

    Parameters
    ----------
    adj_list : list of lists
        adj_list[d] = list of source indices connected to doc d
    num_src_subs : int
        total number of possible source nodes

    Returns
    -------
    np.ndarray of shape (num_docs, num_src_subs), dtype=float
    """
    num_docs = len(adj_list)
    mask = np.zeros((num_docs, num_src_subs), dtype=float)

    for d, edges in enumerate(adj_list):
        if edges:
            mask[d, edges] = 1.0

    return mask
    
def calc_vocab_size(text_network):
    vocab_size = -1;
    cur_row = 0;
    for row in text_network.src_blobs:
        if len(row) > 0:
            row_max = max(row)
            vocab_size = max(row_max, vocab_size)
    for row in text_network.tgt_blobs:
        if len(row) > 0:
            row_max = max(row)
            vocab_size = max(row_max, vocab_size)
    vocab_size += 1
    return vocab_size

