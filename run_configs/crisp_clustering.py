#!/usr/bin/env python3
"""
CRISP domain task adaptation https://arxiv.org/abs/2410.03735
Multi-language clustering with shared roots and per-cluster config.

Creates:
- 32-clusters/
    config.json
    0/lang.jsonl.gz
    ...
- 1024-clusters/
    config.json
    c0/c0/lang.jsonl.gz
    ...
"""

import os, gzip, json, glob, random
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

# Paths
TRAIN_GLOB = '/dccstor/sdg/meet/data/DDS/bge-m3/bge-m3-langwise/merged_*.jsonl.gz'
VAL_GLOB   = '/dccstor/sdg/meet/data/DDS/bge-m3/MLDR/*.jsonl'

ROOTS = {
    '32-clusters': {'depth': 1},
    '1024-clusters': {'depth': 2}
}

# Clustering hyperparams (same as before)
K = 32
SAMPLE_SIZE = 10000
STEPS = 10
BALANCING_RATIO = 1.5 / K
SEED = 42
DEPTH = 2
BATCH_SIZE = 1024
random.seed(SEED)
np.random.seed(SEED)


class TreeNode:
    def __init__(self, indices, depth_idx, path):
        """
        indices: list/np.array of training-example indices assigned to this node
        depth_idx: 0-based depth index (0 = root)
        path: tuple/list of path numbers from root until this node (digits 0..K-1)
        """
        self.indices = np.array(indices, dtype=int)
        self.depth_idx = depth_idx
        self.path = tuple(map(str, path))
        self.children = None   # list of TreeNode or None
        self.centroids = None  # shape (K, dim) after clustering (normalized centroids)



def sample_indices(indices, sample_size):
    n = len(indices)
    if n <= sample_size:
        return indices.copy()
    return np.random.choice(indices, size=sample_size, replace=False)

def fit_node_kmeans(X, node_indices, k=K, sample_size=SAMPLE_SIZE, steps=STEPS):
    """
    EM-like iterative fitting for a node:
      - repeatedly sample small batches of node points and update MiniBatchKMeans.
      - initialize with k-means++ by calling one full fit on first sample (n_init=1).
    Returns fitted centroids (normalized).
    """
    if len(node_indices) == 0:
        return None

    # Normalize the data for kmeans
    Xnode = X[node_indices]
    Xnode = normalize(Xnode, norm='l2', axis=1)

    # If node has fewer than k points, create centroids from available points (pad by duplication)
    if Xnode.shape[0] <= k:
        centroids = np.zeros((k, Xnode.shape[1]), dtype=float)
        for i in range(k):
            centroids[i] = Xnode[i % Xnode.shape[0]]
        # ensure normalized
        centroids = normalize(centroids, axis=1)
        return centroids

    # initialize MiniBatchKMeans
    mbk = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=min(BATCH_SIZE, sample_size),
                          max_no_improvement=10, verbose=False, random_state=SEED)
    # first fit on an initial sample to initialize with k-means++
    init_sample_idx = sample_indices(node_indices, min(sample_size, len(node_indices)))
    mbk.partial_fit(normalize(X[init_sample_idx], axis=1))

    # then perform several iterative partial_fits on fresh samples (EM-like)
    for step in range(steps):
        sidx = sample_indices(node_indices, min(sample_size, len(node_indices)))
        mbk.partial_fit(normalize(X[sidx], axis=1))
    centroids = mbk.cluster_centers_
    centroids = normalize(centroids, axis=1)
    return centroids

def assign_to_children(X, node, centroids):
    """
    Assign ALL indices in node.indices to nearest centroid.
    Return array of shape (len(node.indices),) with child ids [0..K-1].
    """
    if centroids is None:
        return np.array([], dtype=int)
    # compute distances in normalized space: first normalize points
    Xnode = normalize(X[node.indices], axis=1)
    # distances = 1 - dot (since normalized) -> equivalent to cosine; but using Euclidean (normed) is fine:
    # we'll use Euclidean on normalized vectors
    # compute squared distances efficiently:
    # (a-b)^2 = a^2 + b^2 - 2a.b ; a^2 and b^2 are 1 for normalized, so distance^2 = 2 - 2 * (a.b)
    dots = np.dot(Xnode, centroids.T)  # shape (n_node, K)
    dists = 2.0 - 2.0 * dots
    child_ids = np.argmin(dists, axis=1)
    return child_ids

def balance_children_assignments(assignments, node_indices):
    """
    assignments: array of child ids for node.indices
    node_indices: node.indices array
    Balancing: if a child's size fraction > BALANCING_RATIO, move random items to smallest cluster
    Strategy: while any > allowed_max, pick largest cluster L and smallest S and move m items from L to S,
    where m = min(size_L - allowed_max, size_S) (or at least 1).
    """
    n = len(assignments)
    if n == 0:
        return assignments
    allowed_max = int(np.ceil(BALANCING_RATIO * n))
    # ensure not 0
    allowed_max = max(1, allowed_max)

    # build mapping child->indices positions in the assignments array
    child_to_pos = {i: np.where(assignments == i)[0].tolist() for i in range(K)}
    sizes = {i: len(child_to_pos[i]) for i in range(K)}
    # while loop to reduce oversized clusters
    loop_guard = 0
    while True:
        loop_guard += 1
        if loop_guard > 10000:
            break
        # find largest and smallest
        largest = max(sizes.items(), key=lambda x: x[1])
        smallest = min(sizes.items(), key=lambda x: x[1])
        if largest[1] <= allowed_max:
            break
        if largest[1] == 0:
            break
        li = largest[0]; si = smallest[0]
        # compute how many to move
        to_move = min(largest[1] - allowed_max, max(1, smallest[1]))
        # choose positions to move from largest
        pos_list = child_to_pos[li]
        if len(pos_list) == 0:
            break
        sel = random.sample(pos_list, min(to_move, len(pos_list)))
        # move them: set their assignment to si
        for p in sel:
            assignments[p] = si
            # update lists
            child_to_pos[li].remove(p)
            child_to_pos[si].append(p)
        sizes[li] = len(child_to_pos[li])
        sizes[si] = len(child_to_pos[si])
    return assignments


def read_lang_texts(path):
    texts = []
    for line in gzip.open(path, 'rt'):
        # texts.append(json.loads(line).get('query',""))
        texts.append(json.loads(line))
    return texts

def read_val_texts(glob_pattern):
    texts=[]
    for f in glob.glob(glob_pattern):
        for line in open(f, 'rt'):
            texts.append(json.loads(line).get('query',""))
    return texts

def traverse_tree(emb, root, max_depth):
    """
    Route embedding `emb` down the tree and return a tuple of length `max_depth`
    representing the child ids at each level (0-based).
    """
    v = emb / (np.linalg.norm(emb) + 1e-12)
    node = root
    path = []
    # iterate exactly max_depth times (levels)
    for level in range(max_depth):
        # if node has no centroids, choose 0
        if node.centroids is None:
            cid = 0
        else:
            cid = int(np.argmax(np.dot(node.centroids, v)))
        path.append(str(cid))
        # move to child if available, else stop early (pad remains won't happen since we iterate fixed times)
        if node.children is None:
            # no children: remaining levels will be zeros (or we break)
            # safer to break to reflect structure; break and return current path length <= max_depth
            break
        node = node.children[cid]
    return tuple(path)

# ---------------- Main Loop ----------------
if __name__ == "__main__":
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    val_texts = read_val_texts(VAL_GLOB)
    val_emb = model.encode(val_texts, show_progress_bar=True, convert_to_numpy=True)

    # Store config entries per root
    configs = {root: [] for root in ROOTS}

    for train_path in tqdm(glob.glob(TRAIN_GLOB)):
        lang = Path(train_path).stem.replace('merged_','')
        print(f"Language: {lang}", flush=True)
        lang_texts_obj = read_lang_texts(train_path)
        lang_texts = [x.get('query', "") for x in lang_texts_obj]
        n = len(lang_texts)
        sample_idxs = np.random.choice(n, size=min(n, SAMPLE_SIZE), replace=False)
        sample_texts = [lang_texts[i] for i in sample_idxs]
        X_sample = model.encode(sample_texts, show_progress_bar=True, convert_to_numpy=True)
        
        root = TreeNode(indices=np.arange(len(sample_texts)), depth_idx=0, path=())
        queue = deque([root])
        all_nodes = defaultdict(list)
        all_nodes[0].append(root)

        while queue:
            print(len(queue), flush=True)
            node = queue.popleft()
            depth = node.depth_idx
            if depth >= DEPTH:
                continue
            centroids = fit_node_kmeans(X_sample, node.indices, k=K, sample_size=SAMPLE_SIZE, steps=STEPS)
            node.centroids = centroids
            assignments = assign_to_children(X_sample, node, centroids)
            assignments = balance_children_assignments(assignments, node.indices)
            node.children = []
            for cid in range(K):
                idxs = node.indices[assignments == cid]
                child = TreeNode(idxs, depth_idx=depth+1, path=node.path + (cid,))
                node.children.append(child)
                all_nodes[depth+1].append(child)
                if child.depth_idx < DEPTH:
                    queue.append(child)

        # compute validation weights
        leaf_weights = defaultdict(int)
        for ve in val_emb:
            leaf = traverse_tree(ve, root, max(ROOTS[r]['depth'] for r in ROOTS))
            leaf_weights[leaf] += 1

        # embed full lang
        lang_emb = model.encode(lang_texts, show_progress_bar=True, convert_to_numpy=True)

        # route all lang texts and save per root and path
        for root_name, info in ROOTS.items():
            d = info['depth']
            base = root_name
            os.makedirs(base, exist_ok=True)
            assignments = defaultdict(list)
            for idx, emb in enumerate(lang_emb):
                leaf = traverse_tree(emb, root, d)
                assignments[leaf].append(lang_texts_obj[idx])
            # save
            for leaf, texts in assignments.items():
                folder = os.path.join(base, *leaf)
                os.makedirs(folder, exist_ok=True)
                fname = f"{lang}"
                with open(os.path.join(folder, fname), 'wt') as f:
                    for t in texts:
                        f.write(json.dumps(t)+"\n")
                rel = "./" + "/".join(list(leaf) + [fname])
                configs[base].append({
                    "name": rel,
                    "lines": len(texts),
                    "weight": leaf_weights.get(leaf, 0)
                })

    # after all languages, write config files
    for base in ROOTS:
        with open(os.path.join(base, "config.json"), 'w') as f:
            json.dump(configs[base], f, indent=2)

    print("All done.")
