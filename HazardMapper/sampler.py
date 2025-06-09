""" HazardMapper - Sampler Module
=========================
Work in progress, not yet functional.
This module is designed to sample patches from a dataset based on a partition map,
extract features, build a k-NN graph, and perform community detection to identify clusters.
"""

import numpy as np
import faiss
import torch
from torch.utils.data import Dataset
import igraph as ig
from tqdm import tqdm
from HazardMapper.dataset import HazardDataset
from HazardMapper.utils import plot_npy_arrays

def get_valid_indices(partition_map: np.ndarray, split: int, patch_size: int = 5) -> list:
    H, W = partition_map.shape
    margin = patch_size // 2
    flat_valid = np.flatnonzero(partition_map == split)
    coords = np.array([(i // W, i % W) for i in flat_valid])
    mask = (coords[:, 0] >= margin) & (coords[:, 0] < H - margin) & \
           (coords[:, 1] >= margin) & (coords[:, 1] < W - margin)
    coords = coords[mask]
    return (coords[:, 0] * W + coords[:, 1]).tolist()

def extract_features(dataset: Dataset, indices: list) -> np.ndarray:
    print(f"Extracting {len(indices)} patches...")
    features = []
    for idx in tqdm(indices):
        patch = dataset[idx]
        if isinstance(patch, tuple):
            patch = patch[0]
        features.append(patch.flatten().numpy())
    return np.stack(features).astype(np.float32)

def build_knn_graph(X: np.ndarray, k: int = 10) -> list:
    print("Building k-NN graph...")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    _, knn = index.search(X, k)
    edges = [(i, j) for i, neighbors in enumerate(knn) for j in neighbors if i != j]
    return edges

def run_community_detection(edges: list) -> list:
    print("Running community detection...")
    g = ig.Graph(edges=edges, directed=False)
    return g.community_leiden()

def medoid_index(X: np.ndarray, indices: list) -> int:
    sub_X = X[indices]
    dists = np.linalg.norm(sub_X[:, None] - sub_X[None, :], axis=2)
    return indices[np.argmin(dists.sum(axis=0))]

def cluster_split(partition_map: np.ndarray,
                  dataset: Dataset,
                  split: int,
                  patch_size: int = 5,
                  max_samples: int = 300_000,
                  k: int = 10,
                  seed: int = 42) -> list:
    np.random.seed(seed)
    torch.manual_seed(seed)

    valid_indices = get_valid_indices(partition_map, split, patch_size)
    sampled_indices = np.random.choice(valid_indices, max_samples, replace=False) \
                      if len(valid_indices) > max_samples else valid_indices

    X = extract_features(dataset, sampled_indices)
    edges = build_knn_graph(X, k=k)
    communities = run_community_detection(edges)

    print(f"Split {split}: Detected {len(communities)} clusters. Selecting medoids...")
    return [sampled_indices[medoid_index(X, comm)] for comm in communities]

def create_reduced_partition_map(shape, train_indices, val_indices, test_indices):
    reduced_map = np.zeros(shape, dtype=np.uint8)
    flat_map = reduced_map.ravel()
    flat_map[train_indices] = 1
    flat_map[val_indices] = 2
    flat_map[test_indices] = 3
    return flat_map.reshape(shape)

if __name__ == "__main__":
    sizes = [1_000_000, 200_000, 200_000]
    hazard = 'landslide'

    partition_map_path = f"Input/Europe/partition_map/{hazard}_partition_map.npy"
    partition_map = np.load(partition_map_path)
    shape = partition_map.shape

    print("Loading dataset...")
    variables = ['elevation', 'slope', 'landcover', 'aspect', 'NDVI', 'precipitation_daily', 'accuflux', 'HWSD', 'GEM', 'curvature', 'GLIM']
    dataset = HazardDataset(hazard=hazard, patch_size=5, variables=variables)

    train_indices = cluster_split(partition_map, dataset, split=1, max_samples=sizes[0], k=10, seed=42)
    val_indices   = cluster_split(partition_map, dataset, split=2, max_samples=sizes[1], k=10, seed=42)
    test_indices  = cluster_split(partition_map, dataset, split=3, max_samples=sizes[2], k=10, seed=42)

    reduced_partition_map = create_reduced_partition_map(shape, train_indices, val_indices, test_indices)

    output_path = f"Input/Europe/partition_map/{hazard}_sampled_partition_map.npy"
    np.save(output_path, reduced_partition_map)
    print(f"Saved reduced partition map to {output_path}")
    plot_npy_arrays(reduced_partition_map, type='partition', name="", title=f"{hazard} sampled partition map", save_path=f"Input/Europe/partition_map/{hazard}_sampled_partition_map.png")