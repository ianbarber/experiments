"""Sparse graph generators for benchmarks."""

import torch
import networkx as nx


def er_graph(num_nodes: int, degree: int, device: str = "cuda"):
    """Erdős–Rényi-like regular random graph (each node has ~degree neighbors).

    Returns CSR row_ptr, col_ind, edge_index.
    """
    G = nx.gnm_random_graph(num_nodes, num_nodes * degree // 2, seed=42)
    # Make bidirectional for directed CSR
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index_to_csr(edge_index, num_nodes)


def ba_graph(num_nodes: int, m: int, device: str = "cuda"):
    """Barabási–Albert scale-free graph."""
    G = nx.barabasi_albert_graph(num_nodes, m, seed=42)
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index_to_csr(edge_index, num_nodes)


def edge_index_to_csr(edge_index, num_nodes):
    """Convert [2, E] edge_index to CSR format on GPU."""
    device = edge_index.device
    src = edge_index[0]
    dst = edge_index[1]
    # Sort by dst for incoming CSR
    perm = torch.argsort(dst)
    src = src[perm]
    dst = dst[perm]

    # Build row_ptr via histogram + cumsum
    counts = torch.bincount(dst, minlength=num_nodes)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32, device=device)
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    col_ind = src.to(torch.int32)
    weights = torch.ones(col_ind.numel(), dtype=torch.float32, device=device)
    return row_ptr, col_ind, weights, edge_index
