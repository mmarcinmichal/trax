# coding=utf-8
"""Simple Graph Neural Network models for Trax.

This module provides minimal building blocks for graph neural networks with
basic functionality like adjacency normalization, optional self-loops and a
light-weight graph attention layer.
"""

from jax import nn

from trax import layers as tl
from trax.fastmath import numpy as jnp


def normalize_adjacency(adj, add_self_loops=True, eps=1e-8):
    """Returns normalized ``adj`` applying ``D^-1/2 (A + I) D^-1/2``.

    Args:
        adj: ``(..., N, N)`` adjacency matrices, optionally batched.
        add_self_loops: Whether to add identity connections before normalizing.
        eps: Small constant for numerical stability.

    Returns:
        Normalized adjacency matrices with the same shape as ``adj``.
    """
    if add_self_loops:
        eye = jnp.eye(adj.shape[-1])
        eye = jnp.broadcast_to(eye, adj.shape)
        adj = adj + eye
    deg = jnp.sum(adj, axis=-1)
    inv_sqrt_deg = 1.0 / jnp.sqrt(deg + eps)
    norm = adj * inv_sqrt_deg[..., None] * inv_sqrt_deg[..., None, :]
    return norm


def GraphConv(out_dim, activation=tl.Relu, add_self_loops=True):
    """Returns a graph convolution layer using normalized adjacency.

    The layer expects inputs ``(node_features, adjacency_matrix)`` and
    returns ``(new_features, adjacency_matrix)`` so that multiple graph
    convolution layers can be chained.

    Args:
      out_dim: Size of the output node representation.
      activation: Activation layer constructor applied after the dense step.

    Returns:
      A :class:`~trax.layers.Serial` layer implementing graph convolution.
    """

    def _conv(f, a):
        a_norm = normalize_adjacency(a, add_self_loops=add_self_loops)
        return jnp.matmul(a_norm, f)

    return tl.Serial(
        tl.Branch(
            tl.Serial(
                tl.Fn("Aggregate", _conv, n_out=1),
                tl.Dense(out_dim),
                activation(),
            ),
            tl.Select([1]),  # Pass adjacency unchanged.
        )
    )


def GraphConvNet(hidden_sizes=(16, 2), activation=tl.Relu):
    """Baseline graph neural network built from :func:`GraphConv` layers."""
    layers = []
    for size in hidden_sizes[:-1]:
        layers.append(GraphConv(size, activation=activation))
    layers.append(GraphConv(hidden_sizes[-1], activation=tl.Serial))
    return tl.Serial(*layers)


def GraphAttentionConv(out_dim, num_heads=1, activation=tl.Relu):
    """Graph convolution with attention akin to GAT."""

    def _attention(q, k, v, a):
        q = q.reshape((q.shape[0], q.shape[1], num_heads, out_dim))
        k = k.reshape((k.shape[0], k.shape[1], num_heads, out_dim))
        v = v.reshape((v.shape[0], v.shape[1], num_heads, out_dim))
        logits = jnp.einsum("bnhd,bmhd->bhnm", q, k) / jnp.sqrt(out_dim)
        mask = (a > 0).astype(jnp.float32)
        logits = logits - 1e9 * (1.0 - mask[:, None, :, :])
        attn = nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape((out.shape[0], out.shape[1], num_heads * out_dim))
        return out

    return tl.Serial(
        tl.Branch(
            tl.Serial(
                tl.Select([0, 0, 0, 1]),
                tl.Parallel(
                    tl.Dense(out_dim * num_heads),
                    tl.Dense(out_dim * num_heads),
                    tl.Dense(out_dim * num_heads),
                    None,
                ),
                tl.Fn("GAT", _attention, n_out=1),
                tl.Dense(out_dim),
                activation(),
            ),
            tl.Select([1]),
        )
    )


def GraphAttentionNet(hidden_sizes=(16, 2), activation=tl.Relu, num_heads=1):
    """Stack of :func:`GraphAttentionConv` layers for small graphs."""
    layers = []
    for size in hidden_sizes[:-1]:
        layers.append(
            GraphAttentionConv(size, num_heads=num_heads, activation=activation)
        )
    layers.append(
        GraphAttentionConv(hidden_sizes[-1], num_heads=num_heads, activation=tl.Serial)
    )
    return tl.Serial(*layers)


def GraphEdgeConv(node_out_dim, edge_out_dim, activation=tl.Relu, add_self_loops=True):
    """Graph layer updating both node and edge features."""

    def _prep(nodes, edges, adj):
        adj_norm = normalize_adjacency(adj, add_self_loops=add_self_loops)
        n_i = nodes[:, :, None, :]
        n_j = nodes[:, None, :, :]
        n_i = jnp.broadcast_to(n_i, edges.shape[:-1] + (nodes.shape[-1],))
        n_j = jnp.broadcast_to(n_j, edges.shape[:-1] + (nodes.shape[-1],))
        msg = jnp.concatenate([n_i, n_j, edges], axis=-1)
        agg = jnp.einsum("bij,bijd->bid", adj_norm, msg)
        node_in = jnp.concatenate([nodes, agg], axis=-1)
        return node_in, msg, adj

    return tl.Serial(
        tl.Fn("Prepare", _prep, n_out=3),
        tl.Parallel(tl.Dense(node_out_dim), tl.Dense(edge_out_dim), None),
        tl.Parallel(activation(), activation(), None),
    )


def GraphEdgeNet(
    node_sizes=(16, 2), edge_sizes=(4, 2), activation=tl.Relu, add_self_loops=True
):
    """Stack of :func:`GraphEdgeConv` layers with edge updates."""
    if len(node_sizes) != len(edge_sizes):
        raise ValueError("node_sizes and edge_sizes must match length")
    layers = []
    for n_size, e_size in zip(node_sizes[:-1], edge_sizes[:-1]):
        layers.append(
            GraphEdgeConv(
                n_size,
                e_size,
                activation=activation,
                add_self_loops=add_self_loops,
            )
        )
    layers.append(
        GraphEdgeConv(
            node_sizes[-1],
            edge_sizes[-1],
            activation=tl.Serial,
            add_self_loops=add_self_loops,
        )
    )
    return tl.Serial(*layers)
