# coding: utf-8
"""Reusable TextGCN graph builder for text classification datasets."""

from __future__ import annotations

import math

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from data.preprocessing.raw.graph.textgcn.base import preprocess_newsgroups_textgcn
from datasets import Dataset, DatasetDict, load_dataset
from scipy.sparse import coo_matrix, diags, eye
from sklearn.feature_extraction.text import TfidfVectorizer

import trax.data.loader.raw.base


@dataclass(frozen=True)
class TextGraphConfig:
    dataset_name: str
    is_local: bool = False
    dataset_config: Optional[str] = None
    train_split: str = "train"
    test_split: Optional[str] = "test"
    text_field: str = "text"
    label_field: str = "label"
    max_train_docs: Optional[int] = 25000
    max_test_docs: Optional[int] = 25000
    shuffle_seed: Optional[int] = 0
    max_vocab_size: Optional[int] = 20000
    min_df: int = 5
    max_df: float = 0.5
    stop_words: Optional[str] = "english"
    pmi_window_size: int = 20
    val_ratio: float = 0.1


def load_texts_labels(
    config: TextGraphConfig,
) -> Tuple[list, np.ndarray, list, np.ndarray]:

    test_ds = None

    if config.is_local:
        train_texts, train_labels = trax.data.loader.raw.base.load_dataset(
            config.dataset_name, config.train_split
        )
        test_texts, test_labels = trax.data.loader.raw.base.load_dataset(
            config.dataset_name, config.test_split
        )

        dataset = DatasetDict(
            train=Dataset.from_dict({"text": train_texts, "label": train_labels}),
            test=Dataset.from_dict({"text": test_texts, "label": test_labels}),
        )

        train_ds = dataset["train"]
        test_ds = dataset["test"]
    else:
        train_ds = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.train_split,
        )

        if config.test_split is not None:
            test_ds = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split=config.test_split,
            )

    if config.shuffle_seed is not None:
        train_ds = train_ds.shuffle(seed=config.shuffle_seed)
        if test_ds is not None:
            test_ds = test_ds.shuffle(seed=config.shuffle_seed)

    if config.max_train_docs is not None:
        train_ds = train_ds.select(range(min(config.max_train_docs, len(train_ds))))
    if test_ds is not None and config.max_test_docs is not None:
        test_ds = test_ds.select(range(min(config.max_test_docs, len(test_ds))))

    texts_train = list(train_ds[config.text_field])
    y_train = np.array(train_ds[config.label_field], dtype=np.int64)

    if test_ds is None:
        texts_test = []
        y_test = np.array([], dtype=np.int64)
    else:
        texts_test = list(test_ds[config.text_field])
        y_test = np.array(test_ds[config.label_field], dtype=np.int64)

    return texts_train, y_train, texts_test, y_test


def build_tfidf(texts_all, config: TextGraphConfig):
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,          # już zrobione w clean_str_textgcn
        max_features=config.max_vocab_size,
        min_df=1,                 # min_df kontrolujemy przez min_freq w remove_words
        max_df=1.0,
        stop_words=None,          # stopwords już odfiltrowaliśmy wcześniej
        norm=None,                # jak w build_graph.py (brak normalizacji TF-IDF)
        smooth_idf=False,         # idf = log(N / df), bez +1
    )
    X_tfidf = vectorizer.fit_transform(texts_all)
    vocab = vectorizer.vocabulary_
    analyzer = vectorizer.build_analyzer()  # to będzie po prostu `lambda s: s.split()`

    return X_tfidf, vocab, analyzer


def build_word_pmi_edges(texts_all, vocab, analyzer, window_size: int):
    vocab_size = len(vocab)
    word_window_counts = np.zeros(vocab_size, dtype=np.int64)
    cooc_counts = Counter()
    total_windows = 0

    for text in texts_all:
        tokens = [t for t in analyzer(text) if t in vocab]
        if not tokens:
            continue
        token_ids = [vocab[t] for t in tokens]
        n = len(token_ids)

        # === FIXED WINDOWING (TextGCN-style) ===
        if n <= window_size:
            windows = [token_ids]
        else:
            windows = [
                token_ids[i : i + window_size] for i in range(n - window_size + 1)
            ]

        for window in windows:
            unique_list = sorted(set(window))

            for wid in unique_list:
                word_window_counts[wid] += 1

            for i_idx, wi in enumerate(unique_list):
                for wj in unique_list[i_idx + 1 :]:
                    cooc_counts[(wi, wj)] += 1

            total_windows += 1

    rows_ww = []
    cols_ww = []
    data_ww = []

    for (wi, wj), cij in cooc_counts.items():
        p_ij = cij / total_windows
        p_i = word_window_counts[wi] / total_windows
        p_j = word_window_counts[wj] / total_windows
        if p_i <= 0 or p_j <= 0 or p_ij <= 0:
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        if pmi <= 0:
            continue

        rows_ww.append(wi)
        cols_ww.append(wj)
        data_ww.append(pmi)
        rows_ww.append(wj)
        cols_ww.append(wi)
        data_ww.append(pmi)

    return rows_ww, cols_ww, data_ww, vocab_size


def build_global_adj_sparse(X_tfidf, rows_ww, cols_ww, data_ww, num_docs, vocab_size):
    num_nodes = num_docs + vocab_size

    X_coo = X_tfidf.tocoo()
    rows_dw = X_coo.row
    cols_dw = X_coo.col
    data_dw = X_coo.data

    rows_dw_nodes = rows_dw
    cols_dw_nodes = num_docs + cols_dw

    rows_wd_nodes = cols_dw_nodes
    cols_wd_nodes = rows_dw_nodes
    data_wd = data_dw

    rows_ww_nodes = num_docs + np.array(rows_ww, dtype=np.int64)
    cols_ww_nodes = num_docs + np.array(cols_ww, dtype=np.int64)
    data_ww_nodes = np.array(data_ww, dtype=np.float32)

    all_rows = np.concatenate([rows_dw_nodes, rows_wd_nodes, rows_ww_nodes])
    all_cols = np.concatenate([cols_dw_nodes, cols_wd_nodes, cols_ww_nodes])
    all_data = np.concatenate(
        [data_dw.astype(np.float32), data_wd.astype(np.float32), data_ww_nodes]
    )

    adj_coo = coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj_coo.sum_duplicates()

    adj_hat = adj_coo.tocsr() + eye(num_nodes, dtype=np.float32, format="csr")
    deg = np.array(adj_hat.sum(axis=1)).reshape(-1)
    inv_sqrt_deg = 1.0 / np.sqrt(deg + 1e-8)
    d_inv_sqrt = diags(inv_sqrt_deg, offsets=0, format="csr")
    adj_norm = d_inv_sqrt @ adj_hat @ d_inv_sqrt
    adj_norm.sort_indices()

    return adj_norm


def make_splits(num_train, num_test, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)

    num_docs = num_train + num_test
    all_train_indices = np.arange(num_train)

    num_val = int(round(num_train * val_ratio))
    num_train_final = num_train - num_val

    rng.shuffle(all_train_indices)
    train_indices = all_train_indices[:num_train_final]
    val_indices = all_train_indices[num_train_final:]

    train_mask = np.zeros(num_docs, dtype=bool)
    val_mask = np.zeros(num_docs, dtype=bool)
    test_mask = np.zeros(num_docs, dtype=bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    if num_test > 0:
        test_mask[num_train:] = True

    return train_mask, val_mask, test_mask


def build_textgcn_graph(config: TextGraphConfig, output_file: Path):
    texts_train, y_train, texts_test, y_test = load_texts_labels(config)

    texts_train, texts_test, _ = preprocess_newsgroups_textgcn(
        texts_train, texts_test, min_freq=5
    )

    num_train = len(texts_train)
    num_test = len(texts_test)
    num_docs = num_train + num_test

    texts_all = texts_train + texts_test

    X_tfidf, vocab, analyzer = build_tfidf(texts_all, config)
    vocab_size = len(vocab)

    rows_ww, cols_ww, data_ww, vocab_size2 = build_word_pmi_edges(
        texts_all, vocab, analyzer, window_size=config.pmi_window_size
    )
    assert vocab_size == vocab_size2

    adj_csr = build_global_adj_sparse(
        X_tfidf,
        rows_ww,
        cols_ww,
        data_ww,
        num_docs=num_docs,
        vocab_size=vocab_size,
    )

    labels = np.zeros(num_docs, dtype=np.int64)
    labels[:num_train] = y_train
    if num_test > 0:
        labels[num_train:] = y_test

    train_mask, val_mask, test_mask = make_splits(
        num_train, num_test, val_ratio=config.val_ratio, seed=0
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        adj_data=adj_csr.data.astype(np.float32),
        adj_indices=adj_csr.indices.astype(np.int32),
        adj_indptr=adj_csr.indptr.astype(np.int64),
        adj_shape=np.array(adj_csr.shape, dtype=np.int64),
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_train=y_train,
        y_test=y_test,
    )
