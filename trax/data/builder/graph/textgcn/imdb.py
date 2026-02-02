# coding: utf-8
"""Build a global TextGCN graph for IMDb."""

from pathlib import Path

from resources.examples.python.base import find_project_root
from trax.data.builder.graph.textgcn.base import TextGraphConfig, build_textgcn_graph
from trax.utils import logging as trax_logging

# --- CONFIG ---
MAX_TRAIN_DOCS = 25000
MAX_TEST_DOCS = 25000
SHUFFLE_SEED = 0

MAX_VOCAB_SIZE = 20000
MIN_DF = 5
MAX_DF = 0.5
STOP_WORDS = "english"

PMI_WINDOW_SIZE = 20
VAL_RATIO = 0.1

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
OUTPUT_FILE = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "gnn"
    / "graph_imdb_textgcn_sparse.npz"
)


def main():
    config = TextGraphConfig(
        dataset_name="imdb",
        dataset_config=None,
        train_split="train",
        test_split="test",
        text_field="text",
        label_field="label",
        max_train_docs=MAX_TRAIN_DOCS,
        max_test_docs=MAX_TEST_DOCS,
        shuffle_seed=SHUFFLE_SEED,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=STOP_WORDS,
        pmi_window_size=PMI_WINDOW_SIZE,
        val_ratio=VAL_RATIO,
    )
    trax_logging.info("%s", OUTPUT_FILE, stdout=True)
    build_textgcn_graph(config, OUTPUT_FILE)


if __name__ == "__main__":
    main()
